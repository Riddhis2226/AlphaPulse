"""Week 2: Vectorized Quantitative Risk Engine (Log Returns, Rolling Volatility,
Covariance/Correlation, 10,000+ Cholesky Monte Carlo Paths, VaR/CVaR, Sharpe Ratio).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from scipy.stats import norm

from src.config import (
    PORTFOLIO_ASSETS,
    PROCESSED_DIR,
    TABLEAU_DIR,
    Settings,
    get_settings,
)

logger = logging.getLogger("alphapulse.quant")

TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.0675  # ~6.75% India 91-day T-bill / G-Sec proxy


def load_market_prices(parquet_path: Path | None = None) -> pl.DataFrame:
    """Load cleaned Week 1 market prices from local Parquet."""
    path = parquet_path or (PROCESSED_DIR / "market_prices.parquet")
    if not path.exists():
        msg = f"Missing {path}. Run Week 1 ingestion first."
        raise FileNotFoundError(msg)
    return pl.read_parquet(path).sort(["asset_id", "date"])


def build_price_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Pivot long market_prices into a date-indexed wide float32 close-price DataFrame with zero NaNs."""
    wide = (
        df.pivot(on="asset_id", index="date", values="close")
        .sort("date")
        .to_pandas()
        .set_index("date")
    )
    asset_order = [a.asset_id for a in PORTFOLIO_ASSETS if a.asset_id in wide.columns]
    return wide[asset_order].dropna(how="any").astype(np.float32)


def compute_log_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute vectorized daily log returns: ln(P_t / P_{t-1})."""
    prices_arr = price_matrix.to_numpy(dtype=np.float64)
    log_ret_arr = np.diff(np.log(prices_arr), axis=0).astype(np.float32)
    return pd.DataFrame(log_ret_arr, index=price_matrix.index[1:], columns=price_matrix.columns)


def compute_rolling_volatility(log_returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute annualized 30-day rolling volatility per asset."""
    rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    return rolling_vol.dropna().astype(np.float32)


def compute_portfolio_kpis(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float]:
    """Compute vectorized portfolio return, volatility, Sharpe, Parametric & Historical VaR/CVaR."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    ret_arr = log_returns.to_numpy(dtype=np.float64)
    mean_daily = ret_arr.mean(axis=0)
    cov_daily = np.cov(ret_arr, rowvar=False)

    # Vectorized matrix math: w^T * mu and w^T * Sigma * w
    port_daily_mean = float(w @ mean_daily)
    port_daily_var = float(w.T @ cov_daily @ w)
    port_daily_vol = float(np.sqrt(port_daily_var))

    ann_return = port_daily_mean * TRADING_DAYS
    ann_vol = port_daily_vol * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Portfolio historical daily returns via matrix-vector product
    port_hist_returns = ret_arr @ w
    alpha = 1.0 - confidence

    hist_var_1d = float(-np.percentile(port_hist_returns, alpha * 100.0))
    tail_losses = port_hist_returns[port_hist_returns <= -hist_var_1d]
    hist_cvar_1d = float(-tail_losses.mean()) if len(tail_losses) > 0 else hist_var_1d

    z_score = float(norm.ppf(alpha))
    param_var_1d = float(-(port_daily_mean + z_score * port_daily_vol))

    # Max drawdown from cumulative log returns
    cum_wealth = np.exp(np.cumsum(port_hist_returns))
    running_peak = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - running_peak) / running_peak
    max_drawdown = float(drawdowns.min())

    return {
        "annualized_return": round(ann_return, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "historical_var_95_1d": round(hist_var_1d, 6),
        "historical_cvar_95_1d": round(hist_cvar_1d, 6),
        "parametric_var_95_1d": round(param_var_1d, 6),
        "max_drawdown": round(max_drawdown, 6),
    }


def run_monte_carlo(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10000,
    horizon_days: int = 252,
    initial_capital: float = 1_000_000.0,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """Run vectorized Cholesky-correlated Monte Carlo simulations with zero Python loops."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    ret_arr = log_returns.to_numpy(dtype=np.float64)
    mean_daily = ret_arr.mean(axis=0)
    cov_daily = np.cov(ret_arr, rowvar=False)

    # Cholesky decomposition: Sigma = L @ L^T
    chol = np.linalg.cholesky(cov_daily)
    rng = np.random.default_rng(seed)

    # Shape: (horizon_days, n_simulations, n_assets)
    n_assets = ret_arr.shape[1]
    z = rng.standard_normal((horizon_days, n_simulations, n_assets), dtype=np.float64)
    correlated_shocks = z @ chol.T
    sim_asset_returns = mean_daily + correlated_shocks

    # Portfolio daily log returns across all simulations: (horizon_days, n_simulations)
    sim_port_log_ret = sim_asset_returns @ w
    cum_log_paths = np.cumsum(sim_port_log_ret, axis=0)
    portfolio_paths = (initial_capital * np.exp(cum_log_paths)).astype(np.float32)

    terminal_values = portfolio_paths[-1, :]
    terminal_returns = (terminal_values / initial_capital) - 1.0
    alpha = 1.0 - confidence
    mc_var_pct = float(-np.percentile(terminal_returns, alpha * 100.0))
    mc_var_inr = float(initial_capital * mc_var_pct)

    percentiles = np.percentile(portfolio_paths, [5, 25, 50, 75, 95], axis=1)

    return {
        "portfolio_paths": portfolio_paths,
        "terminal_values": terminal_values,
        "terminal_returns": terminal_returns.astype(np.float32),
        "mc_var_95_horizon_pct": round(mc_var_pct, 6),
        "mc_var_95_horizon_inr": round(mc_var_inr, 2),
        "expected_terminal_inr": round(float(terminal_values.mean()), 2),
        "percentile_fan": pd.DataFrame(
            {
                "day": np.arange(1, horizon_days + 1),
                "p05": percentiles[0],
                "p25": percentiles[1],
                "p50": percentiles[2],
                "p75": percentiles[3],
                "p95": percentiles[4],
            }
        ),
    }


def export_tableau_csvs(
    price_matrix: pd.DataFrame,
    log_returns: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    mc_result: dict[str, object],
) -> None:
    """Export clean flat CSVs for Tableau Public consumption."""
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

    prices_long = price_matrix.reset_index().melt(
        id_vars="date", var_name="asset_id", value_name="close_inr"
    )
    ret_long = log_returns.reset_index().melt(
        id_vars="date", var_name="asset_id", value_name="log_return"
    )
    merged = prices_long.merge(ret_long, on=["date", "asset_id"], how="left")
    merged.to_csv(TABLEAU_DIR / "daily_prices_returns.csv", index=False)

    rolling_vol.reset_index().melt(
        id_vars="date", var_name="asset_id", value_name="rolling_vol_30d"
    ).to_csv(TABLEAU_DIR / "rolling_volatility.csv", index=False)

    corr_matrix.reset_index().rename(columns={"index": "asset_a"}).melt(
        id_vars="asset_a", var_name="asset_b", value_name="correlation"
    ).to_csv(TABLEAU_DIR / "correlation_matrix.csv", index=False)

    fan_df = mc_result["percentile_fan"]
    if isinstance(fan_df, pd.DataFrame):
        fan_df.to_csv(TABLEAU_DIR / "monte_carlo_summary.csv", index=False)

    logger.info("Exported Tableau CSVs to %s", TABLEAU_DIR)


def save_quant_to_mongodb(
    kpis: dict[str, float],
    corr_matrix: pd.DataFrame,
    mc_result: dict[str, object],
    settings: Settings,
) -> None:
    """Persist quantitative KPIs, correlation matrix, and simulation summary to MongoDB."""
    try:
        client: MongoClient = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except PyMongoError as exc:
        logger.warning("MongoDB unavailable (%s); skipping Week 2 MongoDB save.", exc)
        return

    db = client[settings.mongo_db]
    run_ts = datetime.now(UTC).isoformat()

    db["risk_metrics"].insert_one(
        {
            "run_at": run_ts,
            "kpis": kpis,
            "correlation_matrix": corr_matrix.round(4).to_dict(),
        }
    )

    fan_df = mc_result["percentile_fan"]
    if isinstance(fan_df, pd.DataFrame):
        ops = [
            UpdateOne(
                {"day": int(row["day"])},
                {
                    "$set": {
                        "day": int(row["day"]),
                        "p05": float(row["p05"]),
                        "p25": float(row["p25"]),
                        "p50": float(row["p50"]),
                        "p75": float(row["p75"]),
                        "p95": float(row["p95"]),
                        "run_at": run_ts,
                    }
                },
                upsert=True,
            )
            for _, row in fan_df.iterrows()
        ]
        if ops:
            db["simulation_runs"].bulk_write(ops, ordered=False)

    logger.info("Saved Week 2 risk_metrics & simulation_runs to MongoDB (%s)", settings.mongo_db)
    client.close()


def push_quant_to_huggingface(settings: Settings, max_retries: int = 3) -> None:
    """Upload Week 2 Tableau/quant exports to Hugging Face Hub with retry."""
    if not settings.hf_token or "your_" in settings.hf_token:
        logger.warning("HF_TOKEN not configured; skipping Week 2 Hugging Face upload.")
        return

    api = HfApi(token=settings.hf_token)
    for csv_path in TABLEAU_DIR.glob("*.csv"):
        for attempt in range(1, max_retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(csv_path),
                    path_in_repo=f"tableau_exports/{csv_path.name}",
                    repo_id=settings.hf_repo_id,
                    repo_type="dataset",
                    commit_message=f"Update Week 2 quant export {csv_path.name}",
                )
                break
            except (HfHubHTTPError, httpx.HTTPError, OSError) as exc:
                logger.warning("HF upload %s attempt %d/%d failed: %s", csv_path.name, attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(2.0 * attempt)
    logger.info("Uploaded Week 2 quant CSVs to Hugging Face: %s", settings.hf_repo_id)


def run_week2_pipeline() -> dict[str, object]:
    """Execute full Week 2 quantitative analysis pipeline."""
    settings = get_settings()
    df = load_market_prices()
    price_matrix = build_price_matrix(df)
    log_returns = compute_log_returns(price_matrix)
    rolling_vol = compute_rolling_volatility(log_returns, window=30)
    corr_matrix = log_returns.corr().astype(np.float32)

    weights = np.array([a.weight for a in PORTFOLIO_ASSETS if a.asset_id in price_matrix.columns])
    kpis = compute_portfolio_kpis(log_returns, weights, confidence=settings.var_confidence)
    mc_result = run_monte_carlo(
        log_returns,
        weights,
        n_simulations=settings.mc_simulations,
        confidence=settings.var_confidence,
    )

    export_tableau_csvs(price_matrix, log_returns, rolling_vol, corr_matrix, mc_result)
    save_quant_to_mongodb(kpis, corr_matrix, mc_result, settings)
    push_quant_to_huggingface(settings)

    return {
        "price_matrix": price_matrix,
        "log_returns": log_returns,
        "rolling_vol": rolling_vol,
        "corr_matrix": corr_matrix,
        "kpis": kpis,
        "mc_result": mc_result,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = run_week2_pipeline()
    logger.info("Portfolio KPIs: %s", results["kpis"])