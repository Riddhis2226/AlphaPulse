"""Week 2: Vectorized Quantitative Risk Engine (Log Returns, Rolling Volatility,
Covariance/Correlation, Euler Risk Attribution, Kupiec & Christoffersen VaR
Backtesting, and 10,000+ Normal / Student-t Cholesky Monte Carlo Paths).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from scipy.stats import chi2, norm

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
    """Pivot long market_prices into a date-indexed wide float64 close-price DataFrame with zero NaNs."""
    wide = (
        df.pivot(on="asset_id", index="date", values="close")
        .sort("date")
        .to_pandas()
        .set_index("date")
    )
    asset_order = [a.asset_id for a in PORTFOLIO_ASSETS if a.asset_id in wide.columns]
    return wide[asset_order].dropna(how="any").astype(np.float64)


def compute_log_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute vectorized daily log returns in float64 precision: ln(P_t / P_{t-1})."""
    prices_arr = price_matrix.to_numpy(dtype=np.float64)
    log_ret_arr = np.diff(np.log(prices_arr), axis=0)
    return pd.DataFrame(
        log_ret_arr,
        index=price_matrix.index[1:],
        columns=price_matrix.columns,
        dtype=np.float64,
    )


def compute_rolling_volatility(log_returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute annualized rolling volatility per asset in float64 precision."""
    rolling_vol = log_returns.astype(np.float64).rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    return rolling_vol.dropna()


def compute_euler_risk_attribution(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Compute exact float64 Euler homogeneous volatility decomposition.

    Returns marginal contribution to volatility (MCR), component contribution
    to volatility (CCR = w_i * MCR_i, summing to annualized portfolio vol),
    and additive variance risk share (RC_i = w_i * (Sigma w)_i / (w^T Sigma w),
    summing to 1.0 exactly).
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    ret_arr = log_returns.to_numpy(dtype=np.float64)
    cov_ann = np.cov(ret_arr, rowvar=False) * TRADING_DAYS
    port_var = float(w.T @ cov_ann @ w)
    port_vol = float(np.sqrt(port_var))

    if port_var <= 0.0 or port_vol <= 0.0:
        n = len(w)
        return pd.DataFrame(
            {
                "asset_id": list(log_returns.columns),
                "weight": w,
                "mcr": np.zeros(n, dtype=np.float64),
                "ccr": np.zeros(n, dtype=np.float64),
                "risk_share": np.zeros(n, dtype=np.float64),
            }
        )

    sigma_w = cov_ann @ w
    mcr = sigma_w / port_vol
    ccr = w * mcr
    risk_share = (w * sigma_w) / port_var

    return pd.DataFrame(
        {
            "asset_id": list(log_returns.columns),
            "weight": w,
            "mcr": mcr,
            "ccr": ccr,
            "risk_share": risk_share,
        }
    )


def compute_portfolio_kpis(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float]:
    """Compute vectorized float64 portfolio return, volatility, Sharpe, Parametric & Historical VaR/CVaR."""
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


def backtest_var_exceptions(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    window: int = 250,
    significance: float = 0.05,
) -> dict[str, float | int | bool]:
    """Run rolling out-of-sample 1D Historical VaR backtest with Kupiec POF
    (unconditional coverage) and Christoffersen (independence & conditional coverage) tests.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    port_ret = log_returns.to_numpy(dtype=np.float64) @ w

    if len(port_ret) <= window + 5:
        msg = f"Need more than {window + 5} return observations for rolling VaR backtest."
        raise ValueError(msg)

    alpha = 1.0 - confidence
    n_oos = len(port_ret) - window
    exceptions = np.zeros(n_oos, dtype=np.int64)

    for i in range(n_oos):
        hist_slice = port_ret[i : i + window]
        var_thresh = -float(np.percentile(hist_slice, alpha * 100.0))
        exceptions[i] = 1 if (-port_ret[i + window] > var_thresh) else 0

    x = int(exceptions.sum())
    p_hat = x / n_oos
    eps = 1e-12

    # Kupiec Proportion of Failures (POF) LR test (df=1)
    p_clip = min(max(p_hat, eps), 1.0 - eps)
    log_l0 = (n_oos - x) * np.log(1.0 - alpha) + x * np.log(alpha)
    log_l1 = (n_oos - x) * np.log(1.0 - p_clip) + x * np.log(p_clip)
    lr_pof = float(max(0.0, -2.0 * (log_l0 - log_l1)))
    p_val_pof = float(chi2.sf(lr_pof, df=1))

    # Christoffersen Independence LR test (df=1)
    prev_ex = exceptions[:-1]
    curr_ex = exceptions[1:]
    n00 = int(np.sum((prev_ex == 0) & (curr_ex == 0)))
    n01 = int(np.sum((prev_ex == 0) & (curr_ex == 1)))
    n10 = int(np.sum((prev_ex == 1) & (curr_ex == 0)))
    n11 = int(np.sum((prev_ex == 1) & (curr_ex == 1)))

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_all = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _binom_ll(k: int, n: int, p: float) -> float:
        if n == 0:
            return 0.0
        pc = min(max(p, eps), 1.0 - eps)
        return float((n - k) * np.log(1.0 - pc) + k * np.log(pc))

    ll_ind_null = _binom_ll(n01 + n11, n00 + n01 + n10 + n11, pi_all)
    ll_ind_alt = _binom_ll(n01, n00 + n01, pi0) + _binom_ll(n11, n10 + n11, pi1)
    lr_ind = float(max(0.0, -2.0 * (ll_ind_null - ll_ind_alt)))
    p_val_ind = float(chi2.sf(lr_ind, df=1))

    # Joint Conditional Coverage LR test (df=2)
    lr_cc = lr_pof + lr_ind
    p_val_cc = float(chi2.sf(lr_cc, df=2))

    return {
        "n_observations": n_oos,
        "n_exceptions": x,
        "expected_rate": round(alpha, 4),
        "observed_rate": round(p_hat, 4),
        "kupiec_lr": round(lr_pof, 4),
        "kupiec_p_value": round(p_val_pof, 4),
        "pass_kupiec": p_val_pof >= significance,
        "christoffersen_lr": round(lr_ind, 4),
        "christoffersen_p_value": round(p_val_ind, 4),
        "pass_christoffersen": p_val_ind >= significance,
        "conditional_coverage_lr": round(lr_cc, 4),
        "conditional_coverage_p_value": round(p_val_cc, 4),
        "pass_conditional_coverage": p_val_cc >= significance,
    }


def run_monte_carlo(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10000,
    horizon_days: int = 252,
    initial_capital: float = 1_000_000.0,
    confidence: float = 0.95,
    seed: int = 42,
    distribution: Literal["normal", "student_t"] = "normal",
    df: float = 5.0,
) -> dict[str, object]:
    """Run vectorized float64 Cholesky-correlated Monte Carlo simulations with zero Python loops.

    Supports Gaussian ('normal') and variance-matched Multivariate Student-t ('student_t') shocks.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    ret_arr = log_returns.to_numpy(dtype=np.float64)
    mean_daily = ret_arr.mean(axis=0)
    cov_daily = np.cov(ret_arr, rowvar=False)

    # Cholesky decomposition in float64: Sigma = L @ L^T
    chol = np.linalg.cholesky(cov_daily)
    rng = np.random.default_rng(seed)

    # Shape: (horizon_days, n_simulations, n_assets)
    n_assets = ret_arr.shape[1]
    z = rng.standard_normal((horizon_days, n_simulations, n_assets), dtype=np.float64)

    if distribution == "student_t":
        if df <= 2.0:
            msg = "Student-t degrees of freedom (df) must be > 2.0 to preserve finite variance."
            raise ValueError(msg)
        # Shared chi-square scaling per (day, path) induces multivariate tail dependence
        chi2_draws = rng.chisquare(df, size=(horizon_days, n_simulations, 1))
        scale = np.sqrt((df - 2.0) / chi2_draws)
        z = z * scale

    correlated_shocks = z @ chol.T
    sim_asset_returns = mean_daily + correlated_shocks

    # Portfolio daily log returns across all simulations: (horizon_days, n_simulations)
    sim_port_log_ret = sim_asset_returns @ w
    cum_log_paths = np.cumsum(sim_port_log_ret, axis=0)
    portfolio_paths = initial_capital * np.exp(cum_log_paths)

    terminal_values = portfolio_paths[-1, :]
    terminal_returns = (terminal_values / initial_capital) - 1.0
    alpha = 1.0 - confidence
    mc_var_pct = float(-np.percentile(terminal_returns, alpha * 100.0))
    mc_var_inr = float(initial_capital * mc_var_pct)

    percentiles = np.percentile(portfolio_paths, [5, 25, 50, 75, 95], axis=1)

    return {
        "distribution": distribution,
        "portfolio_paths": portfolio_paths.astype(np.float32),
        "terminal_values": terminal_values.astype(np.float32),
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
    corr_matrix = log_returns.corr()

    weights = np.array([a.weight for a in PORTFOLIO_ASSETS if a.asset_id in price_matrix.columns])
    kpis = compute_portfolio_kpis(log_returns, weights, confidence=settings.var_confidence)
    euler_attr = compute_euler_risk_attribution(log_returns, weights)
    backtest = backtest_var_exceptions(log_returns, weights, confidence=settings.var_confidence)
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
        "euler_attribution": euler_attr,
        "var_backtest": backtest,
        "mc_result": mc_result,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = run_week2_pipeline()
    logger.info("Portfolio KPIs: %s", results["kpis"])
    logger.info("VaR Backtest: %s", results["var_backtest"])