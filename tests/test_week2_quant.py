"""Week 2 Statistical Sanity & Quantitative Risk Engine Tests for AlphaPulse."""

import numpy as np
import pandas as pd

from src.quant_engine import (
    backtest_var_exceptions,
    compute_euler_risk_attribution,
    compute_log_returns,
    compute_portfolio_kpis,
    compute_rolling_volatility,
    run_monte_carlo,
)


def _synthetic_prices(n_days: int = 400, n_assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    shocks = rng.normal(loc=0.0004, scale=0.015, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    cols = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=cols, dtype=np.float64)


def test_log_returns_and_rolling_vol_shapes() -> None:
    """Verify log returns stay float64 and 30-day rolling volatility has finite positive values."""
    prices = _synthetic_prices()
    log_ret = compute_log_returns(prices)
    assert log_ret.shape == (len(prices) - 1, prices.shape[1])
    assert all(dt == np.float64 for dt in log_ret.dtypes)

    roll_vol = compute_rolling_volatility(log_ret, window=30)
    assert not roll_vol.isna().any().any()
    assert (roll_vol.to_numpy() > 0).all()


def test_portfolio_var_and_cvar_ordering() -> None:
    """CVaR (Expected Shortfall) must always be >= VaR at the 95% confidence level."""
    prices = _synthetic_prices()
    log_ret = compute_log_returns(prices)
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    kpis = compute_portfolio_kpis(log_ret, weights, confidence=0.95)
    assert kpis["historical_cvar_95_1d"] >= kpis["historical_var_95_1d"] > 0
    assert kpis["parametric_var_95_1d"] > 0
    assert -1.0 <= kpis["max_drawdown"] <= 0.0


def test_euler_risk_attribution_exact_additivity() -> None:
    """Euler risk shares must sum to 1.0 and component vol contributions must sum to portfolio vol."""
    prices = _synthetic_prices()
    log_ret = compute_log_returns(prices)
    weights = np.array([0.35, 0.25, 0.25, 0.15])

    euler = compute_euler_risk_attribution(log_ret, weights)
    kpis = compute_portfolio_kpis(log_ret, weights, confidence=0.95)

    assert np.isclose(euler["risk_share"].sum(), 1.0, atol=1e-12)
    assert np.isclose(euler["ccr"].sum(), kpis["annualized_volatility"], atol=1e-5)


def test_var_backtest_kupiec_and_christoffersen() -> None:
    """Rolling VaR backtest must produce valid Kupiec POF and Christoffersen p-values."""
    prices = _synthetic_prices(n_days=500)
    log_ret = compute_log_returns(prices)
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    bt = backtest_var_exceptions(log_ret, weights, confidence=0.95, window=250)
    assert bt["n_observations"] == len(log_ret) - 250
    assert 0 <= int(bt["n_exceptions"]) <= int(bt["n_observations"])
    assert 0.0 <= float(bt["kupiec_p_value"]) <= 1.0
    assert 0.0 <= float(bt["christoffersen_p_value"]) <= 1.0
    assert 0.0 <= float(bt["conditional_coverage_p_value"]) <= 1.0


def test_monte_carlo_10k_sanity() -> None:
    """Verify 10,000+ Monte Carlo simulation shape and positive terminal wealth."""
    prices = _synthetic_prices()
    log_ret = compute_log_returns(prices)
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    mc = run_monte_carlo(log_ret, weights, n_simulations=10000, horizon_days=60)
    paths = mc["portfolio_paths"]
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (60, 10000)
    assert (paths > 0).all()


def test_monte_carlo_student_t_tail_mode() -> None:
    """Verify Multivariate Student-t Cholesky mode runs cleanly and yields valid positive paths."""
    prices = _synthetic_prices()
    log_ret = compute_log_returns(prices)
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    mc_t = run_monte_carlo(
        log_ret,
        weights,
        n_simulations=5000,
        horizon_days=30,
        distribution="student_t",
        df=5.0,
    )
    paths = mc_t["portfolio_paths"]
    assert mc_t["distribution"] == "student_t"
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (30, 5000)
    assert (paths > 0).all()