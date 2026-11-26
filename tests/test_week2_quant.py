"""Week 2 Statistical Sanity & Quantitative Risk Engine Tests for AlphaPulse."""

import numpy as np
import pandas as pd

from src.quant_engine import (
    compute_log_returns,
    compute_portfolio_kpis,
    compute_rolling_volatility,
    run_monte_carlo,
)


def _synthetic_prices(n_days: int = 300, n_assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    shocks = rng.normal(loc=0.0004, scale=0.015, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    cols = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=cols, dtype=np.float32)


def test_log_returns_and_rolling_vol_shapes() -> None:
    """Verify log returns and 30-day rolling volatility output dimensions and finite values."""
    prices = _synthetic_prices()
    log_ret = compute_log_returns(prices)
    assert log_ret.shape == (len(prices) - 1, prices.shape[1])

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