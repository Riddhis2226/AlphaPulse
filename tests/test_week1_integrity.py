"""Week 1 Data Integrity & Corporate Action Sanity Tests for AlphaPulse."""

from pathlib import Path

import polars as pl
import pytest

from src.config import PORTFOLIO_ASSETS, PROCESSED_DIR

PARQUET_PATH = PROCESSED_DIR / "market_prices.parquet"


@pytest.fixture(scope="module")
def market_df() -> pl.DataFrame:
    """Load the cleaned Week 1 market_prices Parquet dataset."""
    if not Path(PARQUET_PATH).exists():
        pytest.skip("Run `uv run python -m scripts.week1_ingest` before running integrity tests.")
    return pl.read_parquet(PARQUET_PATH)


def test_all_portfolio_assets_present(market_df: pl.DataFrame) -> None:
    """Verify all configured portfolio assets exist in the cleaned dataset."""
    expected = {asset.asset_id for asset in PORTFOLIO_ASSETS}
    actual = set(market_df["asset_id"].unique().to_list())
    assert expected == actual


def test_zero_nulls_and_positive_closes(market_df: pl.DataFrame) -> None:
    """Ensure forward-fill removed all nulls and close prices remain strictly positive."""
    assert market_df.filter(pl.col("close").is_null()).height == 0
    assert market_df.filter(pl.col("close") <= 0).height == 0


def test_float32_memory_optimization(market_df: pl.DataFrame) -> None:
    """Verify numeric price/volume columns are downcast to float32."""
    for col in ("open", "high", "low", "close", "volume"):
        assert market_df.schema[col] == pl.Float32


def test_no_unadjusted_split_artifacts(market_df: pl.DataFrame) -> None:
    """Check for extreme single-day drops (>60%) that indicate unadjusted stock splits."""
    jumps = (
        market_df.sort(["asset_id", "date"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("asset_id") - 1.0).alias("ret"))
        .filter(pl.col("ret").abs() > 0.60)
    )
    assert jumps.height == 0, f"Potential unadjusted corporate action detected:\n{jumps}"
