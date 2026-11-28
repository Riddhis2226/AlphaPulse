"""Week 1: Multi-asset market data ingestion via Twelve Data (with yfinance fallback),
INR normalization, forward-fill cleaning, float32 optimization, MongoDB & Hugging Face sync.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import httpx
import polars as pl
import yfinance as yf
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from src.config import (
    FX_ASSET,
    PORTFOLIO_ASSETS,
    PROCESSED_DIR,
    RAW_DIR,
    AssetSpec,
    Settings,
    get_settings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("alphapulse.ingest")

TD_BASE_URL = "https://api.twelvedata.com/time_series"
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]


def fetch_twelvedata(
    asset: AssetSpec,
    api_key: str,
    outputsize: int,
) -> pl.DataFrame | None:
    """Fetch daily adjusted OHLCV time series from Twelve Data REST API."""
    if not api_key or api_key == "your_twelve_data_api_key_here":
        return None

    params: dict[str, str | int] = {
        "symbol": asset.td_symbol,
        "interval": "1day",
        "outputsize": outputsize,
        "adjust": "all",
        "apikey": api_key,
    }
    if asset.td_exchange:
        params["exchange"] = asset.td_exchange

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(TD_BASE_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Twelve Data HTTP %s for %s (%s); falling back to yfinance.",
            exc.response.status_code,
            asset.asset_id,
            asset.td_symbol,
        )
        return None
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Twelve Data request failed for %s: %s", asset.asset_id, exc)
        return None

    if payload.get("status") != "ok" or "values" not in payload:
        logger.warning(
            "Twelve Data unavailable for %s (%s): %s",
            asset.asset_id,
            asset.td_symbol,
            payload.get("message", "unknown error"),
        )
        return None

    rows = [
        {
            "date": item["datetime"][:10],
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
            "volume": float(item.get("volume") or 0.0),
        }
        for item in payload["values"]
    ]

    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            *[pl.col(c).cast(pl.Float64) for c in NUMERIC_COLS],
            pl.lit(asset.asset_id).alias("asset_id"),
            pl.lit(asset.sector).alias("sector"),
            pl.lit("twelvedata").alias("source"),
        )
        .sort("date")
    )


def fetch_yfinance_fallback(
    asset: AssetSpec,
    lookback_days: int,
    max_retries: int = 3,
) -> pl.DataFrame:
    """Fallback fetcher using yfinance with retries, Ticker.history fallback, and local Parquet fallback."""
    period = "5y" if lookback_days >= 1000 else "2y"
    df_pd = None

    for attempt in range(1, max_retries + 1):
        try:
            df_pd = yf.download(
                asset.yf_symbol,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if df_pd is None or df_pd.empty:
                df_pd = yf.Ticker(asset.yf_symbol).history(
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                )
            if df_pd is not None and not df_pd.empty:
                break
        except Exception as exc:
            logger.warning(
                "yfinance attempt %d/%d failed for %s (%s): %s",
                attempt,
                max_retries,
                asset.asset_id,
                asset.yf_symbol,
                exc,
            )
        if attempt < max_retries:
            time.sleep(2.0 * attempt)

    if df_pd is None or df_pd.empty:
        cached_parquet = PROCESSED_DIR / "market_prices.parquet"
        if cached_parquet.exists():
            cached_asset = (
                pl.read_parquet(cached_parquet)
                .filter(pl.col("asset_id") == asset.asset_id)
                .select(["date", *NUMERIC_COLS, "asset_id", "sector", "source"])
                .with_columns(*[pl.col(c).cast(pl.Float64) for c in NUMERIC_COLS])
                .sort("date")
            )
            if cached_asset.height > 0:
                logger.warning(
                    "yfinance returned empty for %s after %d attempts; using cached local slice (%d rows).",
                    asset.asset_id,
                    max_retries,
                    cached_asset.height,
                )
                return cached_asset
        msg = f"No market data returned for {asset.asset_id} ({asset.yf_symbol})"
        raise ValueError(msg)

    df_pd = df_pd.reset_index()
    date_col = "Date" if "Date" in df_pd.columns else df_pd.columns[0]
    df_pd[date_col] = df_pd[date_col].astype(str).str.slice(0, 10)

    return (
        pl.from_pandas(df_pd[[date_col, "Open", "High", "Low", "Close", "Volume"]])
        .rename(
            {
                date_col: "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            *[pl.col(c).cast(pl.Float64) for c in NUMERIC_COLS],
            pl.lit(asset.asset_id).alias("asset_id"),
            pl.lit(asset.sector).alias("sector"),
            pl.lit("yfinance").alias("source"),
        )
        .sort("date")
    )


def load_asset_series(asset: AssetSpec, settings: Settings) -> pl.DataFrame:
    """Fetch an asset from Twelve Data first, falling back to yfinance if needed."""
    df = fetch_twelvedata(asset, settings.twelve_data_api_key, settings.lookback_days)
    if df is not None and df.height > 0:
        logger.info("Fetched %s via Twelve Data (%d rows)", asset.asset_id, df.height)
        return df

    logger.info("Falling back to yfinance for %s (%s)", asset.asset_id, asset.yf_symbol)
    return fetch_yfinance_fallback(asset, settings.lookback_days)


def align_and_clean(frames: list[pl.DataFrame], fx_df: pl.DataFrame) -> pl.DataFrame:
    """Convert USD crypto to INR, align on common overlapping NSE calendar with ffill, and downcast to float32."""
    fx_clean = fx_df.select(["date", pl.col("close").cast(pl.Float64).alias("usdinr")])
    combined = pl.concat(frames, how="vertical_relaxed")

    # Convert USD-denominated assets (e.g., BTC_INR) into INR
    usd_ids = {a.asset_id for a in PORTFOLIO_ASSETS if a.currency == "USD"}
    combined = combined.join(fx_clean, on="date", how="left").sort(["asset_id", "date"])
    combined = combined.with_columns(
        pl.col("usdinr").forward_fill().backward_fill().over("asset_id")
    )

    price_cols = ["open", "high", "low", "close"]
    combined = combined.with_columns(
        [
            pl.when(pl.col("asset_id").is_in(usd_ids))
            .then(pl.col(col) * pl.col("usdinr"))
            .otherwise(pl.col(col))
            .alias(col)
            for col in price_cols
        ]
    ).drop("usdinr")

    # Find common overlapping start date across all assets so every asset shares the same window
    common_start = (
        combined.group_by("asset_id")
        .agg(pl.col("date").min().alias("min_date"))
        .select(pl.col("min_date").max())
        .item()
    )

    nse_dates = (
        combined.filter((pl.col("asset_id") == "TCS") & (pl.col("date") >= common_start))
        .select("date")
        .unique()
        .sort("date")
    )
    assets_meta = combined.select(["asset_id", "sector", "source"]).unique(subset=["asset_id"])
    grid = nse_dates.join(assets_meta, how="cross").sort(["asset_id", "date"])

    cleaned = (
        grid.join(
            combined.select(["date", "asset_id", "open", "high", "low", "close", "volume"]),
            on=["date", "asset_id"],
            how="left",
        )
        .sort(["asset_id", "date"])
        .with_columns(
            [
                pl.col(c).forward_fill().backward_fill().over("asset_id").cast(pl.Float32)
                for c in NUMERIC_COLS
            ]
        )
        .drop_nulls(subset=["close"])
    )
    return cleaned


def run_integrity_checks(df: pl.DataFrame) -> None:
    """Validate zero nulls, positive prices, and equal row counts across assets."""
    null_closes = df.filter(pl.col("close").is_null()).height
    non_positive = df.filter(pl.col("close") <= 0).height
    if null_closes > 0 or non_positive > 0:
        msg = f"Data integrity failure: {null_closes} nulls, {non_positive} non-positive closes"
        raise ValueError(msg)

    counts = df.group_by("asset_id").len()["len"].unique().to_list()
    if len(counts) != 1:
        msg = f"Uneven asset row counts after alignment: {counts}"
        raise ValueError(msg)

    logger.info(
        "Integrity check passed: %d total rows (%d days x %d assets)",
        df.height,
        counts[0],
        df["asset_id"].n_unique(),
    )


def save_to_mongodb(df: pl.DataFrame, settings: Settings) -> None:
    """Replace/upsert cleaned daily market records into MongoDB for Compass inspection."""
    try:
        client: MongoClient = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except PyMongoError as exc:
        logger.warning("MongoDB unavailable (%s); skipping MongoDB upsert.", exc)
        return

    db = client[settings.mongo_db]
    coll = db["market_prices"]
    coll.create_index([("asset_id", 1), ("date", 1)], unique=True)

    ingested_at = datetime.now(UTC).isoformat()
    ops = [
        UpdateOne(
            {"asset_id": row["asset_id"], "date": row["date"].isoformat()},
            {
                "$set": {
                    "asset_id": row["asset_id"],
                    "sector": row["sector"],
                    "date": row["date"].isoformat(),
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                    "volume": float(row["volume"]),
                    "source": row["source"],
                    "ingested_at": ingested_at,
                }
            },
            upsert=True,
        )
        for row in df.iter_rows(named=True)
    ]

    if ops:
        coll.bulk_write(ops, ordered=False)
        logger.info("Upserted %d records into MongoDB (%s.market_prices)", len(ops), settings.mongo_db)
    client.close()


def push_to_huggingface(parquet_path: str, settings: Settings, max_retries: int = 3) -> None:
    """Upload cleaned Parquet dataset to Hugging Face Hub with retry for transient socket resets."""
    if not settings.hf_token or settings.hf_token == "your_huggingface_write_token_here":
        logger.warning("HF_TOKEN not configured; skipping Hugging Face upload.")
        return

    api = HfApi(token=settings.hf_token)
    for attempt in range(1, max_retries + 1):
        try:
            api.create_repo(repo_id=settings.hf_repo_id, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=parquet_path,
                path_in_repo="data/market_prices.parquet",
                repo_id=settings.hf_repo_id,
                repo_type="dataset",
                commit_message=f"Update AlphaPulse market prices ({datetime.now(UTC).date()})",
            )
            logger.info("Uploaded dataset to Hugging Face: %s", settings.hf_repo_id)
            return
        except (HfHubHTTPError, httpx.HTTPError, OSError) as exc:
            logger.warning("Hugging Face upload attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2.0 * attempt)


def main() -> None:
    """Execute Week 1 ingestion pipeline."""
    settings = get_settings()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fx_df = load_asset_series(FX_ASSET, settings)
    asset_frames: list[pl.DataFrame] = []
    for asset in PORTFOLIO_ASSETS:
        asset_frames.append(load_asset_series(asset, settings))
        time.sleep(1.0)

    cleaned = align_and_clean(asset_frames, fx_df)
    run_integrity_checks(cleaned)

    parquet_file = PROCESSED_DIR / "market_prices.parquet"
    csv_file = PROCESSED_DIR / "market_prices.csv"
    cleaned.write_parquet(parquet_file)
    cleaned.write_csv(csv_file)
    logger.info("Saved local Parquet (%s) and CSV (%s)", parquet_file, csv_file)

    save_to_mongodb(cleaned, settings)
    push_to_huggingface(str(parquet_file), settings)


if __name__ == "__main__":
    main()