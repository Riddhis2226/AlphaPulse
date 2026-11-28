"""Typed configuration and portfolio universe definitions for AlphaPulse."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TABLEAU_DIR = DATA_DIR / "tableau_exports"


class AssetSpec:
    """Metadata and ticker mapping for a portfolio asset."""

    def __init__(
        self,
        asset_id: str,
        name: str,
        sector: str,
        td_symbol: str,
        td_exchange: str | None,
        yf_symbol: str,
        weight: float,
        currency: str = "INR",
    ) -> None:
        self.asset_id = asset_id
        self.name = name
        self.sector = sector
        self.td_symbol = td_symbol
        self.td_exchange = td_exchange
        self.yf_symbol = yf_symbol
        self.weight = weight
        self.currency = currency


PORTFOLIO_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("TCS", "Tata Consultancy Services", "Technology", "TCS", "NSE", "TCS.NS", 0.18),
    AssetSpec("INFY", "Infosys", "Technology", "INFY", "NSE", "INFY.NS", 0.15),
    AssetSpec("HDFCBANK", "HDFC Bank", "Financials", "HDFCBANK", "NSE", "HDFCBANK.NS", 0.18),
    AssetSpec("RELIANCE", "Reliance Industries", "Energy", "RELIANCE", "NSE", "RELIANCE.NS", 0.17),
    AssetSpec("SUNPHARMA", "Sun Pharmaceutical", "Healthcare", "SUNPHARMA", "NSE", "SUNPHARMA.NS", 0.12),
    AssetSpec("GOLDBEES", "Nippon India Gold BeES", "Commodities", "GOLDBEES", "NSE", "GOLDBEES.NS", 0.12),
    AssetSpec("BTC_INR", "Bitcoin (INR)", "Crypto", "BTC/USD", None, "BTC-USD", 0.08, currency="USD"),
)

FX_ASSET = AssetSpec("USDINR", "USD/INR Spot", "FX", "USD/INR", None, "USDINR=X", 0.0, currency="INR")


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    twelve_data_api_key: str = Field(default="", alias="TWELVE_DATA_API_KEY")
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="alphapulse", alias="MONGO_DB")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    hf_repo_id: str = Field(default="RS2226/alphapulse-market-data", alias="HF_REPO_ID")
    lookback_days: int = 1260
    mc_simulations: int = 10000
    var_confidence: float = 0.95


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
