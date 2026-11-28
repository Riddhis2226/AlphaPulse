"""End-to-end automated pipeline for AlphaPulse (Week 1 Ingest -> Week 2 Quant -> Sync)."""

from __future__ import annotations

import logging

from scripts.week1_ingest import main as run_week1_ingest
from src.quant_engine import run_week2_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("alphapulse.pipeline")


def main() -> None:
    """Execute full data ingestion, quantitative modeling, MongoDB & Hugging Face sync."""
    logger.info("Starting AlphaPulse automated pipeline...")
    run_week1_ingest()
    results = run_week2_pipeline()
    logger.info("AlphaPulse pipeline completed | KPIs: %s", results["kpis"])


if __name__ == "__main__":
    main()
