# AlphaPulse — Institutional Portfolio Risk & Volatility Monitor

**AlphaPulse** is an end-to-end quantitative market risk engine and executive **Finance Intelligence** dashboard built for multi-asset portfolio risk monitoring (Infotact Solutions Enterprise Data Analytics Project 2). It models a diversified 7-asset INR portfolio across Indian NSE equities, Gold ETF, and INR-normalized Bitcoin using vectorized NumPy/SciPy matrix math, **10,000-path Cholesky Monte Carlo simulations**, **Value at Risk (VaR / CVaR)**, **Euler Marginal Risk Attribution**, and interactive **What-If** macro stress testing.

---

## Business & Analytical Impact

| Portfolio Metric (Full Overlapping Window: 850 NSE Trading Days) | Verified Value | Analytical Interpretation |
| :--- | :---: | :--- |
| **1-Day 95% Historical VaR** | **1.28%** | Tightly aligned with **Parametric Normal VaR (1.29%)** |
| **1-Day 95% Expected Shortfall (CVaR)** | **1.86%** | Average daily loss beyond the 95% VaR tail threshold |
| **Annualized Return / Volatility** | **7.80% / 12.73%** | Diversified across 6 sectors in consistent INR denomination |
| **Diversification Ratio** | **1.90×** | Cuts volatility by **11.45%** vs. 24.18% standalone weighted average |
| **Maximum Drawdown** | **-22.41%** | Peak-to-trough portfolio loss tracked via underwater curve |
| **Top Risk Contributor (Euler Share)** | **TCS (23.8%)** | 18.0% capital weight drives 23.8% of total portfolio variance |
| **Top Risk-Adjusted Sleeve** | **SUNPHARMA (0.70 Sharpe)** | +20.33% annualized return on 19.26% annualized volatility |

---

## Architecture & Tech Stack

```text
Twelve Data REST API (USD/INR, BTC/USD, INFY) + yfinance Fallback (NSE .NS)
        │
        ▼
Polars ETL Pipeline (INR Normalization -> NSE Calendar Alignment -> ffill -> float32)
        │
        ├──▶ MongoDB (alphapulse DB: market_prices, risk_metrics, simulation_runs)
        └──▶ Hugging Face Hub (RS2226/alphapulse-market-data Parquet + Tableau CSVs)
                │
                ▼
Vectorized NumPy / SciPy Risk Engine (wᵀ Σ w, Cholesky 10,000-Path Monte Carlo, VaR/CVaR)
                │
                ▼
Plotly Dash Executive App (Dual Light/Dark UI, What-If Stress Waterfall, CSV Tear-Sheet)
```

* **Data & ETL:** Python 3.11, `uv`, `polars`, `pyarrow`, `httpx`, `twelvedata`, `yfinance`, `pydantic-settings`
* **Quantitative Core:** `numpy` (vectorized matrix ops & `float32` downcasting), `scipy` (Cholesky, normal quantiles)
* **Storage & BI:** **MongoDB Compass** (`pymongo`), **Hugging Face Datasets** (`huggingface-hub`), Tableau CSV exports (`data/tableau_exports/`)
* **Dashboard & Deployment:** **Plotly Dash** (`dash`, `dash-bootstrap-components`, `plotly`, `gunicorn`), **Docker** on **Hugging Face Spaces**
* **Quality Assurance:** `ruff` (zero lint warnings), `pytest` (7 automated data-integrity & statistical sanity tests)

---

## Quick Start (Local with `uv`)

```powershell
git clone https://github.com/Riddhis2226/AlphaPulse.git
cd AlphaPulse
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv sync
```

1. Copy `.env.example` to `.env` and fill in `TWELVE_DATA_API_KEY`, `MONGO_URI`, `HF_TOKEN`, and `HF_REPO_ID` (never commit `.env`).
2. Run the automated pipeline, tests, and dashboard:

```powershell
uv run ruff check .
uv run python -m scripts.run_pipeline
uv run pytest -v
uv run python app.py
```

Open `http://127.0.0.1:8050` in your browser.

---

## Repository Structure

```text
AlphaPulse/
├── app.py                              # Dual-theme Plotly Dash Finance Intelligence app
├── assets/custom.css                   # Executive Light & Institutional Dark CSS system
├── Dockerfile & requirements.txt       # Free cloud container config (Hugging Face Spaces)
├── pyproject.toml                      # uv dependencies & Ruff lint rules
├── src/
│   ├── config.py                       # Typed Pydantic settings & 7-asset INR universe
│   └── quant_engine.py                 # Vectorized VaR/CVaR, Sharpe, 10k Monte Carlo & sync
├── scripts/
│   ├── week1_ingest.py                 # Twelve Data + yfinance ingestion & integrity checks
│   └── run_pipeline.py                 # Automated end-to-end refresh script
├── notebooks/
│   ├── week1_data_acquisition.ipynb    # Interactive ingestion & corporate-action audit
│   └── week2_quant_risk.ipynb          # Interactive Monte Carlo & volatility analysis
├── tests/
│   ├── test_week1_integrity.py         # Schema, float32, null & split continuity tests
│   └── test_week2_quant.py             # VaR/CVaR ordering & 10k simulation sanity tests
├── data/tableau_exports/               # Flat CSVs ready for Tableau Public
└── docs/AlphaPulse_Project_Report.pdf  # Final Infotact Executive Project Report (5 pages)
```
