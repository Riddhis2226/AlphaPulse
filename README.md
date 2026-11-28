
# AlphaPulse — Portfolio Risk & Volatility Monitor

Multi-asset INR market-risk engine with VaR/CVaR, Euler risk attribution, and 10,000-path Cholesky Monte Carlo stress testing.

[**Live App (Plotly Cloud)**](https://alphapulse.plotly.app) · [**Dataset (Hugging Face)**](https://huggingface.co/datasets/RS2226/alphapulse-market-data) · [**Source Code**](https://github.com/Riddhis2226/AlphaPulse)

---

## Headline Risk Readout

*850 overlapping NSE trading days (`15 Jun 2023 – 25 Nov 2026`) · Base currency: **INR (₹)** · Risk-free proxy: **6.75%** (India T-Bill).*

| Metric | Value | Takeaway |
| :--- | :--- | :--- |
| **1D 95% Historical VaR** | **1.28%** | Matches Parametric Normal VaR (**1.29%**) |
| **1D 95% Expected Shortfall (CVaR)** | **1.86%** | **1.45× VaR** — size loss limits off CVaR, not VaR alone |
| **Annualized Return / Volatility** | **7.79% / 12.73%** | **1.90×** Diversification Ratio (**−47%** vol vs. 24.18% standalone) |
| **Sharpe / Sortino / Max Drawdown** | **0.08 / 0.12 / −22.41%** | Tech drag compresses excess return; rebalancing is the main lever |
| **IT Sell-Off (−15%) Stress Test** | **−₹49,500 (−4.95%)** | On ₹10 L capital; 252D 95% MC VaR = **₹1.16 L (12.22%)** |

---

## Portfolio Attribution (7 Assets · 6 Sectors)

Technology (**TCS + INFY**) carries **33.0% of capital** but drives **45.5% of total variance**, while **SUNPHARMA** and **GOLDBEES** anchor return and drawdown defense.

| Asset | Sector | Weight | Risk Share | Ann. Return | Ann. Vol | Sharpe | 1D 95% VaR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCS** | Technology | 18.0% | **23.8%** | −4.29% | 23.71% | −0.47 | 2.15% |
| **INFY** | Technology | 15.0% | **21.7%** | −0.44% | 26.48% | −0.27 | 2.54% |
| **RELIANCE** | Energy | 17.0% | 16.5% | +5.03% | 21.64% | −0.08 | 2.04% |
| **HDFCBANK** | Financials | 18.0% | 16.3% | +3.29% | 20.51% | −0.17 | 1.96% |
| **BTC_INR** | Crypto | 8.0% | 9.2% | +28.48% | 46.19% | 0.47 | 4.31% |
| **SUNPHARMA** | Healthcare | 12.0% | **7.1%** | +20.33% | 19.26% | **0.70** | 1.95% |
| **GOLDBEES** | Commodities | 12.0% | **5.4%** | +20.51% | 21.36% | 0.64 | 1.66% |

---

## Dashboard & Stack

- **Overview:** KPI strip, cumulative wealth & underwater drawdown, weight-vs-risk bars, Euler attribution table, Paper/Graphite themes, and 1-click CSV tear-sheet export.
- **Stress Test:** Instant sector shocks (*IT −15%*, *Crypto −40%*, *Energy −20%*, *Gold +15%*) + `@lru_cache` **10,000-path Cholesky Monte Carlo** (`30D / 90D / 1Y`) with **Normal** and **Student-t (`df=5`)** fat-tail modes.
- **Risk & Technicals:** Correlation heatmap, risk-vs-return map, 30D rolling volatility regimes, **Kupiec & Christoffersen rolling VaR backtesting**, and single-asset technical tear-sheet.
- **Engineering Stack:** Python 3.11 (`uv`), `polars`, `numpy`, `scipy`, **Plotly Dash** (deployed on **Plotly Cloud**), **Twelve Data** + `yfinance`, **MongoDB**, **Hugging Face Hub**, Tableau CSVs, `ruff` (0 warnings), `pytest` (10/10 passing).

---

## Quick Start

```bash
git clone https://github.com/Riddhis2226/AlphaPulse.git
cd AlphaPulse
uv venv .venv --python 3.11 && uv sync
uv run python app.py
```

Open `http://127.0.0.1:8050`. If `data/processed/market_prices.parquet` is missing locally, `app.py` automatically loads the cleaned dataset from Hugging Face Hub. To refresh data and run checks:

```bash
cp .env.example .env  # add TWELVE_DATA_API_KEY, MONGO_URI, HF_TOKEN, HF_REPO_ID
uv run python -m scripts.run_pipeline && uv run ruff check . && uv run pytest -v
```

---

## Model Note & Author

Metrics use daily log returns (`252` trading days/yr) with fixed weights, zero transaction costs, and `forward_fill()` holiday alignment; Switch between Gaussian and Multivariate Student-t (`df=5`) Cholesky paths to stress-test crisis fat tails. Built strictly for financial analytics—not investment advice.

**Riddhima Singh** · [GitHub](https://github.com/Riddhis2226) · *Infotact Solutions Enterprise Data Analytics Program (Q4 2025)*
