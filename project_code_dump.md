
================================================================================
FILE: .env.example
================================================================================
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
MONGO_URI=mongodb://localhost:27017
MONGO_DB=alphapulse
HF_TOKEN=your_huggingface_write_token_here
HF_REPO_ID=RS2226/alphapulse-market-data


================================================================================
FILE: .github/workflows/ci.yml
================================================================================
﻿name: AlphaPulse CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python 3.11
        run: uv python install 3.11

      - name: Sync dependencies
        run: uv sync --frozen

      - name: Run Ruff linter
        run: uv run ruff check .

      - name: Run Pytest suite
        run: uv run pytest -v


================================================================================
FILE: .github/workflows/daily_refresh.yml
================================================================================
﻿name: AlphaPulse Daily Market Refresh

on:
  schedule:
    # Runs Monday-Friday at 11:00 UTC (16:30 IST, 1 hour after NSE market close)
    - cron: "0 11 * * 1-5"
  workflow_dispatch:

jobs:
  refresh-market-data:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python 3.11
        run: uv python install 3.11

      - name: Sync dependencies
        run: uv sync --frozen

      - name: Run AlphaPulse ingestion & quant pipeline
        env:
          TWELVE_DATA_API_KEY: ${{ secrets.TWELVE_DATA_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_REPO_ID: ${{ secrets.HF_REPO_ID }}
          MONGO_URI: ${{ secrets.MONGO_URI }}
          MONGO_DB: alphapulse
        run: uv run python -m scripts.run_pipeline

      - name: Verify quantitative integrity
        run: uv run pytest -v


================================================================================
FILE: app.py
================================================================================
"""AlphaPulse — Portfolio risk monitor (Plotly Dash).

Buy-side / FP&A risk-desk view: dual Light/Dark themes, historical & parametric
VaR / expected shortfall, Sharpe / Sortino / Calmar, diversification ratio, Euler
risk attribution, a CSV tear-sheet export, macro stress tests and a 10,000-path
Cholesky Monte Carlo.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, dcc, html
from flask import jsonify
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from plotly.subplots import make_subplots

from src.config import PORTFOLIO_ASSETS, PROCESSED_DIR, get_settings
from src.quant_engine import (
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS,
    build_price_matrix,
    compute_log_returns,
    compute_portfolio_kpis,
    compute_rolling_volatility,
    run_monte_carlo,
)

logger = logging.getLogger("alphapulse.dash")

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)
FONT_STACK = "IBM Plex Sans, system-ui, sans-serif"
MONO_STACK = "IBM Plex Mono, ui-monospace, monospace"

STRESS_PRESETS: dict[str, tuple[str, int]] = {
    "baseline": ("Technology", 0),
    "it_selloff": ("Technology", -15),
    "crypto_winter": ("Crypto", -40),
    "energy_shock": ("Energy", -20),
    "gold_hedge": ("Commodities", 15),
}

WINDOW_PHRASES = {252: "the last year", 504: "the last two years", 0: "the full history"}
WINDOW_SLUGS = {252: "1y", 504: "2y", 0: "full"}
MC_PATHS = 10_000
CONCENTRATION_FLAG = 1.5  # risk share ÷ capital weight above this is flagged


# ── Data ────────────────────────────────────────────────────────────────────
def load_dataset() -> pl.DataFrame:
    """Load latest market_prices.parquet from Hugging Face Hub, falling back to local Parquet."""
    local_path = PROCESSED_DIR / "market_prices.parquet"
    settings = get_settings()
    repo_id = settings.hf_repo_id or "RS2226/alphapulse-market-data"

    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename="data/market_prices.parquet",
            repo_type="dataset",
            token=settings.hf_token or None,
            force_download=True,
        )
        return pl.read_parquet(Path(cached_path)).sort(["asset_id", "date"])
    except (HfHubHTTPError, OSError) as exc:
        logger.warning("Hugging Face fetch failed (%s); falling back to local Parquet.", exc)
        if local_path.exists():
            return pl.read_parquet(local_path).sort(["asset_id", "date"])
        msg = f"Unable to load market_prices.parquet from Hugging Face ({repo_id}) or locally: {exc}"
        raise RuntimeError(msg) from exc


RAW_DF = load_dataset()
PRICE_MATRIX = build_price_matrix(RAW_DF)
LOG_RETURNS = compute_log_returns(PRICE_MATRIX)
ROLLING_VOL = compute_rolling_volatility(LOG_RETURNS, window=30)

ASSET_IDS = list(PRICE_MATRIX.columns)
ASSET_SPECS = [a for a in PORTFOLIO_ASSETS if a.asset_id in ASSET_IDS]
WEIGHTS_MAP = {a.asset_id: a.weight for a in ASSET_SPECS}
SECTOR_MAP = {a.asset_id: a.sector for a in ASSET_SPECS}
NAME_MAP = {a.asset_id: a.name for a in ASSET_SPECS}
BASE_WEIGHTS = np.array([WEIGHTS_MAP[aid] for aid in ASSET_IDS], dtype=np.float64)
BASE_WEIGHTS = BASE_WEIGHTS / BASE_WEIGHTS.sum()
NORM_WEIGHTS = dict(zip(ASSET_IDS, BASE_WEIGHTS.tolist(), strict=True))  # always sums to 1
SECTORS = sorted(set(SECTOR_MAP.values()))


def _fmt_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


DATA_START = _fmt_date(PRICE_MATRIX.index[0])
DATA_END = _fmt_date(PRICE_MATRIX.index[-1])
RF_LABEL = f"{DEFAULT_RISK_FREE_RATE * 100:g}%"


def window_returns(window_days: int | None) -> pd.DataFrame:
    """Log returns for the selected look-back (0 / None = full history)."""
    return LOG_RETURNS.tail(window_days) if window_days and window_days > 0 else LOG_RETURNS


# ── Formatting ──────────────────────────────────────────────────────────────
def fmt_inr(value: float, signed: bool = False) -> str:
    """₹ with lakh / crore compaction: ₹87,500 · ₹10.00 L · ₹1.25 Cr."""
    sign = ("+" if value >= 0 else "−") if signed else ("−" if value < 0 else "")
    mag = abs(value)
    if mag >= 1e7:
        body = f"{mag / 1e7:.2f} Cr"
    elif mag >= 1e5:
        body = f"{mag / 1e5:.2f} L"
    else:
        body = f"{mag:,.0f}"
    return f"{sign}₹{body}"


def _fmt_inr_axis(value: float) -> str:
    sign = "−" if value < 0 else ""
    mag = abs(value)
    if mag >= 1e7:
        body = f"{mag / 1e7:g} Cr"
    elif mag >= 1e5:
        body = f"{mag / 1e5:g} L"
    else:
        body = f"{mag:,.0f}"
    return f"{sign}₹{body}"


def inr_axis(lo: float, hi: float, target: int = 6) -> dict[str, list]:
    """Tick values/labels in lakh-crore notation (Plotly has no native support)."""
    span = hi - lo
    if span <= 0:
        return {}
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    vals = [float(v) for v in np.arange(math.ceil(lo / step) * step, hi + step * 0.5, step)]
    return {"tickvals": vals, "ticktext": [_fmt_inr_axis(v) for v in vals]}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# ── Theme ───────────────────────────────────────────────────────────────────
def get_palette(is_dark: bool) -> dict[str, str]:
    """Semantic chart colours. Keep in sync with the CSS tokens in assets/custom.css."""
    if is_dark:
        return {
            "template": "plotly_dark",
            "bg": "rgba(0, 0, 0, 0)",
            "card": "#111723",
            "mid": "#1c2536",
            "text": "#e8edf5",
            "muted": "#98a3b6",
            "grid": "rgba(148, 163, 184, 0.13)",
            "primary": "#6aa6ff",
            "emerald": "#34d399",
            "rose": "#fb7185",
            "amber": "#fbbf24",
            "violet": "#a393f5",
            "teal": "#2dd4bf",
        }
    return {
        "template": "plotly_white",
        "bg": "rgba(255, 255, 255, 0)",
        "card": "#ffffff",
        "mid": "#f1f3f7",
        "text": "#0e1726",
        "muted": "#566175",
        "grid": "#e6eaf0",
        "primary": "#1f4fd8",
        "emerald": "#0a7f5a",
        "rose": "#c8283d",
        "amber": "#c26a0a",
        "violet": "#6d5bd0",
        "teal": "#0f8b8d",
    }


def sector_color_map(pal: dict[str, str]) -> dict[str, str]:
    """One colour per sector, reused by the table, scatter and volatility chart."""
    # Rose is last on purpose: it doubles as the "loss" colour, so avoid it unless sectors run out.
    cycle = [pal["primary"], pal["amber"], pal["emerald"], pal["violet"], pal["teal"], pal["muted"], pal["rose"]]
    return {s: cycle[i % len(cycle)] for i, s in enumerate(SECTORS)}


TOP_LEGEND = {"orientation": "h", "x": 0, "y": 1.1, "yanchor": "bottom"}


def apply_chart_theme(fig: go.Figure, is_dark: bool, height: int = 360) -> go.Figure:
    """Shared typography, quiet gridlines, theme-aware hover labels, transparent canvas."""
    pal = get_palette(is_dark)
    fig.update_layout(
        template=pal["template"],
        paper_bgcolor=pal["bg"],
        plot_bgcolor=pal["bg"],
        font={"family": FONT_STACK, "color": pal["text"], "size": 12},
        margin={"l": 56, "r": 20, "t": 28, "b": 40},
        height=height,
        hoverlabel={
            "bgcolor": pal["card"],
            "bordercolor": pal["grid"],
            "font": {"family": FONT_STACK, "color": pal["text"], "size": 12},
        },
        legend={"font": {"size": 11, "color": pal["muted"]}, "bgcolor": "rgba(0,0,0,0)"},
    )
    axis_style = {
        "gridcolor": pal["grid"],
        "zerolinecolor": pal["grid"],
        "linecolor": pal["grid"],
        "tickfont": {"size": 11, "color": pal["muted"]},
        "title_font": {"size": 11, "color": pal["muted"]},
        "automargin": True,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


# ── Analytics ───────────────────────────────────────────────────────────────
def compute_asset_intelligence(log_ret: pd.DataFrame, weights: np.ndarray, confidence: float) -> pd.DataFrame:
    """Per-asset return, vol, Sharpe, Sortino, beta to portfolio, 1D VaR and Euler risk share."""
    arr = log_ret.to_numpy(dtype=np.float64)
    port_ret = arr @ weights
    port_var = float(np.var(port_ret, ddof=1))

    ann_ret = arr.mean(axis=0) * TRADING_DAYS
    ann_vol = arr.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    cov_ann = np.cov(arr, rowvar=False) * TRADING_DAYS

    port_vol_ann = float(np.sqrt(weights.T @ cov_ann @ weights))
    marginal = (cov_ann @ weights) / port_vol_ann if port_vol_ann > 0 else np.zeros_like(weights)
    risk_share = (weights * marginal) / port_vol_ann if port_vol_ann > 0 else np.zeros_like(weights)
    alpha = 1.0 - confidence

    rows = []
    for i, aid in enumerate(log_ret.columns):
        r_i = arr[:, i]
        downside = r_i[r_i < 0]
        down_vol = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else ann_vol[i]
        sharpe_i = (ann_ret[i] - DEFAULT_RISK_FREE_RATE) / ann_vol[i] if ann_vol[i] > 0 else 0.0
        sortino_i = (ann_ret[i] - DEFAULT_RISK_FREE_RATE) / down_vol if down_vol > 0 else 0.0
        beta_i = float(np.cov(r_i, port_ret, ddof=1)[0, 1] / port_var) if port_var > 0 else 1.0
        var_1d = float(-np.percentile(r_i, alpha * 100.0))

        rows.append(
            {
                "Asset": aid,
                "Name": NAME_MAP[aid],
                "Sector": SECTOR_MAP[aid],
                "Weight (%)": round(float(weights[i] * 100.0), 1),
                "Risk Share (%)": round(float(risk_share[i] * 100.0), 1),
                "Beta": round(beta_i, 2),
                "Ann. Return (%)": round(float(ann_ret[i] * 100.0), 2),
                "Ann. Vol (%)": round(float(ann_vol[i] * 100.0), 2),
                "Sharpe": round(float(sharpe_i), 2),
                "Sortino": round(float(sortino_i), 2),
                "1D VaR (%)": round(var_1d * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def cached_monte_carlo(start_capital: float, horizon_days: int, confidence: float) -> dict:
    """Memoised simulation.

    The 10k-path run is the slowest step in the app. Every control and the theme
    switch feed the same callback, so without a cache a dark-mode toggle re-runs the
    simulation (and, if the engine is not seeded, shifts the reported VaR).
    Treat the returned dict as read-only.
    """
    return run_monte_carlo(
        LOG_RETURNS,
        BASE_WEIGHTS,
        n_simulations=MC_PATHS,
        horizon_days=horizon_days,
        initial_capital=start_capital,
        confidence=confidence,
    )


# ── Components ──────────────────────────────────────────────────────────────
def make_kpi_card(title: str, value: str, sub_left: str, sub_right: str, accent: str, hint: str | None = None) -> html.Div:
    """KPI tile: label, headline figure, and two supporting metrics."""
    return html.Div(
        [
            html.Div(title, className="ap-kpi-label"),
            html.Div(value, className="ap-kpi-value ap-mono", style={"color": accent}),
            html.Div(
                [html.Span(sub_left, className="ap-muted"), html.Span(sub_right, className="ap-mono ap-kpi-sub")],
                className="ap-kpi-foot",
            ),
        ],
        className="ap-kpi h-100",
        title=hint,
    )


def build_card(
    title: str,
    content: object,
    subtitle: str | None = None,
    full_height: bool = True,
) -> dbc.Card:
    """Panel with a title on the left and a one-line qualifier on the right."""
    header = [html.Span(title, className="ap-card-title")]
    if subtitle:
        header.append(html.Span(subtitle, className="ap-card-sub"))
    return dbc.Card(
        [
            dbc.CardHeader(header, className="ap-card-header"),
            dbc.CardBody(content, className="p-3"),
        ],
        className="ap-card h-100" if full_height else "ap-card",
    )


def chart(graph_id: str) -> dcc.Loading:
    """Graph with a loading indicator (the Monte Carlo run is not instant)."""
    return dcc.Loading(
        dcc.Graph(id=graph_id, config={"displayModeBar": False, "displaylogo": False}),
        type="circle",
        color="#7b8aa0",
    )


def segmented(component_id: str, options: list[dict], value: object) -> dcc.RadioItems:
    return dcc.RadioItems(
        id=component_id,
        options=options,
        value=value,
        inline=True,
        className="ap-seg",
        labelClassName="ap-seg-opt",
        inputClassName="ap-seg-input",
    )


def _num_td(text: str, color: str | None = None, bold: bool = False) -> html.Td:
    cls = "text-end ap-mono" + (" fw-semibold" if bold else "")
    return html.Td(text, className=cls, style={"color": color} if color else None)


def build_holdings_table(asset_df: pd.DataFrame, pal: dict[str, str], totals: dict[str, float]) -> html.Div:
    """Risk attribution matrix: in-cell bars compare capital weight with risk share."""
    ordered = asset_df.sort_values("Risk Share (%)", ascending=False)
    sector_colors = sector_color_map(pal)
    bar_max = max(float(ordered["Weight (%)"].max()), float(ordered["Risk Share (%)"].max()), 1.0)

    def bar_td(value: float, color: str, text: str, bold: bool = False) -> html.Td:
        pct = min(max(value / bar_max * 100.0, 0.0), 100.0)
        fill = f"linear-gradient(90deg, {rgba(color, 0.2)} {pct:.1f}%, transparent {pct:.1f}%)"
        cls = "ap-bar ap-mono" + (" fw-semibold" if bold else "")
        return html.Td(html.Div(text, className=cls, style={"background": fill}), className="text-end")

    headers = [
        html.Th("Instrument"),
        html.Th("Sector"),
        html.Th("Weight", className="text-end", title="Share of invested capital"),
        html.Th("Risk share", className="text-end", title="Euler contribution to portfolio variance; shares sum to 100%"),
        html.Th(
            "Risk ÷ weight",
            className="text-end",
            title="Risk share per unit of capital. Above 1× the holding adds more risk than its weight.",
        ),
        html.Th("Beta", className="text-end", title="Sensitivity to the portfolio's daily return"),
        html.Th("Return", className="text-end", title="Annualised return"),
        html.Th("Volatility", className="text-end", title="Annualised volatility"),
        html.Th("Sharpe", className="text-end"),
        html.Th("Sortino", className="text-end"),
        html.Th("1D VaR", className="text-end", title="One-day historical value at risk at the selected confidence"),
    ]

    rows = []
    for _, r in ordered.iterrows():
        ret_col = pal["emerald"] if r["Ann. Return (%)"] >= 0 else pal["rose"]
        sh_col = pal["emerald"] if r["Sharpe"] >= 0.5 else (pal["amber"] if r["Sharpe"] >= 0 else pal["rose"])
        ratio = r["Risk Share (%)"] / r["Weight (%)"] if r["Weight (%)"] > 0 else float("nan")
        if math.isnan(ratio):
            ratio_txt, ratio_col = "—", pal["muted"]
        else:
            ratio_txt = f"{ratio:.1f}×"
            ratio_col = pal["amber"] if ratio >= CONCENTRATION_FLAG else (pal["emerald"] if ratio <= 0.75 else pal["muted"])
        rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            html.Div(r["Asset"], className="fw-semibold"),
                            html.Div(r["Name"], className="ap-muted", style={"fontSize": "0.74rem"}),
                        ]
                    ),
                    html.Td(
                        html.Span(
                            [
                                html.Span(className="ap-dot", style={"backgroundColor": sector_colors[r["Sector"]]}),
                                r["Sector"],
                            ],
                            className="ap-sector",
                        )
                    ),
                    bar_td(r["Weight (%)"], pal["muted"], f"{r['Weight (%)']:.1f}%"),
                    bar_td(r["Risk Share (%)"], pal["primary"], f"{r['Risk Share (%)']:.1f}%", bold=True),
                    _num_td(ratio_txt, ratio_col, bold=True),
                    _num_td(f"{r['Beta']:.2f}"),
                    _num_td(f"{r['Ann. Return (%)']:+.2f}%", ret_col, bold=True),
                    _num_td(f"{r['Ann. Vol (%)']:.2f}%"),
                    _num_td(f"{r['Sharpe']:.2f}", sh_col, bold=True),
                    _num_td(f"{r['Sortino']:.2f}"),
                    _num_td(f"{r['1D VaR (%)']:.2f}%", pal["rose"]),
                ]
            )
        )

    footer = html.Tfoot(
        html.Tr(
            [
                html.Td("Portfolio", colSpan=2),
                _num_td("100.0%"),
                _num_td("100.0%"),
                _num_td("—", pal["muted"]),
                _num_td("1.00"),
                _num_td(f"{totals['ret'] * 100:+.2f}%", pal["emerald"] if totals["ret"] >= 0 else pal["rose"]),
                _num_td(f"{totals['vol'] * 100:.2f}%"),
                _num_td(f"{totals['sharpe']:.2f}"),
                _num_td(f"{totals['sortino']:.2f}"),
                _num_td(f"{totals['var'] * 100:.2f}%", pal["rose"]),
            ]
        )
    )
    return html.Div(
        html.Table([html.Thead(html.Tr(headers)), html.Tbody(rows), footer], className="ap-table"),
        className="table-responsive",
    )


def build_footer() -> dbc.Row:
    """Method notes: what the numbers assume, so a reader can challenge them."""

    def note(title: str, body: str) -> dbc.Col:
        return dbc.Col(html.Div([html.Strong(title), body], className="ap-notes"), md=6, lg=3, className="mb-3")

    return dbc.Row(
        [
            note(
                "Risk measures",
                "VaR and expected shortfall are historical, taken from the empirical distribution of 1-day portfolio "
                "log returns; the parametric figure assumes normal returns. Risk share is each holding's Euler "
                "contribution to variance, so the shares sum to 100%.",
            ),
            note(
                "Ratios",
                f"Sharpe and Sortino use a {RF_LABEL} risk-free rate and annualise over {TRADING_DAYS} trading days. "
                "Calmar is annualised return divided by maximum drawdown.",
            ),
            note(
                "Simulation",
                f"{MC_PATHS:,} paths of correlated returns (Cholesky factor of the sample covariance) at fixed weights, "
                "calibrated on the full history regardless of the window above. A stress shock hits the target sector "
                "instantly, before the paths start.",
            ),
            note(
                "Scope",
                f"Daily data, {DATA_START} to {DATA_END}. Backward-looking analysis for information only; "
                "it is not investment advice.",
            ),
        ],
        className="mt-2",
    )


# ── Tabs ────────────────────────────────────────────────────────────────────
def build_executive_tab() -> dbc.Container:
    """Overview: KPIs → one-line read → wealth & risk charts → attribution table."""
    return dbc.Container(
        [
            html.Div(id="exec-kpi-row", className="mt-3 mb-2"),
            html.Div(id="exec-insights-row", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "Portfolio growth and drawdown",
                            chart("exec-wealth-drawdown"),
                            "Indexed to 100, with peak-to-trough loss below",
                        ),
                        lg=8,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_card(
                            "Capital weight vs. risk share",
                            chart("exec-weight-vs-risk"),
                            "Where the risk actually sits",
                        ),
                        lg=4,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "Holdings and risk attribution",
                            html.Div(id="exec-holdings-table"),
                            "Sorted by risk share",
                        ),
                        lg=12,
                        className="mb-3",
                    )
                ]
            ),
            build_footer(),
        ],
        fluid=True,
        className="px-0",
    )


def build_stress_tab() -> dbc.Container:
    """Stress test: scenario controls → shocked KPIs → Monte Carlo fan → distribution & P&L."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "Scenario",
                            [
                                html.Label("Preset", className="ap-field-label mb-1 d-block"),
                                dcc.Dropdown(
                                    id="dropdown-preset",
                                    options=[
                                        {"label": "Baseline (no shock)", "value": "baseline"},
                                        {"label": "IT sell-off (−15%)", "value": "it_selloff"},
                                        {"label": "Crypto winter (−40%)", "value": "crypto_winter"},
                                        {"label": "Energy drawdown (−20%)", "value": "energy_shock"},
                                        {"label": "Gold safe-haven rally (+15%)", "value": "gold_hedge"},
                                    ],
                                    value="it_selloff",
                                    clearable=False,
                                ),
                                html.Div("A preset fills in the sector and shock below. Change either to customise.", className="ap-help mb-3"),
                                html.Label("Starting capital (₹)", className="ap-field-label mb-1 d-block"),
                                dcc.Input(
                                    id="input-capital",
                                    type="number",
                                    value=1_000_000,
                                    min=100_000,
                                    step=100_000,
                                    debounce=True,
                                    className="form-control ap-input ap-mono mb-3",
                                ),
                                html.Label("Sector to shock", className="ap-field-label mb-1 d-block"),
                                dcc.Dropdown(
                                    id="dropdown-sector",
                                    options=[{"label": s, "value": s} for s in SECTORS],
                                    value="Technology",
                                    clearable=False,
                                    className="mb-3",
                                ),
                                html.Label("Instant price shock", className="ap-field-label mb-2 d-block"),
                                dcc.Slider(
                                    id="slider-shock",
                                    min=-50,
                                    max=30,
                                    step=5,
                                    value=-15,
                                    marks={v: f"{v}%" for v in range(-50, 35, 20)},
                                ),
                                html.Hr(className="ap-rule"),
                                html.Label("Forecast horizon", className="ap-field-label mb-2 d-block"),
                                segmented(
                                    "radio-horizon",
                                    [
                                        {"label": "30D", "value": 30},
                                        {"label": "90D", "value": 90},
                                        {"label": "1Y", "value": 252},
                                    ],
                                    252,
                                ),
                            ],
                            f"{MC_PATHS:,} simulated paths",
                        ),
                        lg=3,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            html.Div(id="stress-kpi-row", className="mb-3"),
                            build_card(
                                "Simulated portfolio value",
                                chart("mc-fan-chart"),
                                "Shaded: 5th–95th and 25th–75th percentile range",
                                full_height=False,
                            ),
                        ],
                        lg=9,
                        className="mb-3",
                    ),
                ],
                className="mt-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "Where the paths end",
                            chart("mc-hist-chart"),
                            "Terminal value, with the VaR cut-off",
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_card(
                            "Immediate shock P&L",
                            chart("stress-waterfall-chart"),
                            "Holdings in the shocked sector",
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
        ],
        fluid=True,
        className="px-0",
    )


def build_quant_tab() -> dbc.Container:
    """Risk & technicals: correlations and efficiency → rolling vol → single-asset tear-sheet."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "Correlation between holdings",
                            chart("quant-corr-heatmap"),
                            "Daily log returns, grouped by sector",
                        ),
                        lg=5,
                        className="mb-3",
                    ),
                    dbc.Col(
                        build_card(
                            "Risk vs. return",
                            chart("quant-efficiency-scatter"),
                            "Above the dotted line beats the portfolio's Sharpe ratio",
                        ),
                        lg=7,
                        className="mb-3",
                    ),
                ],
                className="mt-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "30-day rolling volatility",
                            chart("quant-rolling-vol"),
                            "Annualised; the portfolio is the heavy line",
                        ),
                        lg=12,
                        className="mb-3",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        build_card(
                            "Single-asset view",
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="dropdown-asset",
                                                options=[
                                                    {"label": f"{aid} — {NAME_MAP[aid]} ({SECTOR_MAP[aid]})", "value": aid}
                                                    for aid in ASSET_IDS
                                                ],
                                                value=ASSET_IDS[0],
                                                clearable=False,
                                            ),
                                            lg=4,
                                            className="mb-2",
                                        ),
                                        dbc.Col(html.Div(id="asset-tearsheet-strip"), lg=8, className="mb-2"),
                                    ],
                                    align="center",
                                    className="mb-2",
                                ),
                                chart("asset-price-volume-chart"),
                            ],
                            "Price, 20- and 50-day averages, volume",
                        ),
                        lg=12,
                        className="mb-3",
                    )
                ]
            ),
        ],
        fluid=True,
        className="px-0",
    )


# ── App shell ───────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, GOOGLE_FONTS],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="AlphaPulse | Portfolio risk monitor",
)
server = app.server


@server.route("/healthz")
def healthz():
    """Production healthcheck endpoint for uptime monitors and cloud WSGI probes."""
    return (
        jsonify(
            {
                "status": "ok",
                "service": "alphapulse",
                "data_through": str(DATA_END),
                "trading_days": int(len(PRICE_MATRIX)),
                "assets": int(PRICE_MATRIX.shape[1]),
            }
        ),
        200,
    )

app.layout = html.Div(
    id="app-root",
    **{"data-theme": "light"},
    children=[
        dcc.Download(id="download-tearsheet"),
        dbc.Navbar(
            dbc.Container(
                [
                    html.Div(
                        [
                            html.Span("AlphaPulse", className="ap-brand-title"),
                            html.Span("Portfolio risk monitor", className="ap-brand-sub d-none d-md-inline"),
                        ],
                        className="d-flex align-items-center",
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"Data through {DATA_END}",
                                className="ap-status-pill ap-mono me-3 d-none d-lg-inline-flex",
                            ),
                            dbc.Button("Export CSV", id="btn-export-csv", size="sm", className="ap-btn me-3"),
                            dbc.Switch(
                                id="theme-switch",
                                label="Dark mode",
                                value=False,
                                className="ap-text small fw-medium mb-0",
                            ),
                        ],
                        className="d-flex align-items-center",
                    ),
                ],
                fluid=True,
                className="d-flex justify-content-between align-items-center px-0",
            ),
            className="ap-navbar sticky-top",
        ),
        html.Div(
            dbc.Container(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        [
                                            html.Span("Window", className="ap-field-label"),
                                            segmented(
                                                "global-window",
                                                [
                                                    {"label": "1Y", "value": 252},
                                                    {"label": "2Y", "value": 504},
                                                    {"label": "Full history", "value": 0},
                                                ],
                                                0,
                                            ),
                                        ],
                                        className="ap-control-group",
                                    ),
                                    html.Span(
                                        [
                                            html.Span("VaR confidence", className="ap-field-label"),
                                            segmented(
                                                "global-confidence",
                                                [
                                                    {"label": "95%", "value": 0.95},
                                                    {"label": "99%", "value": 0.99},
                                                ],
                                                0.95,
                                            ),
                                        ],
                                        className="ap-control-group",
                                    ),
                                ],
                                className="d-flex flex-wrap align-items-center gap-4",
                            ),
                            md="auto",
                        ),
                        dbc.Col(
                            html.Span(
                                f"{len(PRICE_MATRIX):,} trading days · {DATA_START} – {DATA_END} · risk-free {RF_LABEL}",
                                className="ap-coverage ap-mono",
                            ),
                            className="d-none d-xl-flex justify-content-end",
                        ),
                    ],
                    align="center",
                    className="g-3",
                ),
                fluid=True,
                className="px-0",
            ),
            className="ap-controls",
        ),
        dbc.Container(
            dbc.Tabs(
                [
                    dbc.Tab(build_executive_tab(), label="Overview", tab_id="tab-exec"),
                    dbc.Tab(build_stress_tab(), label="Stress test", tab_id="tab-mc"),
                    dbc.Tab(build_quant_tab(), label="Risk & technicals", tab_id="tab-quant"),
                ],
                id="main-tabs",
                active_tab="tab-exec",
            ),
            fluid=True,
            className="px-4",
        ),
    ],
)


# ── Callbacks ───────────────────────────────────────────────────────────────
@app.callback(Output("app-root", "data-theme"), Input("theme-switch", "value"))
def toggle_theme(is_dark: bool) -> str:
    """Switch the CSS custom properties between Light and Dark."""
    return "dark" if is_dark else "light"


@app.callback(
    Output("download-tearsheet", "data"),
    Input("btn-export-csv", "n_clicks"),
    State("global-window", "value"),
    State("global-confidence", "value"),
    prevent_initial_call=True,
)
def export_risk_tearsheet(
    _n_clicks: int | None,
    window_days: int,
    confidence: float,
) -> dict[str, object]:
    """Export the risk attribution matrix for the active window and confidence."""
    asset_df = compute_asset_intelligence(window_returns(window_days), BASE_WEIGHTS, confidence=float(confidence))
    filename = f"alphapulse_risk_{WINDOW_SLUGS.get(window_days, 'custom')}_{int(float(confidence) * 100)}.csv"
    return dcc.send_data_frame(asset_df.to_csv, filename, index=False)


@app.callback(
    Output("exec-kpi-row", "children"),
    Output("exec-insights-row", "children"),
    Output("exec-wealth-drawdown", "figure"),
    Output("exec-weight-vs-risk", "figure"),
    Output("exec-holdings-table", "children"),
    Input("global-window", "value"),
    Input("global-confidence", "value"),
    Input("theme-switch", "value"),
)
def update_executive_tab(
    window_days: int,
    confidence: float,
    is_dark: bool,
) -> tuple[dbc.Row, html.Div, go.Figure, go.Figure, html.Div]:
    """Overview tab: KPIs, one-line read, insight cards, growth/drawdown, weight vs. risk, table."""
    pal = get_palette(bool(is_dark))
    sub_ret = window_returns(window_days)
    kpis = compute_portfolio_kpis(sub_ret, BASE_WEIGHTS, confidence=float(confidence))
    asset_df = compute_asset_intelligence(sub_ret, BASE_WEIGHTS, confidence=float(confidence))

    port_daily = sub_ret.to_numpy(dtype=np.float64) @ BASE_WEIGHTS
    downside = port_daily[port_daily < 0]
    down_vol = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else kpis["annualized_volatility"]
    sortino = (kpis["annualized_return"] - DEFAULT_RISK_FREE_RATE) / down_vol if down_vol > 0 else 0.0
    calmar = kpis["annualized_return"] / abs(kpis["max_drawdown"]) if kpis["max_drawdown"] < 0 else 0.0

    weighted_avg_vol = float((asset_df["Weight (%)"] / 100.0 * asset_df["Ann. Vol (%)"] / 100.0).sum())
    port_vol = kpis["annualized_volatility"]
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0
    div_cut = (weighted_avg_vol - port_vol) / weighted_avg_vol if weighted_avg_vol > 0 else 0.0
    top_risk = asset_df.sort_values("Risk Share (%)", ascending=False).iloc[0]
    top_ratio = top_risk["Risk Share (%)"] / top_risk["Weight (%)"] if top_risk["Weight (%)"] > 0 else 1.0
    best_sharpe = asset_df.sort_values("Sharpe", ascending=False).iloc[0]
    conf_label = f"{int(confidence * 100)}%"

    kpi_row = dbc.Row(
        [
            dbc.Col(
                make_kpi_card(
                    f"1-day value at risk ({conf_label})",
                    f"{kpis['historical_var_95_1d']:.2%}",
                    "Parametric (normal)",
                    f"{kpis['parametric_var_95_1d']:.2%}",
                    pal["rose"],
                    hint="Loss not expected to be exceeded on a single day at this confidence, from the historical distribution.",
                ),
                md=6,
                lg=3,
                className="mb-2",
            ),
            dbc.Col(
                make_kpi_card(
                    f"1-day expected shortfall ({conf_label})",
                    f"{kpis['historical_cvar_95_1d']:.2%}",
                    f"Max drawdown {kpis['max_drawdown']:.2%}",
                    f"Calmar {calmar:.2f}",
                    pal["amber"],
                    hint="Average loss on the days that breach VaR.",
                ),
                md=6,
                lg=3,
                className="mb-2",
            ),
            dbc.Col(
                make_kpi_card(
                    "Annualised return",
                    f"{kpis['annualized_return']:.2%}",
                    "Volatility",
                    f"{port_vol:.2%}",
                    pal["emerald"] if kpis["annualized_return"] >= 0 else pal["rose"],
                ),
                md=6,
                lg=3,
                className="mb-2",
            ),
            dbc.Col(
                make_kpi_card(
                    "Sharpe ratio",
                    f"{kpis['sharpe_ratio']:.2f}",
                    "Sortino",
                    f"{sortino:.2f}",
                    pal["primary"],
                    hint=f"Excess return over a {RF_LABEL} risk-free rate per unit of volatility.",
                ),
                md=6,
                lg=3,
                className="mb-2",
            ),
        ]
    )

    headline = html.P(
        [
            f"Over {WINDOW_PHRASES.get(window_days, 'the selected window')}, the portfolio returned ",
            html.Span(f"{kpis['annualized_return']:+.1%}", className="ap-mono"),
            " a year at ",
            html.Span(f"{port_vol:.1%}", className="ap-mono"),
            " volatility (Sharpe ",
            html.Span(f"{kpis['sharpe_ratio']:.2f}", className="ap-mono"),
            "). The worst drawdown was ",
            html.Span(f"{kpis['max_drawdown']:.1%}", className="ap-mono"),
            f". {top_risk['Asset']} is {top_risk['Weight (%)']:.0f}% of capital but "
            f"{top_risk['Risk Share (%)']:.0f}% of risk.",
        ],
        className="ap-headline",
    )

    def insight(title: str, body: str, color: str) -> dbc.Col:
        return dbc.Col(
            html.Div(
                [html.Div(title, className="ap-insight-title"), html.Div(body, className="ap-insight-body")],
                className="ap-insight h-100",
                style={"borderLeftColor": color},
            ),
            md=4,
            className="mb-2",
        )

    concentrated = top_ratio >= CONCENTRATION_FLAG
    insights_row = html.Div(
        [
            headline,
            dbc.Row(
                [
                    insight(
                        f"Diversification: {div_ratio:.2f}×",
                        f"Holding these assets together cuts volatility {div_cut:.0%}, from {weighted_avg_vol:.2%} "
                        f"(weighted standalone) to {port_vol:.2%}.",
                        pal["primary"],
                    ),
                    insight(
                        "Concentrated risk" if concentrated else "Risk is balanced",
                        f"{top_risk['Asset']} ({top_risk['Sector']}) is {top_risk['Weight (%)']:.1f}% of capital and "
                        f"{top_risk['Risk Share (%)']:.1f}% of variance, {top_ratio:.1f}× its weight.",
                        pal["amber"] if concentrated else pal["emerald"],
                    ),
                    insight(
                        "Best risk-adjusted holding",
                        f"{best_sharpe['Asset']} leads at {best_sharpe['Sharpe']:.2f} Sharpe "
                        f"({best_sharpe['Ann. Return (%)']:+.2f}% return on {best_sharpe['Ann. Vol (%)']:.2f}% volatility).",
                        pal["emerald"],
                    ),
                ]
            ),
        ]
    )

    cum_wealth = 100.0 * np.exp(np.cumsum(port_daily))
    running_peak = np.maximum.accumulate(cum_wealth)
    drawdown = (cum_wealth - running_peak) / running_peak
    i_trough = int(np.argmin(drawdown))

    fig_wd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.66, 0.34])
    fig_wd.add_trace(
        go.Scatter(
            x=sub_ret.index,
            y=cum_wealth,
            name="Portfolio index",
            line={"color": pal["primary"], "width": 2.2},
            hovertemplate="%{y:,.1f}<extra>Index</extra>",
        ),
        row=1,
        col=1,
    )
    fig_wd.add_trace(
        go.Scatter(
            x=sub_ret.index,
            y=drawdown,
            name="Drawdown",
            line={"color": pal["rose"], "width": 1.5},
            fill="tozeroy",
            fillcolor=rgba(pal["rose"], 0.16),
            hovertemplate="%{y:.1%}<extra>Drawdown</extra>",
        ),
        row=2,
        col=1,
    )
    fig_wd.add_hline(y=100, line_width=1, line_dash="dot", line_color=pal["muted"], opacity=0.55, row=1, col=1)
    fig_wd.add_annotation(
        x=sub_ret.index[i_trough],
        y=float(drawdown[i_trough]),
        text=f"Max drawdown {drawdown[i_trough]:.1%}",
        showarrow=True,
        arrowhead=0,
        arrowcolor=pal["rose"],
        ax=0,
        ay=-30,
        font={"color": pal["rose"], "size": 11},
        bgcolor=pal["card"],
        bordercolor=pal["rose"],
        borderpad=3,
        row=2,
        col=1,
    )
    apply_chart_theme(fig_wd, bool(is_dark), height=370)
    fig_wd.update_layout(showlegend=False, hovermode="x unified")
    fig_wd.update_yaxes(title_text="Index (start = 100)", row=1, col=1)
    fig_wd.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)

    ordered = asset_df.sort_values("Risk Share (%)", ascending=True)
    x_top = float(max(ordered["Weight (%)"].max(), ordered["Risk Share (%)"].max())) * 1.2
    fig_wr = go.Figure()
    fig_wr.add_trace(
        go.Bar(
            y=ordered["Asset"],
            x=ordered["Weight (%)"],
            name="Capital weight",
            orientation="h",
            marker_color=rgba(pal["muted"], 0.55),
            text=[f"{v:.1f}%" for v in ordered["Weight (%)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x:.1f}%<extra>Capital weight</extra>",
        )
    )
    fig_wr.add_trace(
        go.Bar(
            y=ordered["Asset"],
            x=ordered["Risk Share (%)"],
            name="Risk share",
            orientation="h",
            marker_color=pal["primary"],
            text=[f"{v:.1f}%" for v in ordered["Risk Share (%)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x:.1f}%<extra>Risk share</extra>",
        )
    )
    apply_chart_theme(fig_wr, bool(is_dark), height=370)
    fig_wr.update_layout(barmode="group", bargap=0.35, legend=TOP_LEGEND, margin={"l": 56, "r": 36, "t": 40, "b": 40})
    fig_wr.update_traces(textfont={"size": 10, "family": MONO_STACK, "color": pal["muted"]})
    fig_wr.update_xaxes(title_text="% of portfolio", range=[0, x_top])

    totals = {
        "ret": kpis["annualized_return"],
        "vol": port_vol,
        "sharpe": kpis["sharpe_ratio"],
        "sortino": sortino,
        "var": kpis["historical_var_95_1d"],
    }
    table = build_holdings_table(asset_df, pal, totals)
    return kpi_row, insights_row, fig_wd, fig_wr, table


@app.callback(
    Output("dropdown-sector", "value"),
    Output("slider-shock", "value"),
    Input("dropdown-preset", "value"),
)
def apply_stress_preset(preset_key: str) -> tuple[str, int]:
    """Sync the sector dropdown and shock slider when a preset is chosen."""
    return STRESS_PRESETS.get(preset_key, ("Technology", -15))


@app.callback(
    Output("stress-kpi-row", "children"),
    Output("mc-fan-chart", "figure"),
    Output("mc-hist-chart", "figure"),
    Output("stress-waterfall-chart", "figure"),
    Input("input-capital", "value"),
    Input("dropdown-sector", "value"),
    Input("slider-shock", "value"),
    Input("radio-horizon", "value"),
    Input("global-confidence", "value"),
    Input("theme-switch", "value"),
)
def update_stress_tab(
    capital: float | None,
    target_sector: str,
    shock_pct: float,
    horizon_days: int,
    confidence: float,
    is_dark: bool,
) -> tuple[dbc.Row, go.Figure, go.Figure, go.Figure]:
    """Stressed KPIs, Monte Carlo fan, terminal distribution and shock P&L."""
    pal = get_palette(bool(is_dark))
    init_cap = float(capital or 1_000_000.0)
    shock_dec = float(shock_pct) / 100.0
    conf_label = f"{int(confidence * 100)}%"

    sector_weight = sum(NORM_WEIGHTS[aid] for aid in ASSET_IDS if SECTOR_MAP[aid] == target_sector)
    port_shock_pct = sector_weight * shock_dec
    shocked_cap = init_cap * (1.0 + port_shock_pct)
    pnl_inr = shocked_cap - init_cap

    mc = cached_monte_carlo(round(shocked_cap, 2), int(horizon_days), float(confidence))
    fan = mc["percentile_fan"]
    term_vals = np.asarray(mc["terminal_values"], dtype=np.float64)
    prob_loss = float(np.mean(term_vals < init_cap))
    horizon_label = "1Y" if horizon_days == 252 else f"{horizon_days}D"

    kpi_row = dbc.Row(
        [
            dbc.Col(
                make_kpi_card(
                    f"{target_sector} shock impact",
                    fmt_inr(pnl_inr, signed=True),
                    f"Sector weight {sector_weight:.1%}",
                    f"{port_shock_pct:+.2%} of portfolio",
                    pal["rose"] if pnl_inr < 0 else pal["emerald"],
                    hint=f"{pnl_inr:+,.0f} INR",
                ),
                md=4,
                className="mb-2",
            ),
            dbc.Col(
                make_kpi_card(
                    "Capital after shock",
                    fmt_inr(shocked_cap),
                    f"Expected at {horizon_label}",
                    fmt_inr(mc["expected_terminal_inr"]),
                    pal["primary"],
                    hint=f"{shocked_cap:,.0f} INR",
                ),
                md=4,
                className="mb-2",
            ),
            dbc.Col(
                make_kpi_card(
                    f"{horizon_label} Monte Carlo VaR ({conf_label})",
                    fmt_inr(mc["mc_var_95_horizon_inr"]),
                    f"Ends below {fmt_inr(init_cap)}",
                    f"{prob_loss:.1%} of paths",
                    pal["rose"],
                    hint="Probability that terminal value finishes below the starting capital.",
                ),
                md=4,
                className="mb-2",
            ),
        ]
    )

    day = np.asarray(fan["day"])
    p05, p25, p50, p75, p95 = (np.asarray(fan[k], dtype=np.float64) for k in ("p05", "p25", "p50", "p75", "p95"))
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=day, y=p05, name="5th percentile", line={"color": pal["rose"], "width": 1.3, "dash": "dot"}))
    fig_fan.add_trace(
        go.Scatter(
            x=day,
            y=p95,
            name="95th percentile",
            line={"color": pal["emerald"], "width": 1.3, "dash": "dot"},
            fill="tonexty",
            fillcolor=rgba(pal["primary"], 0.07),
        )
    )
    fig_fan.add_trace(go.Scatter(x=day, y=p25, name="25th percentile", line={"color": rgba(pal["primary"], 0.5), "width": 1}))
    fig_fan.add_trace(
        go.Scatter(
            x=day,
            y=p75,
            name="75th percentile",
            line={"color": rgba(pal["primary"], 0.5), "width": 1},
            fill="tonexty",
            fillcolor=rgba(pal["primary"], 0.16),
        )
    )
    fig_fan.add_trace(go.Scatter(x=day, y=p50, name="Median", line={"color": pal["primary"], "width": 2.8}))
    fig_fan.update_traces(hovertemplate="₹%{y:,.0f}<extra>%{fullData.name}</extra>")
    fig_fan.add_hline(
        y=init_cap,
        line_dash="dash",
        line_width=1,
        line_color=pal["muted"],
        annotation_text="Starting capital",
        annotation_position="bottom right",
        annotation_font={"color": pal["muted"], "size": 11},
    )
    apply_chart_theme(fig_fan, bool(is_dark), height=320)
    fig_fan.update_layout(legend=TOP_LEGEND, hovermode="x unified")
    fig_fan.update_xaxes(title_text="Trading day")
    lo, hi = min(float(p05.min()), init_cap), max(float(p95.max()), init_cap)
    pad = (hi - lo) * 0.04
    fig_fan.update_yaxes(range=[lo - pad, hi + pad], **inr_axis(lo - pad, hi + pad))

    cutoff_pct = (1.0 - float(confidence)) * 100.0
    var_cutoff = float(np.percentile(term_vals, cutoff_pct))
    med_val = float(np.median(term_vals))

    counts, edges = np.histogram(term_vals, bins=55)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0])
    in_tail = centers <= var_cutoff
    y_top = float(counts.max()) * 1.08
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Bar(
            x=centers[~in_tail],
            y=counts[~in_tail],
            width=width,
            name="Outcomes",
            marker_color=rgba(pal["primary"], 0.75),
            hovertemplate="₹%{x:,.0f}<br>%{y} paths<extra></extra>",
        )
    )
    fig_hist.add_trace(
        go.Bar(
            x=centers[in_tail],
            y=counts[in_tail],
            width=width,
            name=f"Worst {100 - int(confidence * 100)}%",
            marker_color=rgba(pal["rose"], 0.85),
            hovertemplate="₹%{x:,.0f}<br>%{y} paths<extra></extra>",
        )
    )
    for value, label, color, dash in (  # noqa: F402
        (var_cutoff, f"{conf_label} VaR cut-off {fmt_inr(var_cutoff)}", pal["rose"], "dash"),
        (med_val, f"Median {fmt_inr(med_val)}", pal["emerald"], "dot"),
        (init_cap, f"Starting capital {fmt_inr(init_cap)}", pal["muted"], "dash"),
    ):
        fig_hist.add_trace(
            go.Scatter(
                x=[value, value],
                y=[0, y_top],
                mode="lines",
                name=label,
                line={"color": color, "width": 1.5, "dash": dash},
                hoverinfo="skip",
            )
        )
    apply_chart_theme(fig_hist, bool(is_dark), height=330)
    fig_hist.update_layout(
        barmode="overlay",
        bargap=0.04,
        legend={"orientation": "h", "x": 0, "y": 1.12, "yanchor": "bottom", "font": {"size": 10, "color": pal["muted"]}},
    )
    fig_hist.update_xaxes(title_text="Terminal portfolio value", **inr_axis(float(edges[0]), float(edges[-1])))
    fig_hist.update_yaxes(title_text="Simulated paths")

    affected = [aid for aid in ASSET_IDS if SECTOR_MAP[aid] == target_sector] or list(ASSET_IDS)
    impacts = [init_cap * NORM_WEIGHTS[aid] * (shock_dec if SECTOR_MAP[aid] == target_sector else 0.0) for aid in affected]
    fig_wf = go.Figure(
        go.Waterfall(
            x=[*affected, "Total"],
            y=[*impacts, pnl_inr],
            measure=[*["relative"] * len(affected), "total"],
            decreasing={"marker": {"color": pal["rose"]}},
            increasing={"marker": {"color": pal["emerald"]}},
            totals={"marker": {"color": pal["primary"]}},
            connector={"line": {"color": pal["grid"]}},
            text=[fmt_inr(v, signed=True) if v != 0 else "₹0" for v in [*impacts, pnl_inr]],
            textposition="outside",
            textfont={"family": MONO_STACK, "size": 11, "color": pal["muted"]},
            cliponaxis=False,
            hovertemplate="₹%{y:,.0f}<extra></extra>",
        )
    )
    apply_chart_theme(fig_wf, bool(is_dark), height=330)
    fig_wf.update_layout(showlegend=False, margin={"l": 56, "r": 20, "t": 40, "b": 48})
    running = np.cumsum(impacts)
    w_lo, w_hi = min(0.0, float(running.min()), pnl_inr), max(0.0, float(running.max()), pnl_inr)
    if w_hi - w_lo > 0:
        w_pad = (w_hi - w_lo) * 0.2
        fig_wf.update_yaxes(range=[w_lo - w_pad, w_hi + w_pad], **inr_axis(w_lo - w_pad, w_hi + w_pad, target=5))
    fig_wf.update_yaxes(title_text="P&L")

    return kpi_row, fig_fan, fig_hist, fig_wf


@app.callback(
    Output("quant-corr-heatmap", "figure"),
    Output("quant-efficiency-scatter", "figure"),
    Output("quant-rolling-vol", "figure"),
    Input("global-window", "value"),
    Input("global-confidence", "value"),
    Input("theme-switch", "value"),
)
def update_quant_tab(
    window_days: int,
    confidence: float,
    is_dark: bool,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Correlation heatmap, risk/return map with Sharpe line, and rolling volatility."""
    pal = get_palette(bool(is_dark))
    sector_colors = sector_color_map(pal)
    sub_ret = window_returns(window_days)
    asset_df = compute_asset_intelligence(sub_ret, BASE_WEIGHTS, confidence=float(confidence))
    kpis = compute_portfolio_kpis(sub_ret, BASE_WEIGHTS, confidence=float(confidence))

    # Lower triangle only, grouped by sector, so blocks of co-movement are visible.
    order = sorted(sub_ret.columns, key=lambda a: (SECTOR_MAP[a], a))
    corr = sub_ret[order].corr().to_numpy()
    z = corr[1:, :-1].copy()
    z[np.triu(np.ones_like(z, dtype=bool), k=1)] = np.nan
    labels = [["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z]
    fig_corr = go.Figure(
        go.Heatmap(
            z=z,
            x=order[:-1],
            y=order[1:],
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 11, "family": MONO_STACK, "color": pal["text"]},
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, pal["primary"]], [0.5, pal["mid"]], [1.0, pal["rose"]]],
            xgap=2,
            ygap=2,
            hoverongaps=False,
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
            colorbar={"thickness": 10, "outlinewidth": 0, "tickvals": [-1, 0, 1], "tickfont": {"color": pal["muted"], "size": 10}},
        )
    )
    apply_chart_theme(fig_corr, bool(is_dark), height=360)
    fig_corr.update_layout(margin={"l": 76, "r": 56, "t": 28, "b": 48})
    fig_corr.update_yaxes(autorange="reversed", showgrid=False)
    fig_corr.update_xaxes(showgrid=False)

    fig_eff = px.scatter(
        asset_df,
        x="Ann. Vol (%)",
        y="Ann. Return (%)",
        size="Weight (%)",
        size_max=26,
        color="Sector",
        color_discrete_map=sector_colors,
        text="Asset",
        hover_data=["Name", "Sharpe", "Risk Share (%)"],
    )
    label_pos = {"SUNPHARMA": "middle left", "GOLDBEES": "top right"}
    for tr in fig_eff.data:
        if tr.text is not None:
            tr.textposition = [label_pos.get(str(t), "top center") for t in tr.text]
    fig_eff.update_traces(
        textfont={"size": 10, "color": pal["muted"]},
        marker={"line": {"width": 1, "color": pal["card"]}, "opacity": 0.9},
    )
    x_max = float(max(asset_df["Ann. Vol (%)"].max(), kpis["annualized_volatility"] * 100.0)) * 1.12
    rf_pct = DEFAULT_RISK_FREE_RATE * 100.0
    fig_eff.add_trace(
        go.Scatter(
            x=[0, x_max],
            y=[rf_pct, rf_pct + kpis["sharpe_ratio"] * x_max],
            mode="lines",
            name=f"Portfolio Sharpe ({kpis['sharpe_ratio']:.2f})",
            line={"color": pal["muted"], "width": 1.3, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    fig_eff.add_trace(
        go.Scatter(
            x=[kpis["annualized_volatility"] * 100.0],
            y=[kpis["annualized_return"] * 100.0],
            mode="markers+text",
            name="Portfolio",
            text=["Portfolio"],
            textposition="bottom center",
            textfont={"size": 10, "color": pal["text"]},
            marker={"symbol": "star", "size": 17, "color": pal["amber"], "line": {"width": 1.2, "color": pal["card"]}},
        )
    )
    y_min = float(asset_df["Ann. Return (%)"].min()) - 6.0
    y_max = float(asset_df["Ann. Return (%)"].max()) + 8.0
    apply_chart_theme(fig_eff, bool(is_dark), height=360)
    fig_eff.update_layout(yaxis={"range": [y_min, y_max]}, legend={"orientation": "h", "x": 0, "y": -0.2})
    fig_eff.update_xaxes(title_text="Annualised volatility (%)", range=[0, x_max])
    fig_eff.update_yaxes(title_text="Annualised return (%)")

    roll_vol = compute_rolling_volatility(sub_ret, window=30)
    port_series = pd.Series(sub_ret.to_numpy(dtype=np.float64) @ BASE_WEIGHTS, index=sub_ret.index)
    port_roll = (port_series.rolling(30).std() * np.sqrt(TRADING_DAYS)).dropna()

    fig_vol = go.Figure()
    for aid in roll_vol.columns:
        fig_vol.add_trace(
            go.Scatter(
                x=roll_vol.index,
                y=roll_vol[aid],
                name=aid,
                line={"width": 1.2, "color": sector_colors.get(SECTOR_MAP.get(aid, ""), pal["muted"])},
                opacity=0.55,
            )
        )
    fig_vol.add_trace(go.Scatter(x=port_roll.index, y=port_roll, name="Portfolio", line={"color": pal["text"], "width": 3.0}))
    apply_chart_theme(fig_vol, bool(is_dark), height=340)
    fig_vol.update_layout(legend=TOP_LEGEND, hovermode="x unified")
    fig_vol.update_yaxes(title_text="Annualised volatility", tickformat=".0%", hoverformat=".1%")

    return fig_corr, fig_eff, fig_vol


@app.callback(
    Output("asset-tearsheet-strip", "children"),
    Output("asset-price-volume-chart", "figure"),
    Input("dropdown-asset", "value"),
    Input("global-window", "value"),
    Input("global-confidence", "value"),
    Input("theme-switch", "value"),
)
def update_asset_price_volume(
    asset_id: str,
    window_days: int,
    confidence: float,
    is_dark: bool,
) -> tuple[dbc.Row, go.Figure]:
    """Four-figure asset strip plus price, moving averages and volume."""
    pal = get_palette(bool(is_dark))
    sub = RAW_DF.filter(pl.col("asset_id") == asset_id).to_pandas()
    sub["ma20"] = sub["close"].rolling(20).mean()
    sub["ma50"] = sub["close"].rolling(50).mean()

    latest_close = float(sub["close"].iloc[-1])
    high_52w = float(sub["close"].tail(252).max())
    dd_from_high = (latest_close / high_52w) - 1.0
    curr_vol = float(ROLLING_VOL[asset_id].iloc[-1]) if asset_id in ROLLING_VOL.columns else 0.0
    alpha = 1.0 - float(confidence)
    var_1d = float(-np.percentile(window_returns(window_days)[asset_id].to_numpy(), alpha * 100.0))

    def stat(label: str, val: str, color: str) -> dbc.Col:
        return dbc.Col(
            html.Div(
                [
                    html.Div(label, className="ap-stat-label"),
                    html.Div(val, className="ap-stat-value ap-mono", style={"color": color}),
                ],
                className="ap-stat",
            ),
            xs=6,
            md=3,
        )

    strip = dbc.Row(
        [
            stat("Latest close", f"₹{latest_close:,.2f}", pal["text"]),
            stat("vs. 52-week high", f"{dd_from_high:+.1%}", pal["rose"] if dd_from_high < -0.05 else pal["emerald"]),
            stat("30-day volatility", f"{curr_vol:.1%}", pal["amber"]),
            stat(f"1-day VaR ({int(confidence * 100)}%)", f"{var_1d:.2%}", pal["rose"]),
        ],
        className="g-2",
    )

    up_day = sub["close"].diff().fillna(0.0) >= 0
    volume_colors = np.where(up_day, rgba(pal["emerald"], 0.45), rgba(pal["rose"], 0.45))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.74, 0.26])
    fig.add_trace(
        go.Scatter(x=sub["date"], y=sub["close"], name="Close", line={"color": pal["primary"], "width": 2.0},
                   hovertemplate="₹%{y:,.2f}<extra>Close</extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sub["date"], y=sub["ma20"], name="20-day average", line={"color": pal["emerald"], "width": 1.3, "dash": "dot"},
                   hovertemplate="₹%{y:,.2f}<extra>20-day</extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sub["date"], y=sub["ma50"], name="50-day average", line={"color": pal["amber"], "width": 1.5, "dash": "dash"},
                   hovertemplate="₹%{y:,.2f}<extra>50-day</extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=sub["date"], y=sub["volume"], name="Volume", marker_color=volume_colors, hovertemplate="%{y:,.0f}<extra>Volume</extra>"),
        row=2, col=1,
    )
    apply_chart_theme(fig, bool(is_dark), height=420)
    fig.update_layout(legend=TOP_LEGEND, hovermode="x unified", bargap=0)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", tickformat="~s", showgrid=False, row=2, col=1)
    return strip, fig


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(debug=False, host="0.0.0.0", port=port)

================================================================================
FILE: assets/custom.css
================================================================================
/* AlphaPulse — design system
   Light: "Paper" (cool grey canvas, white panels, ultramarine accent)
   Dark:  "Graphite" (blue-black canvas, raised panels, soft blue accent)
   Rules of the house: borders instead of shadows, one radius scale, numbers in
   tabular mono, colour only where it carries meaning (gain / loss / warning). */

#app-root[data-theme="light"] {
  --ap-bg: #f2f4f7;
  --ap-nav: #ffffff;
  --ap-card: #ffffff;
  --ap-card-header: #ffffff;
  --ap-border: #e1e5ec;
  --ap-border-strong: #c7cfdb;
  --ap-text: #0e1726;
  --ap-muted: #566175;
  --ap-input-bg: #ffffff;
  --ap-hover-row: #f5f7fb;
  --ap-accent: #1f4fd8;
  --ap-gain: #0a7f5a;
  --ap-loss: #c8283d;
  --ap-warn: #a85a08;
  --ap-chip-bg: #eef2fb;
  --ap-focus: rgba(31, 79, 216, 0.45);
}

#app-root[data-theme="dark"] {
  --ap-bg: #090c12;
  --ap-nav: #0d121b;
  --ap-card: #111723;
  --ap-card-header: #111723;
  --ap-border: rgba(148, 163, 184, 0.14);
  --ap-border-strong: rgba(148, 163, 184, 0.3);
  --ap-text: #e8edf5;
  --ap-muted: #98a3b6;
  --ap-input-bg: #0b1019;
  --ap-hover-row: rgba(106, 166, 255, 0.07);
  --ap-accent: #6aa6ff;
  --ap-gain: #34d399;
  --ap-loss: #fb7185;
  --ap-warn: #fbbf24;
  --ap-chip-bg: rgba(106, 166, 255, 0.12);
  --ap-focus: rgba(106, 166, 255, 0.55);
}

#app-root {
  --ap-nav-h: 56px;
  --ap-radius: 8px;
  --ap-radius-sm: 6px;
  background-color: var(--ap-bg);
  color: var(--ap-text);
  min-height: 100vh;
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 0.92rem;
  line-height: 1.45;
  padding-bottom: 3rem;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.ap-mono {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}

.ap-muted { color: var(--ap-muted) !important; }
.ap-text  { color: var(--ap-text) !important; }
.ap-gain  { color: var(--ap-gain) !important; }
.ap-loss  { color: var(--ap-loss) !important; }
.ap-warn  { color: var(--ap-warn) !important; }

/* ── Focus: always visible for keyboard users ─────────────────────────── */
#app-root :focus-visible {
  outline: 2px solid var(--ap-focus);
  outline-offset: 2px;
}

/* ── Top bar ──────────────────────────────────────────────────────────── */
.ap-navbar {
  background-color: var(--ap-nav) !important;
  border-bottom: 1px solid var(--ap-border);
  height: var(--ap-nav-h);
  padding: 0 1.5rem;
  z-index: 1030;
}

.ap-brand-title {
  font-weight: 600;
  font-size: 1.12rem;
  letter-spacing: -0.01em;
  color: var(--ap-text);
}

.ap-brand-title::before {
  content: "";
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-right: 0.55rem;
  border-radius: 2px;
  background: var(--ap-accent);
  transform: translateY(-1px);
}

.ap-brand-sub {
  color: var(--ap-muted);
  font-size: 0.85rem;
  padding-left: 0.75rem;
  margin-left: 0.75rem;
  border-left: 1px solid var(--ap-border-strong);
}

.ap-status-pill {
  align-items: center;
  padding: 0.25rem 0.65rem;
  border-radius: var(--ap-radius-sm);
  background: var(--ap-chip-bg);
  color: var(--ap-muted);
  font-size: 0.74rem;
}

.ap-btn.btn {
  background: transparent !important;
  color: var(--ap-text) !important;
  border: 1px solid var(--ap-border-strong) !important;
  border-radius: var(--ap-radius-sm) !important;
  font-size: 0.8rem;
  font-weight: 500;
  padding: 0.3rem 0.8rem;
}

.ap-btn.btn:hover {
  border-color: var(--ap-accent) !important;
  color: var(--ap-accent) !important;
}

#app-root .form-switch .form-check-input {
  background-color: var(--ap-border-strong);
  border-color: var(--ap-border-strong);
  cursor: pointer;
}

#app-root .form-switch .form-check-input:checked {
  background-color: var(--ap-accent);
  border-color: var(--ap-accent);
}

/* ── Global controls (sticky under the top bar) ───────────────────────── */
.ap-controls {
  position: sticky;
  top: var(--ap-nav-h);
  z-index: 1020;
  background-color: var(--ap-nav);
  border-bottom: 1px solid var(--ap-border);
  padding: 0.55rem 1.5rem;
}

.ap-control-group {
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
}

.ap-field-label {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--ap-muted);
}

.ap-coverage {
  font-size: 0.74rem;
  color: var(--ap-muted);
}

/* Segmented control (dcc.RadioItems with className="ap-seg").
   Styles the <label> child so it works with both the classic and the newer
   Radix-based dcc DOM. */
.ap-seg {
  display: inline-flex !important;
  gap: 2px;
  padding: 2px;
  background: var(--ap-bg);
  border: 1px solid var(--ap-border);
  border-radius: var(--ap-radius);
}

.ap-seg > label {
  position: relative;
  display: inline-flex !important;
  align-items: center;
  margin: 0 !important;
  padding: 0.26rem 0.8rem !important;
  border-radius: var(--ap-radius-sm);
  background: transparent !important;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--ap-muted) !important;
  white-space: nowrap;
  cursor: pointer;
  transition: background-color 0.12s ease, color 0.12s ease;
}

.ap-seg > label * { color: inherit !important; font-weight: inherit; }
.ap-seg > label:hover { color: var(--ap-text) !important; }

.ap-seg > label:has(input:checked),
.ap-seg > label.selected {
  background: var(--ap-card) !important;
  color: var(--ap-accent) !important;
  font-weight: 600;
  box-shadow: 0 0 0 1px var(--ap-border-strong);
}

.ap-seg > label:has(input:focus-visible) {
  outline: 2px solid var(--ap-focus);
  outline-offset: 1px;
}

.ap-seg input,
.ap-seg .dash-options-list-option-wrapper {
  position: absolute !important;
  opacity: 0;
  width: 0 !important;
  height: 0 !important;
  margin: 0 !important;
  overflow: hidden;
}

/* ── Tabs: underline style ────────────────────────────────────────────── */
#app-root .nav-tabs {
  border-bottom: 1px solid var(--ap-border) !important;
  gap: 1.4rem;
  margin-top: 0.4rem;
}

#app-root .nav-tabs .nav-item { margin-bottom: -1px; }

#app-root .nav-tabs .nav-link {
  background: transparent !important;
  border: 0 !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--ap-muted) !important;
  font-weight: 500;
  font-size: 0.92rem;
  padding: 0.7rem 0.1rem;
  transition: color 0.12s ease, border-color 0.12s ease;
}

#app-root .nav-tabs .nav-link:hover { color: var(--ap-text) !important; }

#app-root .nav-tabs .nav-link.active {
  color: var(--ap-text) !important;
  font-weight: 600;
  border-bottom-color: var(--ap-accent) !important;
}

/* ── Panels ───────────────────────────────────────────────────────────── */
.ap-card {
  background-color: var(--ap-card) !important;
  border: 1px solid var(--ap-border) !important;
  border-radius: var(--ap-radius) !important;
  color: var(--ap-text) !important;
}

.ap-card-header {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.15rem 1rem;
  padding: 0.7rem 1rem !important;
  background-color: var(--ap-card-header) !important;
  border-bottom: 1px solid var(--ap-border) !important;
  border-top-left-radius: var(--ap-radius) !important;
  border-top-right-radius: var(--ap-radius) !important;
}

.ap-card-title { font-weight: 600; font-size: 0.94rem; color: var(--ap-text); }
.ap-card-sub   { font-size: 0.78rem; color: var(--ap-muted); }

.ap-help {
  margin-top: 0.35rem;
  font-size: 0.76rem;
  color: var(--ap-muted);
}

/* ── KPI tiles ────────────────────────────────────────────────────────── */
.ap-kpi {
  padding: 0.9rem 1rem;
  background-color: var(--ap-card);
  border: 1px solid var(--ap-border);
  border-radius: var(--ap-radius);
}

.ap-kpi-label { font-size: 0.8rem; font-weight: 500; color: var(--ap-muted); }

.ap-kpi-value {
  margin: 0.2rem 0 0.1rem;
  font-size: 1.75rem;
  font-weight: 600;
  line-height: 1.15;
}

.ap-kpi-foot {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-top: 0.55rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--ap-border);
  font-size: 0.78rem;
}

.ap-kpi-sub { font-weight: 500; color: var(--ap-text); }

/* ── Summary line + insights ──────────────────────────────────────────── */
.ap-headline {
  max-width: none;
  margin: 0.15rem 0 0.8rem;
  font-size: 0.98rem;
  line-height: 1.45;
  color: var(--ap-text);
}

.ap-headline .ap-mono { font-weight: 500; }

.ap-insight {
  padding: 0.75rem 1rem;
  background-color: var(--ap-card);
  border: 1px solid var(--ap-border);
  border-left: 3px solid var(--ap-accent);
  border-radius: var(--ap-radius-sm);
}

.ap-insight-title { font-size: 0.82rem; font-weight: 600; margin-bottom: 0.15rem; }
.ap-insight-body  { font-size: 0.84rem; color: var(--ap-muted); }

/* ── Asset stat strip ─────────────────────────────────────────────────── */
.ap-stat {
  padding: 0.4rem 0.7rem;
  background: var(--ap-bg);
  border: 1px solid var(--ap-border);
  border-radius: var(--ap-radius-sm);
}

.ap-stat-label { font-size: 0.72rem; color: var(--ap-muted); }
.ap-stat-value { font-size: 0.98rem; font-weight: 600; }

/* ── Holdings table ───────────────────────────────────────────────────── */
.ap-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.84rem;
}

.ap-table thead th {
  padding: 0.55rem 0.75rem;
  background-color: var(--ap-card-header);
  border-bottom: 1px solid var(--ap-border-strong);
  color: var(--ap-muted);
  font-size: 0.75rem;
  font-weight: 500;
  white-space: nowrap;
}

.ap-table thead th[title] {
  text-decoration: underline dotted var(--ap-border-strong);
  text-underline-offset: 3px;
  cursor: help;
}

.ap-table tbody td,
.ap-table tfoot td {
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid var(--ap-border);
  color: var(--ap-text);
  vertical-align: middle;
}

.ap-table tbody tr:hover { background-color: var(--ap-hover-row); }

.ap-table tfoot td {
  border-top: 1px solid var(--ap-border-strong);
  border-bottom: 0;
  background-color: var(--ap-bg);
  font-weight: 600;
}

.ap-table tfoot td:first-child { border-bottom-left-radius: var(--ap-radius-sm); }
.ap-table tfoot td:last-child  { border-bottom-right-radius: var(--ap-radius-sm); }

.ap-bar {
  min-width: 5.5rem;
  padding: 0.12rem 0.5rem;
  border-radius: 4px;
  text-align: right;
}

.ap-sector {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.8rem;
  color: var(--ap-muted);
}

.ap-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: none;
}

/* ── Inputs ───────────────────────────────────────────────────────────── */
.ap-input {
  background-color: var(--ap-input-bg) !important;
  color: var(--ap-text) !important;
  border: 1px solid var(--ap-border-strong) !important;
  border-radius: var(--ap-radius-sm) !important;
  font-size: 0.88rem;
}

.ap-input:focus {
  border-color: var(--ap-accent) !important;
  box-shadow: 0 0 0 3px var(--ap-focus) !important;
}

/* Dropdowns — Dash 2.x (react-select) */
#app-root .Select-control {
  background-color: var(--ap-input-bg) !important;
  border: 1px solid var(--ap-border-strong) !important;
  border-radius: var(--ap-radius-sm) !important;
  color: var(--ap-text) !important;
  font-size: 0.88rem;
}

#app-root .Select.is-focused:not(.is-open) > .Select-control,
#app-root .Select.is-open > .Select-control {
  border-color: var(--ap-accent) !important;
}

#app-root .Select-value-label,
#app-root .Select-placeholder,
#app-root .Select-input > input { color: var(--ap-text) !important; }

#app-root .Select-menu-outer {
  background-color: var(--ap-card) !important;
  border: 1px solid var(--ap-border-strong) !important;
  border-radius: var(--ap-radius-sm) !important;
  z-index: 1050;
}

#app-root .VirtualizedSelectOption {
  background-color: var(--ap-card) !important;
  color: var(--ap-text) !important;
}

#app-root .VirtualizedSelectFocusedOption {
  background-color: var(--ap-hover-row) !important;
  color: var(--ap-accent) !important;
}

/* Dropdowns — Dash 3.x (best effort; harmless on Dash 2) */
#app-root .dash-dropdown,
#app-root .dash-dropdown-trigger {
  background-color: var(--ap-input-bg) !important;
  color: var(--ap-text) !important;
  border-color: var(--ap-border-strong) !important;
}

#app-root .dash-dropdown-content {
  background-color: var(--ap-card) !important;
  color: var(--ap-text) !important;
  border: 1px solid var(--ap-border-strong) !important;
}


/* Dropdown trigger — Dash 3.1+ / 4 */
#app-root .dash-dropdown {
  border-radius: var(--ap-radius-sm) !important;
  font-size: 0.88rem;
}

#app-root .dash-dropdown:focus-visible,
#app-root .dash-dropdown[data-state="open"] { border-color: var(--ap-accent) !important; }

/* Number input — Dash 3.1+ / 4 (wrapper carries the border; inner input is bare) */
#app-root .ap-input.dash-input-container {
  display: flex;
  align-items: stretch;
  padding: 0 !important;
  overflow: hidden;
}

#app-root .ap-input:focus-within {
  border-color: var(--ap-accent) !important;
  box-shadow: 0 0 0 3px var(--ap-focus) !important;
}

#app-root .ap-input .dash-input-element {
  flex: 1;
  min-width: 0;
  padding: 0.4rem 0.65rem;
  border: 0 !important;
  outline: 0;
  box-shadow: none !important;
  background: transparent !important;
  color: var(--ap-text) !important;
  font: inherit;
}

#app-root .ap-input .dash-input-stepper {
  width: 1.9rem;
  border: 0;
  border-left: 1px solid var(--ap-border);
  background: transparent !important;
  color: var(--ap-muted) !important;
  cursor: pointer;
}

#app-root .ap-input .dash-input-stepper:hover { color: var(--ap-accent) !important; background: var(--ap-hover-row) !important; }

#app-root .dash-slider-container input[type="number"] {
  background-color: var(--ap-input-bg) !important;
  color: var(--ap-text) !important;
  border: 1px solid var(--ap-border-strong) !important;
  border-radius: var(--ap-radius-sm) !important;
}

/* Slider — Dash 3.1+ / 4 */
#app-root .dash-slider-track { background-color: var(--ap-border-strong) !important; }
#app-root .dash-slider-range { background-color: var(--ap-accent) !important; }

#app-root .dash-slider-container [role="slider"] {
  background-color: var(--ap-card) !important;
  border: 2px solid var(--ap-accent) !important;
  box-shadow: none !important;
}

#app-root .dash-slider-container [role="slider"]:focus-visible { box-shadow: 0 0 0 4px var(--ap-focus) !important; }
#app-root .dash-slider-mark { color: var(--ap-muted) !important; font-size: 0.72rem; }

/* Slider — Dash 2.x (rc-slider) */
#app-root .rc-slider-rail {
  background-color: var(--ap-border-strong) !important;
  height: 4px !important;
}

#app-root .rc-slider-track {
  background-color: var(--ap-accent) !important;
  height: 4px !important;
}

#app-root .rc-slider-handle {
  width: 16px !important;
  height: 16px !important;
  margin-top: -6px !important;
  background-color: var(--ap-card) !important;
  border: 2px solid var(--ap-accent) !important;
  opacity: 1 !important;
}

#app-root .rc-slider-handle:focus-visible {
  box-shadow: 0 0 0 4px var(--ap-focus) !important;
}

#app-root .rc-slider-dot { border-color: var(--ap-border-strong) !important; background: var(--ap-card) !important; }
#app-root .rc-slider-mark-text { color: var(--ap-muted) !important; font-size: 0.72rem; }

hr.ap-rule { border-color: var(--ap-border); opacity: 1; margin: 1.1rem 0; }

/* ── Method notes ─────────────────────────────────────────────────────── */
.ap-notes { font-size: 0.78rem; color: var(--ap-muted); line-height: 1.55; }
.ap-notes strong { display: block; margin-bottom: 0.15rem; color: var(--ap-text); font-weight: 600; }

/* ── Small screens ────────────────────────────────────────────────────── */
@media (max-width: 767.98px) {
  .ap-navbar   { padding: 0 1rem; }
  .ap-controls { position: static; padding: 0.55rem 1rem; }
  .ap-kpi-value { font-size: 1.5rem; }
}

@media (prefers-reduced-motion: reduce) {
  #app-root, #app-root * { transition: none !important; }
}


================================================================================
FILE: Dockerfile
================================================================================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py wsgi.py ./
COPY assets ./assets
COPY src ./src

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 7860)}/healthz', timeout=3)" || exit 1

CMD ["sh", "-c", "gunicorn wsgi:application --bind 0.0.0.0:${PORT:-7860} --workers 2 --threads 4 --timeout 120"]

================================================================================
FILE: LICENSE
================================================================================
MIT License

Copyright (c) 2026 Riddhima Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


================================================================================
FILE: notebooks/week1_data_acquisition.ipynb
================================================================================

--- Cell 1 (markdown) ---
# AlphaPulse — Week 1: Data Acquisition & Integrity Check

**Project:** Infotact Solutions Project 2 (Portfolio Risk & Volatility Monitor)  
**Primary Source:** Twelve Data API (with `yfinance` fallback)  
**Targets:** Local Parquet/CSV + **MongoDB Compass** (`alphapulse.market_prices`) + **Hugging Face Hub**

--- Cell 2 (code) ---
import sys
from pathlib import Path

# Ensure project root is on sys.path when running inside notebooks/
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

--- Cell 3 (code) ---
import time

import polars as pl

from scripts.week1_ingest import (
    align_and_clean,
    load_asset_series,
    push_to_huggingface,
    run_integrity_checks,
    save_to_mongodb,
)
from src.config import FX_ASSET, PORTFOLIO_ASSETS, PROCESSED_DIR, get_settings

settings = get_settings()
print(f"Loaded settings | Mongo DB: {settings.mongo_db} | HF Repo: {settings.hf_repo_id}")
td_ok = bool(settings.twelve_data_api_key and "your_" not in settings.twelve_data_api_key)
print(f"Twelve Data key configured: {td_ok}")

--- Cell 4 (markdown) ---
## 1. Fetch Historical OHLCV Data (Twelve Data + Fallback)
Pulls 5 years (`1260` trading days) of corporate-action-adjusted daily bars for `USD/INR`, Indian NSE equities, Gold BeES, and `BTC/USD`.

--- Cell 5 (code) ---
fx_df = load_asset_series(FX_ASSET, settings)
asset_frames = []
for asset in PORTFOLIO_ASSETS:
    df_asset = load_asset_series(asset, settings)
    asset_frames.append(df_asset)
    time.sleep(1.0)

print(f"Fetched FX ({fx_df.height} rows) + {len(asset_frames)} portfolio assets.")

--- Cell 6 (markdown) ---
## 2. INR Normalization, Forward-Fill Alignment & `float32` Optimization

--- Cell 7 (code) ---
cleaned_df = align_and_clean(asset_frames, fx_df)
run_integrity_checks(cleaned_df)

print(cleaned_df.schema)
print(f"Estimated memory: {cleaned_df.estimated_size('mb'):.3f} MB")
cleaned_df.group_by("asset_id", "sector", "source").agg(
    pl.len().alias("rows"),
    pl.col("date").min().alias("start_date"),
    pl.col("date").max().alias("end_date"),
    pl.col("close").min().alias("min_inr"),
    pl.col("close").max().alias("max_inr"),
).sort("asset_id")

--- Cell 8 (markdown) ---
## 3. Corporate Action & Split Continuity Audit

--- Cell 9 (code) ---
jumps = (
    cleaned_df.sort(["asset_id", "date"])
    .with_columns((pl.col("close") / pl.col("close").shift(1).over("asset_id") - 1.0).alias("daily_ret"))
    .filter(pl.col("daily_ret").abs() > 0.60)
)
print(f"Extreme single-day jumps (>60%): {jumps.height}")
if jumps.height > 0:
    display(jumps)

--- Cell 10 (markdown) ---
## 4. Persist Locally, Upsert to MongoDB & Push to Hugging Face

--- Cell 11 (code) ---
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
parquet_path = PROCESSED_DIR / "market_prices.parquet"
csv_path = PROCESSED_DIR / "market_prices.csv"

cleaned_df.write_parquet(parquet_path)
cleaned_df.write_csv(csv_path)
print(f"Saved local Parquet & CSV to {PROCESSED_DIR}")

save_to_mongodb(cleaned_df, settings)
push_to_huggingface(str(parquet_path), settings)

================================================================================
FILE: notebooks/week2_quant_risk.ipynb
================================================================================

--- Cell 1 (markdown) ---
# AlphaPulse — Week 2: Quantitative Risk & Monte Carlo Simulation

**Project:** Infotact Solutions Project 2 (Portfolio Risk & Volatility Monitor)  
**Core Methods:** Daily Log Returns, 30-Day Rolling Volatility, Vectorized Covariance (`w^T Σ w`), 10,000+ Cholesky Monte Carlo Paths, 95% VaR/CVaR, and Sharpe Ratio

--- Cell 2 (code) ---
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

--- Cell 3 (code) ---
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import PORTFOLIO_ASSETS, get_settings
from src.quant_engine import (
    build_price_matrix,
    compute_log_returns,
    compute_portfolio_kpis,
    compute_rolling_volatility,
    export_tableau_csvs,
    load_market_prices,
    push_quant_to_huggingface,
    run_monte_carlo,
    save_quant_to_mongodb,
)

settings = get_settings()
raw_df = load_market_prices()
price_matrix = build_price_matrix(raw_df)
log_returns = compute_log_returns(price_matrix)
rolling_vol = compute_rolling_volatility(log_returns, window=30)
corr_matrix = log_returns.corr()

weights = np.array([a.weight for a in PORTFOLIO_ASSETS if a.asset_id in price_matrix.columns])
print(f"Price matrix: {price_matrix.shape} | Log returns: {log_returns.shape}")

--- Cell 4 (markdown) ---
## 1. Portfolio KPIs: Annualized Return, Volatility, Sharpe & 95% VaR/CVaR

--- Cell 5 (code) ---
kpis = compute_portfolio_kpis(log_returns, weights, confidence=settings.var_confidence)
for k, v in kpis.items():
    print(f"{k:25s}: {v}")

--- Cell 6 (markdown) ---
## 2. Correlation Heatmap & 30-Day Rolling Volatility

--- Cell 7 (code) ---
fig_corr = px.imshow(
    corr_matrix.round(2),
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Cross-Asset Return Correlation Matrix (INR)",
)
fig_corr.show()

fig_vol = px.line(
    rolling_vol,
    title="30-Day Annualized Rolling Volatility",
    labels={"value": "Annualized Volatility", "date": "Date", "variable": "Asset"},
)
fig_vol.show()

--- Cell 8 (markdown) ---
## 3. 10,000-Path Cholesky Monte Carlo Simulation & Statistical Sanity Audit

--- Cell 9 (code) ---
mc_result = run_monte_carlo(
    log_returns,
    weights,
    n_simulations=settings.mc_simulations,
    horizon_days=252,
    initial_capital=1_000_000.0,
    confidence=settings.var_confidence,
)

sim_mean_ret = float(mc_result["terminal_returns"].mean())
print(f"Historical Annualized Return : {kpis['annualized_return']:.2%}")
print(f"Simulated Mean 1Y Return     : {sim_mean_ret:.2%}")
print(f"1Y 95% Monte Carlo VaR (INR) : ₹{mc_result['mc_var_95_horizon_inr']:,.2f}")

fan = mc_result["percentile_fan"]
fig_fan = go.Figure()
fig_fan.add_trace(go.Scatter(x=fan["day"], y=fan["p95"], name="95th Pct", line={"dash": "dot"}))
fig_fan.add_trace(go.Scatter(x=fan["day"], y=fan["p50"], name="Median (P50)", line={"width": 3}))
fig_fan.add_trace(go.Scatter(x=fan["day"], y=fan["p05"], name="5th Pct (95% VaR)", line={"dash": "dot"}))
fig_fan.update_layout(title="Monte Carlo 1-Year Portfolio Fan (10,000 Paths, ₹10L Initial)")
fig_fan.show()

--- Cell 10 (markdown) ---
## 4. Export Tableau CSVs, Save to MongoDB Compass & Push to Hugging Face

--- Cell 11 (code) ---
export_tableau_csvs(price_matrix, log_returns, rolling_vol, corr_matrix, mc_result)
save_quant_to_mongodb(kpis, corr_matrix, mc_result, settings)
push_quant_to_huggingface(settings)

================================================================================
FILE: pyproject.toml
================================================================================
[project]
name = "alphapulse"
version = "0.1.0"
description = "AlphaPulse: Portfolio Risk & Volatility Monitor (Infotact Solutions Project 2)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "dash>=2.18.0",
    "dash-bootstrap-components>=1.6.0",
    "datasets>=3.0.0",
    "gunicorn>=23.0.0",
    "httpx>=0.27.0",
    "huggingface-hub>=0.26.0",
    "nbformat>=5.11.1",
    "numpy>=2.1.0",
    "pandas>=2.2.0",
    "plotly>=5.24.0",
    "polars>=1.12.0",
    "pyarrow>=18.0.0",
    "pydantic-settings>=2.6.0",
    "pymongo>=4.10.0",
    "scipy>=1.14.0",
    "twelvedata>=1.2.28",
    "yfinance>=0.2.50",
]

[dependency-groups]
dev = [
    "ipykernel>=6.29.5",
    "ipywidgets>=8.1.9",
    "pytest>=8.3.0",
    "ruff>=0.8.0",
]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]


================================================================================
FILE: README.md
================================================================================

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
- **Stress Test:** Instant sector shocks (*IT −15%*, *Crypto −40%*, *Energy −20%*, *Gold +15%*) + `@lru_cache` **10,000-path Cholesky Monte Carlo** (`30D / 90D / 1Y`).
- **Risk & Technicals:** Correlation heatmap, risk-vs-return map, 30D rolling volatility regimes, and single-asset technical tear-sheet.
- **Engineering Stack:** Python 3.11 (`uv`), `polars`, `numpy`, `scipy`, **Plotly Dash** (deployed on **Plotly Cloud**), **Twelve Data** + `yfinance`, **MongoDB**, **Hugging Face Hub**, Tableau CSVs, `ruff` (0 warnings), `pytest` (7/7 passing).

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

Metrics use daily log returns (`252` trading days/yr) with fixed weights, zero transaction costs, and `forward_fill()` holiday alignment; Gaussian Cholesky paths can understate crisis fat tails. Built strictly for financial analytics—not investment advice.

**Riddhima Singh** · [GitHub](https://github.com/Riddhis2226) · *Infotact Solutions Enterprise Data Analytics Program (Q4 2025)*

================================================================================
FILE: requirements.txt
================================================================================
dash>=2.18.0
dash-bootstrap-components>=1.6.0
gunicorn>=23.0.0
httpx>=0.27.0
huggingface-hub>=0.26.0
numpy>=2.1.0
pandas>=2.2.0
plotly>=5.24.0
polars>=1.12.0
pyarrow>=18.0.0
pydantic-settings>=2.6.0
pymongo>=4.10.0
scipy>=1.14.0


================================================================================
FILE: scripts/run_pipeline.py
================================================================================
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

================================================================================
FILE: scripts/week1_ingest.py
================================================================================
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

================================================================================
FILE: src/__init__.py
================================================================================


================================================================================
FILE: src/config.py
================================================================================
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

================================================================================
FILE: src/quant_engine.py
================================================================================
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

================================================================================
FILE: tests/__init__.py
================================================================================


================================================================================
FILE: tests/test_week1_integrity.py
================================================================================
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

================================================================================
FILE: tests/test_week2_quant.py
================================================================================
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

================================================================================
FILE: wsgi.py
================================================================================
"""Production WSGI entrypoint for AlphaPulse (Gunicorn / Render / Docker)."""

from app import server as application

__all__ = ["application"]
