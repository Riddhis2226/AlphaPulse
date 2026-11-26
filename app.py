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
    """Load cleaned market_prices.parquet locally or from Hugging Face Hub."""
    local_path = PROCESSED_DIR / "market_prices.parquet"
    if local_path.exists():
        return pl.read_parquet(local_path).sort(["asset_id", "date"])

    settings = get_settings()
    repo_id = settings.hf_repo_id or "RS2226/alphapulse-market-data"
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename="data/market_prices.parquet",
            repo_type="dataset",
            token=settings.hf_token or None,
        )
        return pl.read_parquet(Path(cached_path)).sort(["asset_id", "date"])
    except (HfHubHTTPError, OSError) as exc:
        msg = f"Unable to load market_prices.parquet locally or from Hugging Face ({repo_id}): {exc}"
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
    fig_eff.update_traces(
        textposition="top center",
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
