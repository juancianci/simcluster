"""
CCA Agent-Based Model — Interactive Streamlit Dashboard
=======================================================
Launch:  streamlit run cca_abm/dashboard.py
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Ensure package-relative imports work when run as `streamlit run` ──────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cca_abm.engine import SupplyScheduleType
from cca_abm.simulation import (
    SimulationConfig,
    SimulationResult,
    run_single_simulation,
    run_monte_carlo,
    parameter_sweep,
    summarize_single_run,
    AGENT_TYPES,
)

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

AGENT_COLORS = {
    "EarlyBeliever": "#2196F3",
    "InformedTrader": "#4CAF50",
    "MomentumTrader": "#FF9800",
    "OptionValueOptimizer": "#9C27B0",
    "NoiseTrader": "#9E9E9E",
    "WhaleAgent": "#F44336",
}
AGENT_SHORT = {
    "EarlyBeliever": "Early",
    "InformedTrader": "Informed",
    "MomentumTrader": "Momentum",
    "OptionValueOptimizer": "OptVal",
    "NoiseTrader": "Noise",
    "WhaleAgent": "Whale",
}

PLOTLY_TEMPLATE = "plotly_white"

PRESETS: Dict[str, Dict[str, Any]] = {
    "Baseline": dict(
        T=24, total_supply=5000.0, supply_type="uniform",
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, valuation_dispersion=0.2,
        budget_mean=1200.0, budget_std=400.0, whale_budget_multiplier=10.0,
        sentiment=0.55, leakage_rate=0.05, sentiment_vol=0.05,
        shock_probability=0.05, shock_magnitude=0.3,
        seed=42,
    ),
    "Euphoria": dict(
        T=24, total_supply=5000.0, supply_type="uniform",
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, valuation_dispersion=0.2,
        budget_mean=2000.0, budget_std=400.0, whale_budget_multiplier=10.0,
        sentiment=0.9, leakage_rate=0.01, sentiment_vol=0.02,
        shock_probability=0.05, shock_magnitude=0.3,
        seed=42,
    ),
    "Panic": dict(
        T=24, total_supply=5000.0, supply_type="uniform",
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, valuation_dispersion=0.2,
        budget_mean=600.0, budget_std=400.0, whale_budget_multiplier=10.0,
        sentiment=0.15, leakage_rate=0.2, sentiment_vol=0.12,
        shock_probability=0.15, shock_magnitude=0.3,
        seed=42,
    ),
    "Whale-Dominated": dict(
        T=24, total_supply=5000.0, supply_type="uniform",
        n_early=3, n_informed=5, n_momentum=3, n_option=3, n_noise=5, n_whale=8,
        base_valuation=10.0, valuation_dispersion=0.2,
        budget_mean=1200.0, budget_std=400.0, whale_budget_multiplier=20.0,
        sentiment=0.5, leakage_rate=0.05, sentiment_vol=0.05,
        shock_probability=0.05, shock_magnitude=0.3,
        seed=42,
    ),
    "Retail-Heavy": dict(
        T=24, total_supply=5000.0, supply_type="uniform",
        n_early=5, n_informed=2, n_momentum=5, n_option=1, n_noise=60, n_whale=0,
        base_valuation=10.0, valuation_dispersion=0.2,
        budget_mean=1200.0, budget_std=400.0, whale_budget_multiplier=10.0,
        sentiment=0.5, leakage_rate=0.05, sentiment_vol=0.05,
        shock_probability=0.05, shock_magnitude=0.3,
        seed=42,
    ),
}

SUPPLY_TYPES = ["uniform", "front_loaded", "back_loaded", "bell_curve"]


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _build_config(params: dict) -> SimulationConfig:
    stype = SupplyScheduleType(params["supply_type"])
    return SimulationConfig(
        T=params["T"],
        total_supply=params["total_supply"],
        supply_type=stype,
        n_early=params["n_early"],
        n_informed=params["n_informed"],
        n_momentum=params["n_momentum"],
        n_option=params["n_option"],
        n_noise=params["n_noise"],
        n_whale=params["n_whale"],
        base_valuation=params["base_valuation"],
        valuation_dispersion=params["valuation_dispersion"],
        budget_mean=params["budget_mean"],
        budget_std=params["budget_std"],
        whale_budget_multiplier=params["whale_budget_multiplier"],
        sentiment=params["sentiment"],
        leakage_rate=params["leakage_rate"],
        sentiment_vol=params["sentiment_vol"],
        shock_probability=params["shock_probability"],
        shock_magnitude=params["shock_magnitude"],
        seed=params["seed"],
        price_bounds=(0.1, 50.0),
    )


def _agent_color_list():
    """Return (names, colors) in canonical order for Plotly."""
    names = [AGENT_SHORT[a] for a in AGENT_TYPES]
    colors = [AGENT_COLORS[a] for a in AGENT_TYPES]
    return names, colors


# ══════════════════════════════════════════════════════════════════════════
# Plotly chart builders
# ══════════════════════════════════════════════════════════════════════════

def fig_clearing_prices(result: SimulationResult) -> go.Figure:
    T = result.config.T
    V = result.fundamental_value
    periods = list(range(T))
    prices = result.clearing_prices.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=prices, mode="lines+markers",
        name="Clearing Price P<sub>k</sub>",
        line=dict(color="#1565C0", width=2.5), marker=dict(size=6),
    ))
    fig.add_hline(y=V, line_dash="dash", line_color="#E53935", line_width=2,
                  annotation_text=f"V = {V:.2f}", annotation_position="top left")
    # Shaded region
    fig.add_trace(go.Scatter(
        x=periods + periods[::-1],
        y=prices + [V] * T,
        fill="toself", fillcolor="rgba(21,101,192,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title="Clearing Price Dynamics",
        xaxis_title="Period k", yaxis_title="Price",
        template=PLOTLY_TEMPLATE, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_supply_schedule(result: SimulationResult) -> go.Figure:
    T = result.config.T
    periods = list(range(T))
    supply = result.supply_schedule.tolist()
    cum = np.cumsum(result.supply_schedule).tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=periods, y=supply, name="q<sub>k</sub>",
        marker_color="#26A69A", opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=periods, y=cum, name="Cumulative",
        line=dict(color="#FF6F00", width=2), marker=dict(size=5, symbol="square"),
        mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(
        title="Supply Release Schedule", template=PLOTLY_TEMPLATE, height=380,
        xaxis_title="Period k",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Supply per period", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative supply", secondary_y=True)
    return fig


def fig_entry_timing(result: SimulationResult) -> go.Figure:
    T = result.config.T
    periods = list(range(T))
    entry_data = {a: np.zeros(T) for a in AGENT_TYPES}
    for bid in result.bids:
        if bid.agent_type in entry_data and 0 <= bid.entry_period < T:
            entry_data[bid.agent_type][bid.entry_period] += 1

    fig = go.Figure()
    for atype in AGENT_TYPES:
        counts = entry_data[atype]
        if counts.sum() > 0:
            fig.add_trace(go.Bar(
                x=periods, y=counts.tolist(), name=AGENT_SHORT[atype],
                marker_color=AGENT_COLORS[atype], opacity=0.85,
            ))
    fig.update_layout(
        barmode="stack",
        title="Entry Timing by Agent Type",
        xaxis_title="Entry Period τ", yaxis_title="Number of Bids",
        template=PLOTLY_TEMPLATE, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_info_sentiment(result: SimulationResult) -> go.Figure:
    T = result.config.T
    V = result.fundamental_value
    periods = list(range(T))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=periods, y=result.sentiment_path.tolist(),
        name="Sentiment", mode="lines+markers",
        line=dict(color="#FF9800", width=2), marker=dict(size=5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=periods, y=result.public_signals.tolist(),
        name="Public Signal", mode="lines+markers",
        line=dict(color="#9C27B0", width=2), marker=dict(size=5, symbol="square"),
    ), secondary_y=True)
    fig.add_hline(y=V, line_dash="dash", line_color="#E53935", line_width=1.5,
                  secondary_y=True, annotation_text="V")
    fig.update_layout(
        title="Information & Sentiment", template=PLOTLY_TEMPLATE, height=380,
        xaxis_title="Period k",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Sentiment Index", range=[-0.05, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="Public Signal / V", secondary_y=True)
    return fig


def fig_surplus_boxplot(result: SimulationResult) -> go.Figure:
    rows = []
    for s in result.surplus_results:
        rows.append({
            "Type": AGENT_SHORT.get(s.agent_type, s.agent_type),
            "Surplus/Dollar": s.surplus_per_dollar,
            "agent_type": s.agent_type,
        })
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    order = [AGENT_SHORT[a] for a in AGENT_TYPES if AGENT_SHORT[a] in df["Type"].values]
    color_map = {AGENT_SHORT[k]: v for k, v in AGENT_COLORS.items()}

    fig = go.Figure()
    for t in order:
        sub = df[df["Type"] == t]["Surplus/Dollar"]
        fig.add_trace(go.Box(
            y=sub, name=t, marker_color=color_map.get(t, "#999"),
            boxmean=True, jitter=0.3, pointpos=-1.5,
        ))
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.4)
    fig.update_layout(
        title="Surplus per Dollar by Agent Type",
        yaxis_title="Surplus / Dollar Spent",
        template=PLOTLY_TEMPLATE, height=400, showlegend=False,
    )
    return fig


def fig_total_surplus_bar(result: SimulationResult) -> go.Figure:
    totals = {}
    for atype in AGENT_TYPES:
        totals[atype] = sum(s.total_surplus for s in result.surplus_results if s.agent_type == atype)
    present = [a for a in AGENT_TYPES if any(s.agent_type == a for s in result.surplus_results)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[AGENT_SHORT[a] for a in present],
        y=[totals[a] for a in present],
        marker_color=[AGENT_COLORS[a] for a in present],
        opacity=0.85,
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.4)
    fig.update_layout(
        title="Total Surplus by Agent Type",
        yaxis_title="Surplus (quote asset)",
        template=PLOTLY_TEMPLATE, height=400, showlegend=False,
    )
    return fig


def fig_exec_price_vs_valuation(result: SimulationResult) -> go.Figure:
    fig = go.Figure()
    for atype in AGENT_TYPES:
        xs, ys, sizes, hovers = [], [], [], []
        for s in result.surplus_results:
            if s.agent_type == atype and s.total_spend > 0:
                xs.append(s.valuation)
                ys.append(s.avg_execution_price)
                sizes.append(max(s.total_spend / 30, 5))
                hovers.append(
                    f"<b>{AGENT_SHORT[atype]}</b><br>"
                    f"Valuation: {s.valuation:.2f}<br>"
                    f"Avg Exec Price: {s.avg_execution_price:.2f}<br>"
                    f"Surplus: {s.total_surplus:.1f}<br>"
                    f"Tokens: {s.total_tokens:.1f}<br>"
                    f"Spend: {s.total_spend:.0f}"
                )
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                name=AGENT_SHORT[atype],
                marker=dict(size=sizes, color=AGENT_COLORS[atype], opacity=0.65,
                            line=dict(width=0.5, color="white")),
                text=hovers, hoverinfo="text",
            ))
    # Breakeven line
    all_vals = [s.valuation for s in result.surplus_results if s.total_spend > 0]
    if all_vals:
        mn, mx = min(all_vals) * 0.9, max(all_vals) * 1.1
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="grey", dash="dash", width=1.5),
            name="V = P (breakeven)", showlegend=True,
        ))
    fig.update_layout(
        title="Execution Price vs Valuation",
        xaxis_title="Agent Valuation V", yaxis_title="Avg Execution Price",
        template=PLOTLY_TEMPLATE, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_budget_schedule_heatmap(result: SimulationResult) -> go.Figure:
    """Heatmap showing budget allocation across periods for each bid."""
    bids = result.bids
    T = result.config.T
    if not bids:
        return go.Figure()

    n_bids = min(len(bids), 60)  # cap for readability
    labels = []
    data = np.zeros((n_bids, T))
    for i, bid in enumerate(bids[:n_bids]):
        sched = result.auction_result.budget_schedules[bid.bid_id]
        data[i, :] = sched
        labels.append(f"{AGENT_SHORT.get(bid.agent_type, '?')} #{bid.bid_id}")

    fig = go.Figure(go.Heatmap(
        z=data, x=list(range(T)), y=labels,
        colorscale="Blues", colorbar_title="Budget $",
        hovertemplate="Bid: %{y}<br>Period: %{x}<br>Budget: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title="Budget Allocation Heatmap (Spreading Rule)",
        xaxis_title="Period k", yaxis_title="Bid",
        template=PLOTLY_TEMPLATE, height=max(300, n_bids * 14 + 100),
    )
    return fig


def fig_cumulative_participation(result: SimulationResult) -> go.Figure:
    T = result.config.T
    periods = list(range(T))
    cum_total = np.zeros(T)
    cum_by_type = {a: np.zeros(T) for a in AGENT_TYPES}

    for bid in result.bids:
        if 0 <= bid.entry_period < T:
            cum_total[bid.entry_period] += 1
            if bid.agent_type in cum_by_type:
                cum_by_type[bid.agent_type][bid.entry_period] += 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=np.cumsum(cum_total).tolist(),
        name="All Bids", mode="lines+markers",
        line=dict(color="#333", width=3), marker=dict(size=5),
    ))
    for atype in AGENT_TYPES:
        arr = np.cumsum(cum_by_type[atype])
        if arr[-1] > 0:
            fig.add_trace(go.Scatter(
                x=periods, y=arr.tolist(),
                name=AGENT_SHORT[atype], mode="lines",
                line=dict(color=AGENT_COLORS[atype], width=1.5, dash="dot"),
            ))
    fig.update_layout(
        title="Cumulative Participation",
        xaxis_title="Period k", yaxis_title="Cumulative Bids",
        template=PLOTLY_TEMPLATE, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_per_period_demand_supply(result: SimulationResult) -> go.Figure:
    """Per-period demand at clearing vs supply."""
    T = result.config.T
    periods = list(range(T))
    supply = result.supply_schedule.tolist()
    demand = [pr.total_demand_at_clearing for pr in result.auction_result.period_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=periods, y=supply, name="Supply q<sub>k</sub>",
                         marker_color="#26A69A", opacity=0.7))
    fig.add_trace(go.Scatter(x=periods, y=demand, name="Demand at P<sub>k</sub>",
                             mode="lines+markers", line=dict(color="#D32F2F", width=2),
                             marker=dict(size=6)))
    fig.update_layout(
        title="Supply vs Demand at Clearing",
        xaxis_title="Period k", yaxis_title="Tokens",
        template=PLOTLY_TEMPLATE, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Sweep charts ──────────────────────────────────────────────────────────

def fig_sweep_price(df: pd.DataFrame, param_name: str, V: float) -> go.Figure:
    g = df.groupby("param_value")["mean_clearing_price"]
    means, stds = g.mean(), g.std().fillna(0)
    vals = sorted(df["param_value"].unique())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals, y=[means[v] for v in vals],
        error_y=dict(type="data", array=[stds[v] for v in vals], visible=True),
        mode="lines+markers", line=dict(color="#1565C0", width=2.5),
        marker=dict(size=7), name="Mean Price",
    ))
    fig.add_hline(y=V, line_dash="dash", line_color="#E53935", line_width=2,
                  annotation_text=f"V = {V:.1f}")
    fig.update_layout(
        title="Mean Clearing Price", xaxis_title=param_name,
        yaxis_title="Mean Clearing Price", template=PLOTLY_TEMPLATE, height=370,
    )
    return fig


def fig_sweep_volatility(df: pd.DataFrame, param_name: str) -> go.Figure:
    g = df.groupby("param_value")["price_volatility"]
    means, stds = g.mean(), g.std().fillna(0)
    vals = sorted(df["param_value"].unique())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals, y=[means[v] for v in vals],
        error_y=dict(type="data", array=[stds[v] for v in vals], visible=True),
        mode="lines+markers", line=dict(color="#FF6F00", width=2.5),
        marker=dict(size=7, symbol="square"), name="Volatility",
    ))
    fig.update_layout(
        title="Price Volatility", xaxis_title=param_name,
        yaxis_title="Std of Clearing Prices", template=PLOTLY_TEMPLATE, height=370,
    )
    return fig


def fig_sweep_surplus(df: pd.DataFrame, param_name: str) -> go.Figure:
    vals = sorted(df["param_value"].unique())
    fig = go.Figure()
    for atype in AGENT_TYPES:
        col = f"avg_surplus_{atype}"
        if col not in df.columns:
            continue
        g = df.groupby("param_value")[col].mean()
        y = [g.get(v, 0) for v in vals]
        if any(abs(yy) > 1e-6 for yy in y):
            fig.add_trace(go.Scatter(
                x=vals, y=y, name=AGENT_SHORT[atype], mode="lines+markers",
                line=dict(color=AGENT_COLORS[atype], width=2),
                marker=dict(size=6),
            ))
    fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.4)
    fig.update_layout(
        title="Avg Surplus per Agent by Type", xaxis_title=param_name,
        yaxis_title="Avg Surplus", template=PLOTLY_TEMPLATE, height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_sweep_participation(df: pd.DataFrame, param_name: str) -> go.Figure:
    g = df.groupby("param_value")["n_bids"]
    means, stds = g.mean(), g.std().fillna(0)
    vals = sorted(df["param_value"].unique())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals, y=[means[v] for v in vals],
        error_y=dict(type="data", array=[stds[v] for v in vals], visible=True),
        mode="lines+markers", line=dict(color="#4CAF50", width=2.5),
        marker=dict(size=7, symbol="diamond"), name="Bids",
    ))
    fig.update_layout(
        title="Participation", xaxis_title=param_name,
        yaxis_title="Number of Bids", template=PLOTLY_TEMPLATE, height=370,
    )
    return fig


# ── Scenario comparison charts ────────────────────────────────────────────

def fig_compare_prices(results: Dict[str, SimulationResult], V: float) -> go.Figure:
    colors = px.colors.qualitative.Set1
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        T = res.config.T
        fig.add_trace(go.Scatter(
            x=list(range(T)), y=res.clearing_prices.tolist(),
            name=name, mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=5),
        ))
    fig.add_hline(y=V, line_dash="dash", line_color="#E53935", line_width=2,
                  annotation_text="V")
    fig.update_layout(
        title="Clearing Price Paths", xaxis_title="Period k", yaxis_title="Clearing Price",
        template=PLOTLY_TEMPLATE, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_compare_surplus(results: Dict[str, SimulationResult]) -> go.Figure:
    scenario_names = list(results.keys())
    fig = go.Figure()
    for atype in AGENT_TYPES:
        vals = []
        for name in scenario_names:
            res = results[name]
            vals.append(sum(s.total_surplus for s in res.surplus_results if s.agent_type == atype))
        if any(abs(v) > 1e-6 for v in vals):
            fig.add_trace(go.Bar(
                x=scenario_names, y=vals, name=AGENT_SHORT[atype],
                marker_color=AGENT_COLORS[atype], opacity=0.85,
            ))
    fig.update_layout(
        barmode="stack", title="Total Surplus by Agent Type",
        yaxis_title="Total Surplus", template=PLOTLY_TEMPLATE, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.4)
    return fig


def fig_compare_participation(results: Dict[str, SimulationResult]) -> go.Figure:
    colors = px.colors.qualitative.Set1
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        T = res.config.T
        cum = np.zeros(T)
        for bid in res.bids:
            if 0 <= bid.entry_period < T:
                cum[bid.entry_period] += 1
        fig.add_trace(go.Scatter(
            x=list(range(T)), y=np.cumsum(cum).tolist(),
            name=name, mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=5),
        ))
    fig.update_layout(
        title="Cumulative Participation", xaxis_title="Period k",
        yaxis_title="Cumulative Bids", template=PLOTLY_TEMPLATE, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Sidebar configuration
# ══════════════════════════════════════════════════════════════════════════

def _sidebar_config() -> dict:
    """Build configuration from sidebar widgets, return params dict."""
    st.sidebar.markdown("## Configuration")

    preset = st.sidebar.selectbox(
        "Preset", list(PRESETS.keys()), index=0,
        help="Load a preset configuration. All parameters below will update.",
    )
    p = copy.deepcopy(PRESETS[preset])

    st.sidebar.markdown("---")

    with st.sidebar.expander("Auction Settings", expanded=False):
        p["T"] = st.slider("Periods (T)", 5, 60, p["T"])
        p["total_supply"] = st.number_input("Total Supply (Q)", 100.0, 100000.0, p["total_supply"], step=500.0)
        p["supply_type"] = st.selectbox("Supply Schedule", SUPPLY_TYPES, index=SUPPLY_TYPES.index(p["supply_type"]))

    with st.sidebar.expander("Agent Composition", expanded=False):
        p["n_early"] = st.slider("Early Believers", 0, 40, p["n_early"])
        p["n_informed"] = st.slider("Informed Traders", 0, 40, p["n_informed"])
        p["n_momentum"] = st.slider("Momentum Traders", 0, 40, p["n_momentum"])
        p["n_option"] = st.slider("Option-Value Optimizers", 0, 40, p["n_option"])
        p["n_noise"] = st.slider("Noise Traders", 0, 80, p["n_noise"])
        p["n_whale"] = st.slider("Whale Agents", 0, 15, p["n_whale"])
        total_agents = p["n_early"] + p["n_informed"] + p["n_momentum"] + p["n_option"] + p["n_noise"] + p["n_whale"]
        st.caption(f"**Total agents: {total_agents}**")

    with st.sidebar.expander("Market Parameters", expanded=False):
        p["base_valuation"] = st.number_input("Fundamental Value (V)", 0.1, 1000.0, p["base_valuation"], step=0.5)
        p["valuation_dispersion"] = st.slider("Valuation Dispersion", 0.01, 1.0, p["valuation_dispersion"], step=0.01)
        p["budget_mean"] = st.number_input("Mean Budget ($)", 50.0, 50000.0, p["budget_mean"], step=100.0)
        p["budget_std"] = st.number_input("Budget Std Dev", 0.0, 20000.0, p["budget_std"], step=50.0)
        p["whale_budget_multiplier"] = st.slider("Whale Budget Multiplier", 1.0, 50.0, p["whale_budget_multiplier"], step=1.0)

    with st.sidebar.expander("Sentiment & Information", expanded=False):
        p["sentiment"] = st.slider("Initial Sentiment", 0.0, 1.0, p["sentiment"], step=0.05)
        p["leakage_rate"] = st.slider("Leakage Rate (λ)", 0.0, 0.5, p["leakage_rate"], step=0.01)
        p["sentiment_vol"] = st.slider("Sentiment Volatility", 0.0, 0.3, p["sentiment_vol"], step=0.01)
        p["shock_probability"] = st.slider("Shock Probability", 0.0, 0.5, p["shock_probability"], step=0.01)
        p["shock_magnitude"] = st.slider("Shock Magnitude", 0.0, 1.0, p["shock_magnitude"], step=0.05)

    with st.sidebar.expander("Simulation", expanded=False):
        p["seed"] = st.number_input("Random Seed", 0, 999999, p["seed"], step=1)

    return p


# ══════════════════════════════════════════════════════════════════════════
# Page: Single Run
# ══════════════════════════════════════════════════════════════════════════

def _page_single_run(params: dict):
    st.header("Single Run Dashboard")

    cfg = _build_config(params)
    with st.spinner("Running simulation..."):
        t0 = time.time()
        result = run_single_simulation(cfg)
        elapsed = time.time() - t0

    summary = summarize_single_run(result)

    # ── KPI row ──
    cols = st.columns(6)
    cols[0].metric("Mean Price", f"{summary['mean_clearing_price']:.3f}",
                   help="Mean of positive clearing prices")
    cols[1].metric("Final Price", f"{summary['final_clearing_price']:.3f}")
    cols[2].metric("Price Vol", f"{summary['price_volatility']:.3f}",
                   help="Std of clearing prices")
    cols[3].metric("Price Drift", f"{summary['price_drift']:+.1%}",
                   help="(P_T / P_1) - 1")
    cols[4].metric("Total Surplus", f"{summary['total_surplus']:,.0f}")
    cols[5].metric("Bids", f"{summary['n_bids']}")

    st.caption(f"Simulation completed in {elapsed:.2f}s  ·  V = {result.fundamental_value:.2f}  ·  T = {cfg.T}  ·  Q = {cfg.total_supply:,.0f}")

    # ── Main charts ──
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig_clearing_prices(result), use_container_width=True)
    with c2:
        st.plotly_chart(fig_supply_schedule(result), use_container_width=True)

    c3, c4 = st.columns([2, 1])
    with c3:
        st.plotly_chart(fig_entry_timing(result), use_container_width=True)
    with c4:
        st.plotly_chart(fig_info_sentiment(result), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(fig_surplus_boxplot(result), use_container_width=True)
    with c6:
        st.plotly_chart(fig_total_surplus_bar(result), use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(fig_exec_price_vs_valuation(result), use_container_width=True)
    with c8:
        st.plotly_chart(fig_cumulative_participation(result), use_container_width=True)

    # ── Additional analysis ──
    with st.expander("Budget Allocation Heatmap", expanded=False):
        st.plotly_chart(fig_budget_schedule_heatmap(result), use_container_width=True)

    with st.expander("Supply vs Demand at Clearing", expanded=False):
        st.plotly_chart(fig_per_period_demand_supply(result), use_container_width=True)

    # ── Bid-level details table ──
    with st.expander("Bid-Level Results Table", expanded=False):
        rows = []
        for s in result.surplus_results:
            rows.append({
                "Bid ID": s.bid_id,
                "Agent ID": s.agent_id,
                "Type": AGENT_SHORT.get(s.agent_type, s.agent_type),
                "Entry τ": s.entry_period,
                "Valuation": round(s.valuation, 3),
                "Cap": round(s.cap, 3),
                "Budget": round(s.budget, 1),
                "Spend": round(s.total_spend, 1),
                "Unspent": round(s.unspent_budget, 1),
                "Tokens": round(s.total_tokens, 2),
                "Avg Price": round(s.avg_execution_price, 3),
                "Surplus": round(s.total_surplus, 1),
                "Surplus/$": round(s.surplus_per_dollar, 4),
            })
        df_bids = pd.DataFrame(rows)
        st.dataframe(df_bids, use_container_width=True, height=400)

    # ── Per-type summary ──
    with st.expander("Agent Type Summary", expanded=False):
        type_rows = []
        for atype in AGENT_TYPES:
            subs = [s for s in result.surplus_results if s.agent_type == atype]
            if not subs:
                continue
            type_rows.append({
                "Type": AGENT_SHORT[atype],
                "Count": len(subs),
                "Total Surplus": round(sum(s.total_surplus for s in subs), 1),
                "Avg Surplus": round(sum(s.total_surplus for s in subs) / len(subs), 1),
                "Avg Surplus/$": round(sum(s.surplus_per_dollar for s in subs) / len(subs), 4),
                "Total Spend": round(sum(s.total_spend for s in subs), 0),
                "Total Tokens": round(sum(s.total_tokens for s in subs), 1),
                "Avg Entry": round(sum(s.entry_period for s in subs) / len(subs), 1),
            })
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Page: Parameter Sweep
# ══════════════════════════════════════════════════════════════════════════

SWEEP_PARAMS = {
    "sentiment": {"label": "Sentiment", "default_values": "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9"},
    "budget_mean": {"label": "Budget Mean ($)", "default_values": "300, 500, 750, 1000, 1500, 2000, 3000"},
    "leakage_rate": {"label": "Leakage Rate (λ)", "default_values": "0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4"},
    "n_informed": {"label": "Informed Traders Count", "default_values": "0, 3, 6, 10, 15, 20, 30"},
    "n_noise": {"label": "Noise Traders Count", "default_values": "5, 10, 20, 30, 40, 60"},
    "n_whale": {"label": "Whale Count", "default_values": "0, 1, 2, 4, 6, 8"},
    "valuation_dispersion": {"label": "Valuation Dispersion", "default_values": "0.05, 0.1, 0.2, 0.3, 0.5"},
    "total_supply": {"label": "Total Supply (Q)", "default_values": "1000, 2000, 3000, 5000, 8000, 12000"},
    "whale_budget_multiplier": {"label": "Whale Budget Mult", "default_values": "2, 5, 10, 15, 20, 30"},
    "sentiment_vol": {"label": "Sentiment Volatility", "default_values": "0.0, 0.02, 0.05, 0.1, 0.15, 0.2"},
    "shock_probability": {"label": "Shock Probability", "default_values": "0.0, 0.02, 0.05, 0.1, 0.2, 0.3"},
}


def _page_sweep(params: dict):
    st.header("Parameter Sweep")

    c1, c2, c3 = st.columns([2, 3, 1])
    with c1:
        sweep_key = st.selectbox("Parameter to sweep", list(SWEEP_PARAMS.keys()),
                                 format_func=lambda k: SWEEP_PARAMS[k]["label"])
    with c2:
        default_vals = SWEEP_PARAMS[sweep_key]["default_values"]
        values_str = st.text_input("Values (comma-separated)", default_vals)
    with c3:
        n_runs = st.number_input("Runs per value", 1, 50, 10, step=1)

    try:
        values = [float(v.strip()) for v in values_str.split(",")]
        # If the param is an int type, convert
        if sweep_key.startswith("n_"):
            values = [int(v) for v in values]
    except ValueError:
        st.error("Could not parse values. Use comma-separated numbers.")
        return

    if st.button("Run Sweep", type="primary", use_container_width=True):
        cfg = _build_config(params)
        V = cfg.base_valuation

        progress = st.progress(0, text="Starting sweep...")
        total_steps = len(values) * n_runs
        step = 0

        # Manual sweep with progress
        rows = []
        for vi, val in enumerate(values):
            cfg_copy = copy.copy(cfg)
            setattr(cfg_copy, sweep_key, val)
            cfg_copy.n_runs = n_runs
            mc_results = run_monte_carlo(cfg_copy)
            for run_id, res in enumerate(mc_results):
                row = summarize_single_run(res)
                row["param_name"] = sweep_key
                row["param_value"] = val
                row["run_id"] = run_id
                rows.append(row)
                step += 1
                progress.progress(step / total_steps,
                                  text=f"{SWEEP_PARAMS[sweep_key]['label']} = {val}  ·  run {run_id+1}/{n_runs}")

        progress.empty()
        df = pd.DataFrame(rows)
        st.success(f"Sweep complete: {len(values)} values × {n_runs} runs = {len(df)} data points")

        # ── Charts ──
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_sweep_price(df, sweep_key, V), use_container_width=True)
        with c2:
            st.plotly_chart(fig_sweep_volatility(df, sweep_key), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_sweep_surplus(df, sweep_key), use_container_width=True)
        with c4:
            st.plotly_chart(fig_sweep_participation(df, sweep_key), use_container_width=True)

        # ── Price drift ──
        with st.expander("Price Drift", expanded=False):
            g = df.groupby("param_value")["price_drift"]
            means, stds = g.mean(), g.std().fillna(0)
            vals_sorted = sorted(df["param_value"].unique())
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=vals_sorted, y=[means[v] for v in vals_sorted],
                error_y=dict(type="data", array=[stds[v] for v in vals_sorted], visible=True),
                mode="lines+markers", line=dict(color="#7B1FA2", width=2.5),
            ))
            fig_drift.add_hline(y=0, line_color="black", line_width=1, opacity=0.4)
            fig_drift.update_layout(title="Price Drift (P_T/P_1 - 1)", xaxis_title=sweep_key,
                                    yaxis_title="Drift", template=PLOTLY_TEMPLATE, height=350)
            st.plotly_chart(fig_drift, use_container_width=True)

        # ── Download ──
        st.download_button(
            "Download sweep data (CSV)",
            df.to_csv(index=False).encode(),
            f"sweep_{sweep_key}.csv",
            "text/csv",
        )

        with st.expander("Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════
# Page: Scenario Comparison
# ══════════════════════════════════════════════════════════════════════════

def _page_comparison(params: dict):
    st.header("Scenario Comparison")

    st.markdown("Select scenarios to run side-by-side. Each uses its preset configuration.")

    selected = st.multiselect(
        "Scenarios",
        list(PRESETS.keys()),
        default=["Baseline", "Euphoria", "Panic"],
    )

    if len(selected) < 2:
        st.info("Select at least 2 scenarios to compare.")
        return

    if st.button("Run Scenario Comparison", type="primary", use_container_width=True, key="btn_scenario"):
        results = {}
        progress = st.progress(0, text="Running scenarios...")
        for i, name in enumerate(selected):
            progress.progress(i / len(selected), text=f"Running: {name}...")
            cfg = _build_config(PRESETS[name])
            results[name] = run_single_simulation(cfg)
        progress.empty()

        V = params["base_valuation"]

        # ── Metrics table ──
        metric_rows = []
        for name, res in results.items():
            s = summarize_single_run(res)
            metric_rows.append({
                "Scenario": name,
                "Mean Price": round(s["mean_clearing_price"], 3),
                "Final Price": round(s["final_clearing_price"], 3),
                "Price Vol": round(s["price_volatility"], 3),
                "Drift": f"{s['price_drift']:+.1%}",
                "Bids": s["n_bids"],
                "Total Surplus": round(s["total_surplus"], 0),
            })
        st.dataframe(pd.DataFrame(metric_rows).set_index("Scenario"), use_container_width=True)

        # ── Charts ──
        st.plotly_chart(fig_compare_prices(results, V), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_compare_surplus(results), use_container_width=True)
        with c2:
            st.plotly_chart(fig_compare_participation(results), use_container_width=True)

        # ── Per-scenario type breakdown ──
        with st.expander("Per-Scenario Agent Type Breakdown", expanded=False):
            for name, res in results.items():
                st.subheader(name)
                type_rows = []
                for atype in AGENT_TYPES:
                    subs = [s for s in res.surplus_results if s.agent_type == atype]
                    if not subs:
                        continue
                    type_rows.append({
                        "Type": AGENT_SHORT[atype],
                        "Count": len(subs),
                        "Total Surplus": round(sum(s.total_surplus for s in subs), 1),
                        "Avg Surplus/$": round(sum(s.surplus_per_dollar for s in subs) / len(subs), 4),
                    })
                st.dataframe(pd.DataFrame(type_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Page: Supply Schedule Explorer
# ══════════════════════════════════════════════════════════════════════════

def _page_supply_explorer(params: dict):
    st.header("Supply Schedule Comparison")
    st.markdown("Run the same agent population under different supply release shapes.")

    selected_types = st.multiselect(
        "Schedule types", SUPPLY_TYPES, default=SUPPLY_TYPES,
    )
    if len(selected_types) < 1:
        st.info("Select at least 1 schedule type.")
        return

    if st.button("Run Supply Comparison", type="primary", use_container_width=True, key="btn_supply"):
        results = {}
        for stype in selected_types:
            p = copy.deepcopy(params)
            p["supply_type"] = stype
            cfg = _build_config(p)
            results[stype] = run_single_simulation(cfg)

        V = params["base_valuation"]

        # ── Supply shapes side by side ──
        n = len(selected_types)
        cols = st.columns(min(n, 4))
        for i, stype in enumerate(selected_types):
            res = results[stype]
            with cols[i % len(cols)]:
                fig = go.Figure(go.Bar(
                    x=list(range(res.config.T)),
                    y=res.supply_schedule.tolist(),
                    marker_color="#26A69A",
                ))
                fig.update_layout(title=stype.replace("_", " ").title(), height=220,
                                  template=PLOTLY_TEMPLATE, margin=dict(l=30, r=10, t=40, b=30),
                                  xaxis_title="k", yaxis_title="q_k")
                st.plotly_chart(fig, use_container_width=True)

        # ── Comparison charts ──
        st.plotly_chart(fig_compare_prices(results, V), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_compare_surplus(results), use_container_width=True)
        with c2:
            st.plotly_chart(fig_compare_participation(results), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Page: About
# ══════════════════════════════════════════════════════════════════════════

def _page_about():
    st.header("About the CCA Simulator")

    st.markdown("""
### Continuous Clearing Auction (CCA)

A **Continuous Clearing Auction** is a time-discretized uniform-price auction in which:

1. **Supply** is released over a precommitted schedule — a fixed amount of tokens `q_k` each period
2. **Bids** consist of a budget (dollars to spend) and a cap (maximum acceptable price)
3. **Spreading**: upon entry, the mechanism spreads the bidder's budget across remaining periods proportional to remaining supply
4. **Clearing**: each period, a uniform clearing price is found where aggregate demand equals supply
5. **Execution**: bidders transact if the clearing price is below their cap

### The Two Key Forces

| Force | Description | Agent |
|-------|-------------|-------|
| **Exposure Premium** (Prop. 1) | Early entry → exposure to more clearing periods → access early surplus | EarlyBeliever |
| **Option Value** (Prop. 2) | Waiting → observe signals → condition entry on better information | InformedTrader, OptionValueOptimizer |

### Agent Types

| Type | Strategy | Cap Tightness |
|------|----------|--------------|
| **Early Believer** | High conviction, enters periods 0-2 | Loose (1.3-1.8x) |
| **Informed Trader** | Private Bayesian signals, waits for threshold surplus | Tight (1.1x) |
| **Momentum Trader** | FOMO-driven, enters on price trends | Adaptive |
| **Option-Value Optimizer** | Balances exposure premium vs option value | Very tight (1.05x) |
| **Noise Trader** | Random entry, noisy beliefs | Noisy |
| **Whale Agent** | Splits budget across tranches, impact-aware | Discounted |

### Information Model

- **Public signals** converge to true value V at rate λ (leakage rate)
- **Sentiment** follows a mean-reverting process with jumps
- **Private signals** (InformedTrader only) provide additional precision about V

### How to Use This Dashboard

1. **Single Run**: Configure parameters in the sidebar, examine detailed results
2. **Parameter Sweep**: Pick a parameter, set sweep values, run Monte Carlo trials
3. **Scenario Comparison**: Compare presets (Euphoria, Panic, etc.) side by side
4. **Supply Explorer**: See how different supply shapes affect outcomes

Built by Kosmos Ventures.
""")


# ══════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="CCA Simulator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    /* Tighten spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    /* Metric cards — theme-safe, no hardcoded colors */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Title ──
    st.markdown(
        "<h1 style='margin-bottom:0'>Continuous Clearing Auction Simulator</h1>"
        "<p style='color:#666; margin-top:0'>Agent-Based Model · Interactive Dashboard</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    params = _sidebar_config()

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Run",
        "Parameter Sweep",
        "Scenario Comparison",
        "Supply Explorer",
        "About",
    ])

    with tab1:
        _page_single_run(params)
    with tab2:
        _page_sweep(params)
    with tab3:
        _page_comparison(params)
    with tab4:
        _page_supply_explorer(params)
    with tab5:
        _page_about()


if __name__ == "__main__":
    main()
