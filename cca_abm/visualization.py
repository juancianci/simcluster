"""
CCA Visualization
=================
Single-run dashboards, parameter sweep plots, and scenario comparison charts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .simulation import SimulationResult, AGENT_TYPES

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

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

DPI = 180


def _apply_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# 6.1 Single-run dashboard (7 panels)
# ---------------------------------------------------------------------------

def plot_single_run_dashboard(
    result: SimulationResult,
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate the 7-panel single-run dashboard (§6.1)."""
    _apply_style()

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    T = result.config.T
    periods = np.arange(T)
    V = result.fundamental_value
    prices = result.clearing_prices
    supply = result.supply_schedule

    # ---- Panel 1: Clearing Price Dynamics (top-left, spans 2 cols) ----
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(periods, prices, "o-", color="#1565C0", linewidth=2, markersize=5, label="Clearing Price $P_k$")
    ax1.axhline(V, color="#E53935", linestyle="--", linewidth=1.5, label=f"Fundamental V = {V:.2f}")
    ax1.fill_between(periods, prices, V, alpha=0.12, color="#1565C0")
    ax1.set_xlabel("Period $k$")
    ax1.set_ylabel("Price")
    ax1.set_title("Clearing Price Dynamics" + (f" — {title_suffix}" if title_suffix else ""))
    ax1.legend(loc="best")
    ax1.set_xlim(-0.5, T - 0.5)

    # ---- Panel 2: Supply Release Schedule (top-right) ----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(periods, supply, color="#26A69A", alpha=0.8, label="$q_k$")
    ax2r = ax2.twinx()
    ax2r.plot(periods, np.cumsum(supply), "s-", color="#FF6F00", markersize=4, linewidth=1.5, label="Cumulative")
    ax2.set_xlabel("Period $k$")
    ax2.set_ylabel("Supply per period")
    ax2r.set_ylabel("Cumulative supply")
    ax2.set_title("Supply Release Schedule")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # ---- Panel 3: Entry Timing by Agent Type (middle-left, spans 2 cols) ----
    ax3 = fig.add_subplot(gs[1, :2])
    entry_data = {atype: np.zeros(T) for atype in AGENT_TYPES}
    for bid in result.bids:
        if bid.agent_type in entry_data and 0 <= bid.entry_period < T:
            entry_data[bid.agent_type][bid.entry_period] += 1

    bottom = np.zeros(T)
    for atype in AGENT_TYPES:
        counts = entry_data[atype]
        if counts.sum() > 0:
            ax3.bar(periods, counts, bottom=bottom, color=AGENT_COLORS[atype],
                    label=AGENT_SHORT[atype], alpha=0.85)
            bottom += counts
    ax3.set_xlabel("Entry Period $\\tau$")
    ax3.set_ylabel("Number of Bids")
    ax3.set_title("Entry Timing by Agent Type")
    ax3.legend(loc="upper right", fontsize=9, ncol=2)
    ax3.set_xlim(-0.5, T - 0.5)

    # ---- Panel 4: Information & Sentiment (middle-right) ----
    ax4 = fig.add_subplot(gs[1, 2])
    color_sent = "#FF9800"
    color_sig = "#9C27B0"
    ax4.plot(periods, result.sentiment_path, "o-", color=color_sent, markersize=4, label="Sentiment")
    ax4.set_ylabel("Sentiment Index", color=color_sent)
    ax4.tick_params(axis="y", labelcolor=color_sent)
    ax4.set_ylim(-0.05, 1.05)
    ax4r = ax4.twinx()
    ax4r.plot(periods, result.public_signals, "s-", color=color_sig, markersize=4, label="Public Signal")
    ax4r.axhline(V, color="#E53935", linestyle="--", linewidth=1, alpha=0.7)
    ax4r.set_ylabel("Public Signal / V", color=color_sig)
    ax4r.tick_params(axis="y", labelcolor=color_sig)
    ax4.set_xlabel("Period $k$")
    ax4.set_title("Information & Sentiment")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # ---- Panel 5: Surplus Distribution by Type (bottom-left) ----
    ax5 = fig.add_subplot(gs[2, 0])
    surplus_data = []
    for s in result.surplus_results:
        if s.agent_type in AGENT_SHORT:
            surplus_data.append({
                "Type": AGENT_SHORT[s.agent_type],
                "Surplus/Dollar": s.surplus_per_dollar,
                "agent_type": s.agent_type,
            })
    if surplus_data:
        df_surplus = pd.DataFrame(surplus_data)
        type_order = [AGENT_SHORT[a] for a in AGENT_TYPES if AGENT_SHORT[a] in df_surplus["Type"].values]
        palette = {AGENT_SHORT[k]: v for k, v in AGENT_COLORS.items()}
        sns.boxplot(data=df_surplus, x="Type", y="Surplus/Dollar", order=type_order,
                    palette=palette, ax=ax5, fliersize=3)
    ax5.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax5.set_title("Surplus per Dollar by Type")
    ax5.set_xlabel("")
    ax5.tick_params(axis="x", rotation=30)

    # ---- Panel 6: Aggregate Surplus by Type (bottom-center) ----
    ax6 = fig.add_subplot(gs[2, 1])
    type_totals = {}
    for atype in AGENT_TYPES:
        total = sum(s.total_surplus for s in result.surplus_results if s.agent_type == atype)
        type_totals[atype] = total
    types_present = [a for a in AGENT_TYPES if type_totals.get(a, 0) != 0 or
                     any(s.agent_type == a for s in result.surplus_results)]
    if types_present:
        bars = ax6.bar(
            [AGENT_SHORT[a] for a in types_present],
            [type_totals[a] for a in types_present],
            color=[AGENT_COLORS[a] for a in types_present],
            alpha=0.85,
        )
    ax6.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax6.set_title("Total Surplus by Type")
    ax6.set_ylabel("Surplus (quote asset)")
    ax6.tick_params(axis="x", rotation=30)

    # ---- Panel 7: Execution Price vs Valuation (bottom-right) ----
    ax7 = fig.add_subplot(gs[2, 2])
    for atype in AGENT_TYPES:
        xs, ys, ss = [], [], []
        for s in result.surplus_results:
            if s.agent_type == atype and s.total_spend > 0:
                xs.append(s.valuation)
                ys.append(s.avg_execution_price)
                ss.append(max(s.total_spend / 50.0, 5))
        if xs:
            ax7.scatter(xs, ys, s=ss, c=AGENT_COLORS[atype], alpha=0.65,
                        label=AGENT_SHORT[atype], edgecolors="white", linewidth=0.3)

    lims = ax7.get_xlim()
    ax7.plot(lims, lims, "--", color="grey", linewidth=1, alpha=0.6, label="V = P (breakeven)")
    ax7.set_xlim(lims)
    ax7.set_xlabel("Agent Valuation $V$")
    ax7.set_ylabel("Avg Execution Price")
    ax7.set_title("Execution Price vs Valuation")
    ax7.legend(loc="upper left", fontsize=8, ncol=2)

    fig.suptitle(f"CCA Simulation Dashboard" + (f" — {title_suffix}" if title_suffix else ""),
                 fontsize=15, fontweight="bold", y=0.99)

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6.2 Parameter sweep plot (4 panels)
# ---------------------------------------------------------------------------

def plot_parameter_sweep(
    df: pd.DataFrame,
    param_name: str,
    fundamental_value: float = 10.0,
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate the 4-panel parameter sweep visualization (§6.2)."""
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    grouped = df.groupby("param_value")

    param_vals = sorted(df["param_value"].unique())

    # ---- Panel 1: Mean clearing price ----
    ax = axes[0, 0]
    means = grouped["mean_clearing_price"].mean()
    stds = grouped["mean_clearing_price"].std()
    ax.errorbar(param_vals, means[param_vals], yerr=stds[param_vals],
                fmt="o-", capsize=4, linewidth=2, color="#1565C0")
    ax.axhline(fundamental_value, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"V = {fundamental_value:.1f}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean Clearing Price")
    ax.set_title("Mean Clearing Price vs Parameter")
    ax.legend()

    # ---- Panel 2: Price volatility ----
    ax = axes[0, 1]
    means = grouped["price_volatility"].mean()
    stds = grouped["price_volatility"].std()
    ax.errorbar(param_vals, means[param_vals], yerr=stds[param_vals],
                fmt="s-", capsize=4, linewidth=2, color="#FF6F00")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Price Volatility (Std)")
    ax.set_title("Price Volatility vs Parameter")

    # ---- Panel 3: Average surplus per agent by type ----
    ax = axes[1, 0]
    for atype in AGENT_TYPES:
        col = f"avg_surplus_{atype}"
        if col in df.columns:
            type_means = grouped[col].mean()
            if not type_means.isna().all() and type_means.abs().sum() > 0:
                ax.plot(param_vals, type_means[param_vals], "o-",
                        color=AGENT_COLORS[atype], label=AGENT_SHORT[atype], linewidth=1.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Avg Surplus per Agent")
    ax.set_title("Surplus by Agent Type")
    ax.legend(fontsize=9, ncol=2)

    # ---- Panel 4: Number of bids ----
    ax = axes[1, 1]
    means = grouped["n_bids"].mean()
    stds = grouped["n_bids"].std()
    ax.errorbar(param_vals, means[param_vals], yerr=stds[param_vals],
                fmt="D-", capsize=4, linewidth=2, color="#4CAF50")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Number of Bids")
    ax.set_title("Participation vs Parameter")

    fig.suptitle(f"Parameter Sweep: {param_name}" + (f" — {title_suffix}" if title_suffix else ""),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6.3 Scenario comparison plot (3 panels)
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    results: Dict[str, SimulationResult],
    fundamental_value: float = 10.0,
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate the 3-panel scenario comparison (§6.3)."""
    _apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    scenario_colors = plt.cm.Set1(np.linspace(0, 1, max(len(results), 3)))

    # ---- Panel 1: Overlaid clearing price paths ----
    ax = axes[0]
    for idx, (name, res) in enumerate(results.items()):
        T = res.config.T
        ax.plot(range(T), res.clearing_prices, "o-", color=scenario_colors[idx],
                label=name, linewidth=1.5, markersize=4)
    ax.axhline(fundamental_value, color="#E53935", linestyle="--", linewidth=1.5, label="V")
    ax.set_xlabel("Period $k$")
    ax.set_ylabel("Clearing Price")
    ax.set_title("Clearing Price Paths")
    ax.legend(fontsize=9)

    # ---- Panel 2: Surplus by type (stacked bar) ----
    ax = axes[1]
    scenario_names = list(results.keys())
    n_scenarios = len(scenario_names)
    x = np.arange(n_scenarios)
    width = 0.6

    bottom = np.zeros(n_scenarios)
    for atype in AGENT_TYPES:
        vals = []
        for name in scenario_names:
            res = results[name]
            total = sum(s.total_surplus for s in res.surplus_results if s.agent_type == atype)
            vals.append(total)
        vals = np.array(vals)
        if np.any(vals != 0):
            ax.bar(x, vals, width, bottom=bottom, color=AGENT_COLORS[atype],
                   label=AGENT_SHORT[atype], alpha=0.85)
            # Only stack positive values for bottom
            bottom += np.maximum(vals, 0)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right")
    ax.set_ylabel("Total Surplus")
    ax.set_title("Surplus by Agent Type")
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    # ---- Panel 3: Cumulative participation over time ----
    ax = axes[2]
    for idx, (name, res) in enumerate(results.items()):
        T = res.config.T
        cum_bids = np.zeros(T)
        for bid in res.bids:
            if 0 <= bid.entry_period < T:
                cum_bids[bid.entry_period] += 1
        cum_bids = np.cumsum(cum_bids)
        ax.plot(range(T), cum_bids, "o-", color=scenario_colors[idx],
                label=name, linewidth=1.5, markersize=4)
    ax.set_xlabel("Period $k$")
    ax.set_ylabel("Cumulative Bids")
    ax.set_title("Cumulative Participation")
    ax.legend(fontsize=9)

    fig.suptitle(f"Scenario Comparison" + (f" — {title_suffix}" if title_suffix else ""),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig
