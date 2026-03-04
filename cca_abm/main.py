"""
CCA ABM — Main Entry Point
===========================
Runs all experiments from the spec (§5) and saves outputs.

Usage:
    python -m cca_abm.main
"""

from __future__ import annotations

import os
import time
from copy import copy

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np

from .engine import SupplyScheduleType
from .simulation import (
    SimulationConfig,
    parameter_sweep,
    run_single_simulation,
    summarize_single_run,
)
from .visualization import (
    plot_parameter_sweep,
    plot_scenario_comparison,
    plot_single_run_dashboard,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _banner(msg: str):
    width = max(len(msg) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def _print_summary(result):
    s = summarize_single_run(result)
    print(f"  Bids submitted:      {s['n_bids']}")
    print(f"  Mean clearing price: {s['mean_clearing_price']:.4f}")
    print(f"  Final clearing price:{s['final_clearing_price']:.4f}")
    print(f"  Price volatility:    {s['price_volatility']:.4f}")
    print(f"  Price drift:         {s['price_drift']:.4f}")
    print(f"  Total surplus:       {s['total_surplus']:.2f}")
    print(f"  Fundamental V:       {s['fundamental_value']:.2f}")


# ---------------------------------------------------------------------------
# §5.1 Baseline single run
# ---------------------------------------------------------------------------

def experiment_baseline():
    _banner("Experiment 5.1: Baseline Single Run")
    cfg = SimulationConfig(
        T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, sentiment=0.55, leakage_rate=0.05,
        budget_mean=1200.0, budget_std=400.0,
        seed=42,
    )
    result = run_single_simulation(cfg)
    _print_summary(result)
    path = os.path.join(OUTPUT_DIR, "01_baseline_dashboard.png")
    plot_single_run_dashboard(result, title_suffix="Baseline", save_path=path)
    plt.close("all")
    print(f"  Saved: {path}")
    return result


# ---------------------------------------------------------------------------
# §5.2 Sentiment sweep
# ---------------------------------------------------------------------------

def experiment_sentiment_sweep():
    _banner("Experiment 5.2: Sentiment Sweep")
    cfg = SimulationConfig(
        T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, leakage_rate=0.05,
        budget_mean=1200.0, budget_std=400.0,
    )
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t0 = time.time()
    df = parameter_sweep(cfg, "sentiment", values, n_runs_per=15)
    print(f"  Sweep completed in {time.time() - t0:.1f}s ({len(df)} rows)")

    csv_path = os.path.join(OUTPUT_DIR, "sweep_sentiment.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    fig_path = os.path.join(OUTPUT_DIR, "02_sentiment_sweep.png")
    plot_parameter_sweep(df, "sentiment", fundamental_value=10.0, save_path=fig_path)
    plt.close("all")
    print(f"  Saved: {fig_path}")
    return df


# ---------------------------------------------------------------------------
# §5.3 Demand intensity sweep
# ---------------------------------------------------------------------------

def experiment_demand_sweep():
    _banner("Experiment 5.3: Demand Intensity Sweep")
    cfg = SimulationConfig(
        T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, sentiment=0.5, leakage_rate=0.05,
    )
    values = [300, 500, 750, 1000, 1500, 2000, 3000]
    t0 = time.time()
    df = parameter_sweep(cfg, "budget_mean", values, n_runs_per=15)
    print(f"  Sweep completed in {time.time() - t0:.1f}s ({len(df)} rows)")

    csv_path = os.path.join(OUTPUT_DIR, "sweep_demand.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    fig_path = os.path.join(OUTPUT_DIR, "03_demand_sweep.png")
    plot_parameter_sweep(df, "budget_mean", fundamental_value=10.0, save_path=fig_path)
    plt.close("all")
    print(f"  Saved: {fig_path}")
    return df


# ---------------------------------------------------------------------------
# §5.4 Information leakage sweep
# ---------------------------------------------------------------------------

def experiment_leakage_sweep():
    _banner("Experiment 5.4: Information Leakage Sweep")
    cfg = SimulationConfig(
        T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, sentiment=0.5,
        budget_mean=1200.0, budget_std=400.0,
    )
    values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4]
    t0 = time.time()
    df = parameter_sweep(cfg, "leakage_rate", values, n_runs_per=15)
    print(f"  Sweep completed in {time.time() - t0:.1f}s ({len(df)} rows)")

    csv_path = os.path.join(OUTPUT_DIR, "sweep_leakage.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    fig_path = os.path.join(OUTPUT_DIR, "04_leakage_sweep.png")
    plot_parameter_sweep(df, "leakage_rate", fundamental_value=10.0, save_path=fig_path)
    plt.close("all")
    print(f"  Saved: {fig_path}")
    return df


# ---------------------------------------------------------------------------
# §5.5 Supply schedule comparison
# ---------------------------------------------------------------------------

def experiment_supply_comparison():
    _banner("Experiment 5.5: Supply Schedule Comparison")
    base = SimulationConfig(
        T=24, total_supply=5000.0,
        n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, sentiment=0.5, leakage_rate=0.05,
        budget_mean=1200.0, budget_std=400.0,
        seed=42,
    )

    schedule_results = {}
    for stype in [SupplyScheduleType.UNIFORM, SupplyScheduleType.FRONT_LOADED,
                  SupplyScheduleType.BACK_LOADED, SupplyScheduleType.BELL_CURVE]:
        cfg = copy(base)
        cfg.supply_type = stype
        result = run_single_simulation(cfg)
        schedule_results[stype.value] = result
        _print_summary(result)

    # Comparison plot
    fig_path = os.path.join(OUTPUT_DIR, "05_supply_comparison.png")
    plot_scenario_comparison(schedule_results, fundamental_value=10.0, save_path=fig_path)
    plt.close("all")
    print(f"  Saved: {fig_path}")

    # Individual dashboards for front/back loaded
    for stype_name in ["front_loaded", "back_loaded"]:
        res = schedule_results[stype_name]
        path = os.path.join(OUTPUT_DIR, f"05_{stype_name}_dashboard.png")
        plot_single_run_dashboard(res, title_suffix=stype_name.replace("_", " ").title(), save_path=path)
        plt.close("all")
        print(f"  Saved: {path}")

    return schedule_results


# ---------------------------------------------------------------------------
# §5.6 Agent composition sweep
# ---------------------------------------------------------------------------

def experiment_composition_sweep():
    _banner("Experiment 5.6: Agent Composition Sweep (n_informed)")
    cfg = SimulationConfig(
        T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
        n_early=12, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
        base_valuation=10.0, sentiment=0.5, leakage_rate=0.05,
        budget_mean=1200.0, budget_std=400.0,
    )
    values = [0, 3, 6, 10, 15, 20, 30]
    t0 = time.time()
    df = parameter_sweep(cfg, "n_informed", values, n_runs_per=15)
    print(f"  Sweep completed in {time.time() - t0:.1f}s ({len(df)} rows)")

    csv_path = os.path.join(OUTPUT_DIR, "sweep_composition.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    fig_path = os.path.join(OUTPUT_DIR, "06_informed_sweep.png")
    plot_parameter_sweep(df, "n_informed", fundamental_value=10.0, save_path=fig_path)
    plt.close("all")
    print(f"  Saved: {fig_path}")
    return df


# ---------------------------------------------------------------------------
# §5.7 Extreme scenarios
# ---------------------------------------------------------------------------

def experiment_extreme_scenarios():
    _banner("Experiment 5.7: Extreme Scenarios")

    scenarios = {
        "Euphoria": SimulationConfig(
            T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
            n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
            base_valuation=10.0,
            sentiment=0.9, budget_mean=2000.0, leakage_rate=0.01, sentiment_vol=0.02,
            seed=42,
        ),
        "Panic": SimulationConfig(
            T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
            n_early=12, n_informed=10, n_momentum=15, n_option=6, n_noise=25, n_whale=3,
            base_valuation=10.0,
            sentiment=0.15, budget_mean=600.0, leakage_rate=0.2,
            sentiment_vol=0.12, shock_probability=0.15,
            seed=42,
        ),
        "Whale-dominated": SimulationConfig(
            T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
            n_early=3, n_informed=5, n_momentum=3, n_option=3, n_noise=5, n_whale=8,
            base_valuation=10.0, sentiment=0.5,
            whale_budget_multiplier=20.0,
            seed=42,
        ),
        "Retail-heavy": SimulationConfig(
            T=24, total_supply=5000.0, supply_type=SupplyScheduleType.UNIFORM,
            n_early=5, n_informed=2, n_momentum=5, n_option=1, n_noise=60, n_whale=0,
            base_valuation=10.0, sentiment=0.5,
            seed=42,
        ),
    }

    results = {}
    for name, cfg in scenarios.items():
        print(f"\n  --- {name} ---")
        result = run_single_simulation(cfg)
        results[name] = result
        _print_summary(result)

    fig_path = os.path.join(OUTPUT_DIR, "07_extreme_scenarios.png")
    plot_scenario_comparison(results, fundamental_value=10.0,
                            title_suffix="Extreme Scenarios", save_path=fig_path)
    plt.close("all")
    print(f"  Saved: {fig_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    _ensure_output_dir()
    print("\n" + "#" * 60)
    print("  CCA Agent-Based Model — Full Experiment Suite")
    print("#" * 60)

    experiment_baseline()
    experiment_sentiment_sweep()
    experiment_demand_sweep()
    experiment_leakage_sweep()
    experiment_supply_comparison()
    experiment_composition_sweep()
    experiment_extreme_scenarios()

    elapsed = time.time() - t_start
    _banner(f"All experiments complete in {elapsed:.1f}s")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
