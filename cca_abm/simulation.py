"""
CCA Simulation Orchestration
=============================
Information model, single-run procedure, Monte Carlo runner, and parameter sweep.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .engine import (
    AuctionResult,
    Bid,
    BidSurplusResult,
    SupplyScheduleType,
    build_supply_schedule,
    compute_surplus,
    run_auction,
)
from .agents import (
    BaseAgent,
    MarketState,
    WhaleAgent,
    create_agent_population,
)


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """All parameters for a CCA simulation in a single flat config."""
    # Time & supply
    T: int = 20
    total_supply: float = 5000.0
    supply_type: SupplyScheduleType = SupplyScheduleType.UNIFORM
    custom_supply_weights: Optional[np.ndarray] = None
    price_bounds: Tuple[float, float] = (0.1, 50.0)

    # Agent counts
    n_early: int = 10
    n_informed: int = 8
    n_momentum: int = 12
    n_option: int = 5
    n_noise: int = 20
    n_whale: int = 2

    # Market
    base_valuation: float = 10.0
    valuation_dispersion: float = 0.2
    budget_mean: float = 1200.0
    budget_std: float = 400.0
    whale_budget_multiplier: float = 10.0

    # Information & sentiment
    sentiment: float = 0.5
    leakage_rate: float = 0.05
    sentiment_vol: float = 0.05
    sentiment_mean_reversion: float = 0.9
    shock_probability: float = 0.05
    shock_magnitude: float = 0.3
    fundamental_value: Optional[float] = None  # defaults to base_valuation

    # Simulation
    seed: int = 42
    n_runs: int = 1

    def __post_init__(self):
        if self.fundamental_value is None:
            self.fundamental_value = self.base_valuation


# ---------------------------------------------------------------------------
# Information & sentiment model
# ---------------------------------------------------------------------------

def generate_public_signals(
    T: int,
    fundamental_value: float,
    leakage_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate public signal process (§3.1).

    s_k = V_true + eps_k,  eps_k ~ N(0, (0.5 * V * exp(-lambda*k))^2)
    """
    signals = np.zeros(T)
    for k in range(T):
        noise_std = 0.5 * fundamental_value * math.exp(-leakage_rate * k)
        signals[k] = fundamental_value + rng.normal(0.0, noise_std)
    return signals


def generate_sentiment_path(
    T: int,
    initial_sentiment: float,
    mean_reversion: float = 0.9,
    long_run: Optional[float] = None,
    vol: float = 0.05,
    shock_prob: float = 0.05,
    shock_mag: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate sentiment dynamics (§3.2).

    s_{k+1} = alpha * s_k + (1-alpha) * s_bar + eps_k + J_k
    """
    if rng is None:
        rng = np.random.default_rng()
    if long_run is None:
        long_run = initial_sentiment

    path = np.zeros(T)
    path[0] = initial_sentiment
    for k in range(T - 1):
        eps = rng.normal(0.0, vol)
        jump = 0.0
        if rng.random() < shock_prob:
            jump = shock_mag * rng.choice([-1.0, 1.0])
        s_next = mean_reversion * path[k] + (1.0 - mean_reversion) * long_run + eps + jump
        path[k + 1] = np.clip(s_next, 0.0, 1.0)
    return path


def compute_realized_volatility(prices: np.ndarray, k: int) -> float:
    """Realized log-price volatility from prices observed so far."""
    valid = prices[:k]
    valid = valid[valid > 0]
    if len(valid) < 2:
        return 0.2  # default
    log_returns = np.diff(np.log(valid))
    return max(float(np.std(log_returns)), 0.01)


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Complete result from a single simulation run."""
    config: SimulationConfig
    clearing_prices: np.ndarray
    supply_schedule: np.ndarray
    public_signals: np.ndarray
    sentiment_path: np.ndarray
    bids: List[Bid]
    agents: List[BaseAgent]
    auction_result: AuctionResult
    surplus_results: List[BidSurplusResult]
    fundamental_value: float


def run_single_simulation(config: SimulationConfig) -> SimulationResult:
    """Execute one complete CCA simulation (§4.1)."""
    rng = np.random.default_rng(config.seed)
    T = config.T
    fundamental_value = config.fundamental_value

    # Step 1: Build supply schedule
    supply_schedule = build_supply_schedule(
        T, config.total_supply, config.supply_type, config.custom_supply_weights
    )

    # Step 2: Generate information model
    public_signals = generate_public_signals(T, fundamental_value, config.leakage_rate, rng)
    sentiment_path = generate_sentiment_path(
        T, config.sentiment, config.sentiment_mean_reversion,
        vol=config.sentiment_vol, shock_prob=config.shock_probability,
        shock_mag=config.shock_magnitude, rng=rng,
    )

    # Step 3: Create agent population
    agents = create_agent_population(
        n_early=config.n_early,
        n_informed=config.n_informed,
        n_momentum=config.n_momentum,
        n_option=config.n_option,
        n_noise=config.n_noise,
        n_whale=config.n_whale,
        base_valuation=config.base_valuation,
        valuation_dispersion=config.valuation_dispersion,
        budget_mean=config.budget_mean,
        budget_std=config.budget_std,
        whale_budget_multiplier=config.whale_budget_multiplier,
        sentiment=config.sentiment,
        seed=rng.integers(0, 2**31),
    )

    # Step 4: Initialize
    all_bids: List[Bid] = []
    clearing_prices_so_far = np.zeros(T)
    entered_agents: set = set()
    bid_counter = 0
    price_min, price_max = config.price_bounds

    # Step 5: Period-by-period loop
    for k in range(T):
        # Build cumulative supply
        cum_supply = np.cumsum(supply_schedule)
        remaining_supply = supply_schedule[k:].sum()
        vol = compute_realized_volatility(clearing_prices_so_far, k)

        state = MarketState(
            period=k,
            clearing_prices=clearing_prices_so_far.copy(),
            supply_released=cum_supply,
            remaining_supply=remaining_supply,
            total_periods=T,
            supply_schedule=supply_schedule,
            public_signal=public_signals[k],
            sentiment_index=sentiment_path[k],
            volatility=vol,
        )

        # Agent decisions
        for agent in agents:
            agent.update_beliefs(k, state)

            if isinstance(agent, WhaleAgent):
                # Whales can submit multiple bids (one per tranche)
                if agent.decide_entry(k, state):
                    bid = Bid(
                        bid_id=bid_counter,
                        agent_id=agent.agent_id,
                        agent_type=agent.agent_type,
                        budget=agent.get_split_budget(),
                        cap=agent.decide_cap(k, state),
                        entry_period=k,
                        valuation=agent.current_valuation,
                    )
                    all_bids.append(bid)
                    bid_counter += 1
            else:
                if agent.agent_id not in entered_agents:
                    if agent.decide_entry(k, state):
                        bid = Bid(
                            bid_id=bid_counter,
                            agent_id=agent.agent_id,
                            agent_type=agent.agent_type,
                            budget=agent.budget,
                            cap=agent.decide_cap(k, state),
                            entry_period=k,
                            valuation=agent.current_valuation,
                        )
                        all_bids.append(bid)
                        bid_counter += 1
                        entered_agents.add(agent.agent_id)
                        agent.has_entered = True
                        agent.entry_period = k

        # Run auction on all bids so far to get clearing price for period k
        if all_bids:
            interim_result = run_auction(all_bids, supply_schedule, price_min=price_min, price_max=price_max)
            clearing_prices_so_far[k] = interim_result.clearing_prices[k]
        else:
            clearing_prices_so_far[k] = price_min

    # Step 6: Final full auction with all bids
    if all_bids:
        final_result = run_auction(all_bids, supply_schedule, price_min=price_min, price_max=price_max)
    else:
        # Degenerate case: no bids at all
        final_result = run_auction([], supply_schedule, price_min=price_min, price_max=price_max)

    # Step 7: Compute surplus
    surplus_results = compute_surplus(all_bids, final_result)

    return SimulationResult(
        config=config,
        clearing_prices=final_result.clearing_prices,
        supply_schedule=supply_schedule,
        public_signals=public_signals,
        sentiment_path=sentiment_path,
        bids=all_bids,
        agents=agents,
        auction_result=final_result,
        surplus_results=surplus_results,
        fundamental_value=fundamental_value,
    )


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(config: SimulationConfig) -> List[SimulationResult]:
    """Run config.n_runs simulations with incrementing seeds."""
    results = []
    for i in range(config.n_runs):
        cfg = copy.copy(config)
        cfg.seed = config.seed + i
        results.append(run_single_simulation(cfg))
    return results


# ---------------------------------------------------------------------------
# Result summarization helpers
# ---------------------------------------------------------------------------

AGENT_TYPES = [
    "EarlyBeliever", "InformedTrader", "MomentumTrader",
    "OptionValueOptimizer", "NoiseTrader", "WhaleAgent",
]


def summarize_single_run(result: SimulationResult) -> Dict[str, Any]:
    """Extract summary statistics from a single simulation run."""
    prices = result.clearing_prices
    valid_prices = prices[prices > 0]

    summary: Dict[str, Any] = {
        "mean_clearing_price": float(valid_prices.mean()) if len(valid_prices) > 0 else 0.0,
        "final_clearing_price": float(prices[-1]) if prices[-1] > 0 else 0.0,
        "price_volatility": float(valid_prices.std()) if len(valid_prices) > 1 else 0.0,
        "n_bids": len(result.bids),
        "total_surplus": sum(s.total_surplus for s in result.surplus_results),
        "fundamental_value": result.fundamental_value,
    }

    # Price drift: (P_T / P_1) - 1
    if len(valid_prices) >= 2:
        summary["price_drift"] = float(valid_prices[-1] / valid_prices[0] - 1.0)
    else:
        summary["price_drift"] = 0.0

    # Per-type metrics
    for atype in AGENT_TYPES:
        type_surpluses = [s for s in result.surplus_results if s.agent_type == atype]
        total = sum(s.total_surplus for s in type_surpluses)
        count = len(type_surpluses)
        summary[f"surplus_{atype}"] = total
        summary[f"count_{atype}"] = count
        summary[f"avg_surplus_{atype}"] = total / count if count > 0 else 0.0

    return summary


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(
    base_config: SimulationConfig,
    param_name: str,
    param_values: Sequence,
    n_runs_per: int = 10,
) -> pd.DataFrame:
    """Sweep a parameter and collect results into a DataFrame."""
    rows = []

    for val in param_values:
        cfg = copy.copy(base_config)
        setattr(cfg, param_name, val)
        cfg.n_runs = n_runs_per

        results = run_monte_carlo(cfg)
        for run_id, res in enumerate(results):
            row = summarize_single_run(res)
            row["param_name"] = param_name
            row["param_value"] = val
            row["run_id"] = run_id
            rows.append(row)

    return pd.DataFrame(rows)
