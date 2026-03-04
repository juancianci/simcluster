"""
CCA Mechanism Engine
====================
Supply schedules, spreading rule, uniform-price clearing via bisection,
and surplus computation for the Continuous Clearing Auction.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Supply schedule
# ---------------------------------------------------------------------------

class SupplyScheduleType(enum.Enum):
    UNIFORM = "uniform"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    BELL_CURVE = "bell_curve"
    CUSTOM = "custom"


def build_supply_schedule(
    T: int,
    total_supply: float,
    schedule_type: SupplyScheduleType = SupplyScheduleType.UNIFORM,
    custom_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an array q of length T with q.sum() == total_supply."""
    if schedule_type == SupplyScheduleType.UNIFORM:
        weights = np.ones(T)
    elif schedule_type == SupplyScheduleType.FRONT_LOADED:
        weights = np.linspace(2.0, 0.2, T)
    elif schedule_type == SupplyScheduleType.BACK_LOADED:
        weights = np.linspace(0.2, 2.0, T)
    elif schedule_type == SupplyScheduleType.BELL_CURVE:
        x = np.linspace(-3.0, 3.0, T)
        weights = np.exp(-0.5 * x ** 2)
    elif schedule_type == SupplyScheduleType.CUSTOM:
        if custom_weights is None or len(custom_weights) != T:
            raise ValueError("custom_weights must be an array of length T")
        weights = np.asarray(custom_weights, dtype=float)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    weights = np.maximum(weights, 0.0)
    if weights.sum() == 0:
        raise ValueError("Supply weights sum to zero")
    q = total_supply * weights / weights.sum()
    return q


# ---------------------------------------------------------------------------
# Bid representation
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    """A single bid submitted to the CCA."""
    bid_id: int
    agent_id: int
    agent_type: str
    budget: float
    cap: float                          # max acceptable unit price  (inf = no cap)
    entry_period: int                   # tau
    valuation: float                    # agent's V at time of entry


# ---------------------------------------------------------------------------
# Spreading rules
# ---------------------------------------------------------------------------

def proportional_spreading(
    bid: Bid,
    supply_schedule: np.ndarray,
) -> np.ndarray:
    """Canonical proportional-to-remaining-supply spreading.

    b_k(tau) = B * q_k / sum_{j>=tau} q_j   for k >= tau, else 0.
    """
    T = len(supply_schedule)
    tau = bid.entry_period
    bk = np.zeros(T)
    remaining = supply_schedule[tau:].sum()
    if remaining <= 0:
        return bk
    bk[tau:] = bid.budget * supply_schedule[tau:] / remaining
    return bk


def equal_weight_spreading(
    bid: Bid,
    supply_schedule: np.ndarray,
) -> np.ndarray:
    """Equal-weight spreading: budget divided evenly over remaining periods."""
    T = len(supply_schedule)
    tau = bid.entry_period
    bk = np.zeros(T)
    n_remaining = T - tau
    if n_remaining <= 0:
        return bk
    bk[tau:] = bid.budget / n_remaining
    return bk


def front_weighted_spreading(
    bid: Bid,
    supply_schedule: np.ndarray,
    decay: float = 0.8,
) -> np.ndarray:
    """Front-weighted spreading with geometric decay."""
    T = len(supply_schedule)
    tau = bid.entry_period
    bk = np.zeros(T)
    n_remaining = T - tau
    if n_remaining <= 0:
        return bk
    weights = np.array([decay ** i for i in range(n_remaining)])
    bk[tau:] = bid.budget * weights / weights.sum()
    return bk


# Default spreading rule
DEFAULT_SPREADING = proportional_spreading


# ---------------------------------------------------------------------------
# Clearing engine
# ---------------------------------------------------------------------------

@dataclass
class PeriodResult:
    """Result of clearing a single period."""
    period: int
    clearing_price: float
    supply_released: float
    total_demand_at_clearing: float
    n_active_bids: int
    n_executed_bids: int
    # Per-bid results indexed by bid_id
    bid_spend: dict          # bid_id -> amount spent
    bid_tokens: dict         # bid_id -> tokens received
    bid_executed: dict       # bid_id -> bool


@dataclass
class AuctionResult:
    """Full result from running the auction across all periods."""
    clearing_prices: np.ndarray          # length T
    period_results: List[PeriodResult]
    supply_schedule: np.ndarray
    budget_schedules: dict               # bid_id -> np.ndarray of b_k


def compute_demand(
    price: float,
    budget_fragments: np.ndarray,
    caps: np.ndarray,
) -> float:
    """Total token demand at a given price.

    D(p) = sum_i (b_k^(i) / p) * 1{p <= m_i}
    """
    if price <= 0:
        return np.inf
    active = caps >= price
    return np.sum(budget_fragments[active]) / price


def find_clearing_price(
    supply: float,
    budget_fragments: np.ndarray,
    caps: np.ndarray,
    price_min: float = 0.1,
    price_max: float = 50.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Find uniform clearing price via bisection.

    Finds p such that D(p) = supply, or returns boundary prices on edge cases.
    """
    if len(budget_fragments) == 0 or budget_fragments.sum() <= 0:
        return price_min  # no demand → floor price

    # Check edge cases
    demand_at_min = compute_demand(price_min, budget_fragments, caps)
    if demand_at_min <= supply:
        return price_min  # excess supply even at floor

    demand_at_max = compute_demand(price_max, budget_fragments, caps)
    if demand_at_max >= supply:
        return price_max  # excess demand even at ceiling

    # Bisection
    lo, hi = price_min, price_max
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        d = compute_demand(mid, budget_fragments, caps)
        if abs(d - supply) / max(supply, 1e-12) < tol:
            return mid
        if d > supply:
            lo = mid  # price too low → raise
        else:
            hi = mid  # price too high → lower
    return (lo + hi) / 2.0


def run_auction(
    bids: List[Bid],
    supply_schedule: np.ndarray,
    spreading_fn=None,
    price_min: float = 0.1,
    price_max: float = 50.0,
) -> AuctionResult:
    """Run the full CCA across all T periods.

    Computes budget schedules via the spreading rule, then finds clearing
    prices period by period and determines execution.
    """
    if spreading_fn is None:
        spreading_fn = DEFAULT_SPREADING

    T = len(supply_schedule)

    # Compute budget schedules for every bid
    budget_schedules = {}
    for bid in bids:
        budget_schedules[bid.bid_id] = spreading_fn(bid, supply_schedule)

    clearing_prices = np.zeros(T)
    period_results: List[PeriodResult] = []

    for k in range(T):
        supply_k = supply_schedule[k]

        # Collect budget fragments and caps for period k
        frag_list = []
        cap_list = []
        bid_ids_active = []

        for bid in bids:
            bk = budget_schedules[bid.bid_id][k]
            if bk > 1e-12:
                frag_list.append(bk)
                cap_list.append(bid.cap)
                bid_ids_active.append(bid.bid_id)

        fragments = np.array(frag_list) if frag_list else np.array([])
        caps = np.array(cap_list) if cap_list else np.array([])

        # Find clearing price
        pk = find_clearing_price(supply_k, fragments, caps, price_min, price_max)
        clearing_prices[k] = pk

        # Determine execution per bid
        bid_spend = {}
        bid_tokens = {}
        bid_executed = {}
        n_executed = 0

        for bid in bids:
            bk = budget_schedules[bid.bid_id][k]
            if bk > 1e-12 and pk <= bid.cap:
                bid_spend[bid.bid_id] = bk
                bid_tokens[bid.bid_id] = bk / pk
                bid_executed[bid.bid_id] = True
                n_executed += 1
            else:
                bid_spend[bid.bid_id] = 0.0
                bid_tokens[bid.bid_id] = 0.0
                bid_executed[bid.bid_id] = False

        total_demand = compute_demand(pk, fragments, caps) if len(fragments) > 0 else 0.0

        period_results.append(PeriodResult(
            period=k,
            clearing_price=pk,
            supply_released=supply_k,
            total_demand_at_clearing=total_demand,
            n_active_bids=len(bid_ids_active),
            n_executed_bids=n_executed,
            bid_spend=bid_spend,
            bid_tokens=bid_tokens,
            bid_executed=bid_executed,
        ))

    return AuctionResult(
        clearing_prices=clearing_prices,
        period_results=period_results,
        supply_schedule=supply_schedule,
        budget_schedules=budget_schedules,
    )


# ---------------------------------------------------------------------------
# Surplus computation
# ---------------------------------------------------------------------------

@dataclass
class BidSurplusResult:
    """Surplus decomposition for a single bid."""
    bid_id: int
    agent_id: int
    agent_type: str
    valuation: float
    cap: float
    entry_period: int
    budget: float
    # Computed
    per_period_surplus: np.ndarray       # pi_k for each period
    total_surplus: float
    total_tokens: float
    total_spend: float
    unspent_budget: float
    avg_execution_price: float           # weighted average of prices at which executed
    surplus_per_dollar: float


def compute_surplus(
    bids: List[Bid],
    auction_result: AuctionResult,
) -> List[BidSurplusResult]:
    """Compute surplus for every bid using formula (2) from the paper.

    pi_k(tau, m) = 1{P_k <= m} * (V/P_k - 1) * b_k(tau)
    """
    T = len(auction_result.clearing_prices)
    results = []

    for bid in bids:
        per_period_surplus = np.zeros(T)
        total_tokens = 0.0
        total_spend = 0.0
        weighted_price_sum = 0.0

        for k in range(T):
            pr = auction_result.period_results[k]
            pk = auction_result.clearing_prices[k]
            bk = auction_result.budget_schedules[bid.bid_id][k]

            if bk > 1e-12 and pk <= bid.cap and pk > 0:
                surplus_k = (bid.valuation / pk - 1.0) * bk
                per_period_surplus[k] = surplus_k
                tokens_k = bk / pk
                total_tokens += tokens_k
                total_spend += bk
                weighted_price_sum += bk  # weight for price averaging

        unspent = bid.budget - total_spend
        total_surplus = per_period_surplus.sum()

        if total_spend > 0:
            avg_price = total_spend / total_tokens
            surplus_per_dollar = total_surplus / total_spend
        else:
            avg_price = 0.0
            surplus_per_dollar = 0.0

        results.append(BidSurplusResult(
            bid_id=bid.bid_id,
            agent_id=bid.agent_id,
            agent_type=bid.agent_type,
            valuation=bid.valuation,
            cap=bid.cap,
            entry_period=bid.entry_period,
            budget=bid.budget,
            per_period_surplus=per_period_surplus,
            total_surplus=total_surplus,
            total_tokens=total_tokens,
            total_spend=total_spend,
            unspent_budget=unspent,
            avg_execution_price=avg_price,
            surplus_per_dollar=surplus_per_dollar,
        ))

    return results
