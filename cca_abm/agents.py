"""
CCA Agent Types
===============
Six agent types with belief updates, entry/cap logic, and a population factory.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .engine import Bid


# ---------------------------------------------------------------------------
# Market state observable
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """Information available to agents at each period."""
    period: int
    clearing_prices: np.ndarray       # P_0 .. P_{k-1} (future = 0)
    supply_released: np.ndarray       # cumulative supply through period k
    remaining_supply: float
    total_periods: int
    supply_schedule: np.ndarray       # full q_k vector
    public_signal: float
    sentiment_index: float
    volatility: float


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for all CCA agent types."""

    def __init__(
        self,
        agent_id: int,
        budget: float,
        base_valuation: float,
        valuation_uncertainty: float,
        sentiment_sensitivity: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.agent_id = agent_id
        self.budget = budget
        self.base_valuation = base_valuation
        self.valuation_uncertainty = valuation_uncertainty
        self.sentiment_sensitivity = sentiment_sensitivity
        self.rng = rng or np.random.default_rng()

        # Belief state
        self.current_valuation = base_valuation
        self.belief_mean = base_valuation
        self.belief_var = (valuation_uncertainty * base_valuation) ** 2

        # Entry tracking
        self.has_entered = False
        self.entry_period: Optional[int] = None

    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

    # --- Default belief update (§2.3) ---

    def update_beliefs(self, k: int, state: MarketState) -> None:
        """Two-stage conjugate belief update."""
        # Guard against zero variance
        if self.belief_var < 1e-16:
            self.belief_var = 1e-16
        sigma_v_sq = max((self.valuation_uncertainty * self.base_valuation) ** 2, 1e-16)

        # Stage 1: Public signal update
        lam_prior = 1.0 / self.belief_var
        lam_sig = self.sentiment_sensitivity / sigma_v_sq
        if lam_sig > 0 and state.public_signal != 0:
            mu_post = (lam_prior * self.belief_mean + lam_sig * state.public_signal) / (lam_prior + lam_sig)
            var_post = 1.0 / (lam_prior + lam_sig)
            self.belief_mean = mu_post
            self.belief_var = var_post

        # Stage 2: Price signal update
        if k > 0 and state.clearing_prices[k - 1] > 0:
            price_signal = state.clearing_prices[k - 1] * 1.2
            vol = max(state.volatility, 0.01)
            lam_price = 0.5 / (vol ** 2)
            lam_prior2 = 1.0 / max(self.belief_var, 1e-16)
            mu_post2 = (lam_prior2 * self.belief_mean + lam_price * price_signal) / (lam_prior2 + lam_price)
            var_post2 = 1.0 / (lam_prior2 + lam_price)
            self.belief_mean = mu_post2
            self.belief_var = var_post2

        self.current_valuation = self.belief_mean

    @abstractmethod
    def decide_entry(self, k: int, state: MarketState) -> bool:
        """Return True if the agent wants to enter at period k."""
        ...

    @abstractmethod
    def decide_cap(self, k: int, state: MarketState) -> float:
        """Return the max acceptable unit price."""
        ...


# ---------------------------------------------------------------------------
# 2.4.1 EarlyBeliever
# ---------------------------------------------------------------------------

class EarlyBeliever(BaseAgent):
    """High-conviction agent entering very early (Proposition 1 regime)."""

    def __init__(self, *args, conviction_multiplier: float = 1.5,
                 max_entry_delay: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.conviction_multiplier = conviction_multiplier
        self.max_entry_delay = max_entry_delay

    def decide_entry(self, k: int, state: MarketState) -> bool:
        if self.has_entered or k > self.max_entry_delay:
            return False
        if k == self.max_entry_delay:
            return True
        prob = 1.0 - (0.3 - 0.2 * state.sentiment_index)
        return self.rng.random() < prob

    def decide_cap(self, k: int, state: MarketState) -> float:
        return self.current_valuation * self.conviction_multiplier


# ---------------------------------------------------------------------------
# 2.4.2 InformedTrader
# ---------------------------------------------------------------------------

class InformedTrader(BaseAgent):
    """Receives private signals, conditions entry on signal quality (Prop. 2)."""

    def __init__(self, *args, signal_precision: float = 2.0,
                 entry_threshold: float = 0.10, signal_delay: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_precision = signal_precision
        self.entry_threshold = entry_threshold
        self.signal_delay = signal_delay

    def update_beliefs(self, k: int, state: MarketState) -> None:
        # Default public + price update
        super().update_beliefs(k, state)

        # Private signal arrival
        if k >= self.signal_delay:
            s_priv = self.base_valuation + self.rng.normal(0.0, 1.0 / math.sqrt(self.signal_precision))
            lam_prior = 1.0 / max(self.belief_var, 1e-16)
            lam_priv = self.signal_precision
            self.belief_mean = (lam_prior * self.belief_mean + lam_priv * s_priv) / (lam_prior + lam_priv)
            self.belief_var = 1.0 / (lam_prior + lam_priv)
            self.current_valuation = self.belief_mean

    def decide_entry(self, k: int, state: MarketState) -> bool:
        if self.has_entered:
            return False
        if k < self.signal_delay:
            return False

        # Estimate expected surplus per dollar
        last_price = state.clearing_prices[k - 1] if k > 0 and state.clearing_prices[k - 1] > 0 else 0.8 * self.base_valuation
        expected_surplus = self.belief_mean / last_price - 1.0

        remaining_frac = max((state.total_periods - k) / state.total_periods, 0.01)
        threshold = self.entry_threshold * remaining_frac
        return expected_surplus > threshold

    def decide_cap(self, k: int, state: MarketState) -> float:
        return self.current_valuation * 1.1


# ---------------------------------------------------------------------------
# 2.4.3 MomentumTrader
# ---------------------------------------------------------------------------

class MomentumTrader(BaseAgent):
    """FOMO-driven agent that enters on trend confirmation."""

    def __init__(self, *args, lookback: int = 3, momentum_threshold: float = 0.03,
                 fomo_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookback = lookback
        self.momentum_threshold = momentum_threshold
        self.fomo_factor = fomo_factor

    def update_beliefs(self, k: int, state: MarketState) -> None:
        super().update_beliefs(k, state)
        # Strong price anchoring
        if k > 0 and state.clearing_prices[k - 1] > 0:
            self.current_valuation = 0.6 * self.current_valuation + 0.4 * state.clearing_prices[k - 1] * 1.1

    def _compute_momentum(self, k: int, state: MarketState) -> float:
        # Gather last `lookback` valid prices
        valid = []
        for j in range(max(0, k - self.lookback), k):
            if state.clearing_prices[j] > 0:
                valid.append(state.clearing_prices[j])
        if len(valid) < 2:
            return 0.0
        log_returns = [math.log(valid[i + 1] / valid[i]) for i in range(len(valid) - 1)]
        return sum(log_returns) / len(log_returns)

    def decide_entry(self, k: int, state: MarketState) -> bool:
        if self.has_entered:
            return False

        T = state.total_periods
        momentum = self._compute_momentum(k, state)

        # Momentum trigger
        if momentum > self.momentum_threshold:
            prob = min(1.0, self.fomo_factor * momentum / self.momentum_threshold)
            if self.rng.random() < prob:
                return True

        # Late-stage panic
        if k > 0.8 * T:
            urgency = (k - 0.8 * T) / (0.2 * T)
            if self.rng.random() < min(urgency, 0.5):
                return True

        return False

    def decide_cap(self, k: int, state: MarketState) -> float:
        recent_price = 0.0
        if k > 0:
            for j in range(k - 1, -1, -1):
                if state.clearing_prices[j] > 0:
                    recent_price = state.clearing_prices[j]
                    break
        return max(recent_price * 1.2, self.current_valuation)


# ---------------------------------------------------------------------------
# 2.4.4 OptionValueOptimizer
# ---------------------------------------------------------------------------

class OptionValueOptimizer(BaseAgent):
    """Explicitly models exposure premium vs option value tradeoff (Prop. 2)."""

    def __init__(self, *args, patience: float = 0.6,
                 option_value_decay: float = 0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.option_value_decay = option_value_decay

    def decide_entry(self, k: int, state: MarketState) -> bool:
        if self.has_entered:
            return False

        T = state.total_periods
        remaining = T - k

        # Option value: OV(k) = patience * sqrt(belief_var) * sqrt(T - k)
        ov = self.patience * math.sqrt(max(self.belief_var, 0.0)) * math.sqrt(max(remaining, 0))
        ov *= self.option_value_decay ** k

        # Exposure premium
        last_price = 0.0
        if k > 0:
            for j in range(k - 1, -1, -1):
                if state.clearing_prices[j] > 0:
                    last_price = state.clearing_prices[j]
                    break
        if last_price <= 0:
            last_price = 0.8 * self.base_valuation

        surplus_per_dollar = max(self.belief_mean / last_price - 1.0, 0.0)
        remaining_supply = state.supply_schedule[k:].sum()
        if remaining_supply > 0:
            ep = surplus_per_dollar * (state.supply_schedule[k] / remaining_supply) * self.budget
        else:
            ep = 0.0

        if ep > ov:
            return True

        # Force entry if < 15% periods remain and positive expected surplus
        if remaining < 0.15 * T and surplus_per_dollar > 0:
            return True

        return False

    def decide_cap(self, k: int, state: MarketState) -> float:
        return self.current_valuation * 1.05


# ---------------------------------------------------------------------------
# 2.4.5 NoiseTrader
# ---------------------------------------------------------------------------

class NoiseTrader(BaseAgent):
    """Behavioral retail participant with random entry and noisy cap."""

    def __init__(self, *args, entry_probability: float = 0.15,
                 cap_noise_std: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_probability = entry_probability
        self.cap_noise_std = cap_noise_std

    def update_beliefs(self, k: int, state: MarketState) -> None:
        super().update_beliefs(k, state)
        # Sticky beliefs
        if k > 0 and state.clearing_prices[k - 1] > 0:
            self.current_valuation = 0.9 * self.current_valuation + 0.1 * state.clearing_prices[k - 1]

    def decide_entry(self, k: int, state: MarketState) -> bool:
        if self.has_entered:
            return False
        prob = self.entry_probability * (0.5 + state.sentiment_index)
        return self.rng.random() < prob

    def decide_cap(self, k: int, state: MarketState) -> float:
        eps = self.rng.normal(0.0, self.cap_noise_std)
        return max(self.current_valuation * math.exp(eps), 0.01)


# ---------------------------------------------------------------------------
# 2.4.6 WhaleAgent
# ---------------------------------------------------------------------------

class WhaleAgent(BaseAgent):
    """Large-budget agent that splits entry across multiple periods."""

    def __init__(self, *args, split_periods: int = 3,
                 impact_awareness: float = 0.7, entry_start: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_periods = split_periods
        self.impact_awareness = impact_awareness
        self.entry_start = entry_start

        # Track tranches
        self._tranches_entered = 0
        self._target_periods: List[int] = []
        self._computed_targets = False

    def _compute_target_periods(self, T: int) -> None:
        if self._computed_targets:
            return
        stride = max((T - self.entry_start) / self.split_periods, 1)
        self._target_periods = [
            int(round(self.entry_start + i * stride))
            for i in range(self.split_periods)
        ]
        # Clip to valid range
        self._target_periods = [min(p, T - 1) for p in self._target_periods]
        self._computed_targets = True

    def get_split_budget(self) -> float:
        return self.budget / self.split_periods

    def decide_entry(self, k: int, state: MarketState) -> bool:
        """Whales can enter multiple times (once per tranche)."""
        self._compute_target_periods(state.total_periods)

        if self._tranches_entered >= self.split_periods:
            return False
        if k not in self._target_periods:
            return False
        # Check if already entered this period (handle duplicate target periods)
        # Only enter once per target period call
        if k < self.entry_start:
            return False

        # Enter if price is favorable
        last_price = 0.0
        if k > 0:
            for j in range(k - 1, -1, -1):
                if state.clearing_prices[j] > 0:
                    last_price = state.clearing_prices[j]
                    break
        if last_price > 0 and last_price > 1.1 * self.current_valuation:
            return False

        self._tranches_entered += 1
        return True

    def decide_cap(self, k: int, state: MarketState) -> float:
        return self.current_valuation * (1.0 - self.impact_awareness * 0.1)


# ---------------------------------------------------------------------------
# Population factory
# ---------------------------------------------------------------------------

def create_agent_population(
    n_early: int = 10,
    n_informed: int = 8,
    n_momentum: int = 12,
    n_option: int = 5,
    n_noise: int = 20,
    n_whale: int = 2,
    base_valuation: float = 10.0,
    valuation_dispersion: float = 0.2,
    budget_mean: float = 1000.0,
    budget_std: float = 300.0,
    whale_budget_multiplier: float = 10.0,
    sentiment: float = 0.5,
    seed: int = 42,
) -> List[BaseAgent]:
    """Create a heterogeneous agent population with sampled parameters."""
    rng = np.random.default_rng(seed)
    agents: List[BaseAgent] = []
    agent_id = 0

    # Sentiment shift: positive sentiment shifts valuations up
    sentiment_shift = (sentiment - 0.5) * 2.0 * valuation_dispersion * base_valuation
    sigma_v = valuation_dispersion * base_valuation

    def _sample_valuation() -> float:
        v = base_valuation + sentiment_shift + rng.normal(0.0, sigma_v)
        return max(v, 0.1)

    def _sample_budget() -> float:
        return max(rng.normal(budget_mean, budget_std), 50.0)

    def _sample_whale_budget() -> float:
        return max(budget_mean * whale_budget_multiplier * (1.0 + rng.normal(0.0, 0.2)), 50.0)

    # --- EarlyBelievers ---
    for _ in range(n_early):
        agents.append(EarlyBeliever(
            agent_id=agent_id,
            budget=_sample_budget(),
            base_valuation=_sample_valuation(),
            valuation_uncertainty=valuation_dispersion,
            sentiment_sensitivity=rng.uniform(0.5, 1.5),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            conviction_multiplier=rng.uniform(1.3, 1.8),
            max_entry_delay=int(rng.choice([0, 1, 2])),
        ))
        agent_id += 1

    # --- InformedTraders ---
    for _ in range(n_informed):
        agents.append(InformedTrader(
            agent_id=agent_id,
            budget=_sample_budget(),
            base_valuation=_sample_valuation(),
            valuation_uncertainty=valuation_dispersion,
            sentiment_sensitivity=rng.uniform(0.5, 1.5),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            signal_precision=rng.uniform(1.0, 4.0),
            entry_threshold=rng.uniform(0.05, 0.20),
            signal_delay=int(rng.choice([1, 2, 3, 4])),
        ))
        agent_id += 1

    # --- MomentumTraders ---
    for _ in range(n_momentum):
        agents.append(MomentumTrader(
            agent_id=agent_id,
            budget=_sample_budget(),
            base_valuation=_sample_valuation(),
            valuation_uncertainty=valuation_dispersion,
            sentiment_sensitivity=rng.uniform(0.5, 1.5),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            lookback=int(rng.choice([2, 3, 4, 5])),
            momentum_threshold=rng.uniform(0.01, 0.05),
            fomo_factor=rng.uniform(0.5, 2.0),
        ))
        agent_id += 1

    # --- OptionValueOptimizers ---
    for _ in range(n_option):
        agents.append(OptionValueOptimizer(
            agent_id=agent_id,
            budget=_sample_budget(),
            base_valuation=_sample_valuation(),
            valuation_uncertainty=valuation_dispersion,
            sentiment_sensitivity=rng.uniform(0.5, 1.5),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            patience=rng.uniform(0.3, 1.0),
            option_value_decay=rng.uniform(0.75, 0.95),
        ))
        agent_id += 1

    # --- NoiseTraders ---
    for _ in range(n_noise):
        agents.append(NoiseTrader(
            agent_id=agent_id,
            budget=_sample_budget(),
            base_valuation=_sample_valuation(),
            valuation_uncertainty=valuation_dispersion,
            sentiment_sensitivity=rng.uniform(0.3, 1.0),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            entry_probability=rng.uniform(0.08, 0.28),
            cap_noise_std=rng.uniform(0.1, 0.5),
        ))
        agent_id += 1

    # --- WhaleAgents ---
    for _ in range(n_whale):
        agents.append(WhaleAgent(
            agent_id=agent_id,
            budget=_sample_whale_budget(),
            base_valuation=_sample_valuation(),
            valuation_uncertainty=valuation_dispersion,
            sentiment_sensitivity=rng.uniform(0.5, 1.5),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            split_periods=int(rng.choice([2, 3, 4])),
            impact_awareness=rng.uniform(0.5, 1.0),
            entry_start=int(rng.choice([1, 2, 3])),
        ))
        agent_id += 1

    return agents
