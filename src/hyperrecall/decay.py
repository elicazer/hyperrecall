"""Forgetting curves and reinforcement.

Human memory does two opposing things: it *forgets* over time and it
*strengthens* on retrieval. HyperRecall models both with small, pluggable
functions so a caller can swap in a different curve without touching storage or
retrieval.

The default is an Ebbinghaus-style exponential forgetting curve:

    retention(t) = exp(-decay_rate * elapsed_days)

so activation at time ``now`` is ``base_activation * retention(now - t0)``.
Reinforcement uses a Hebbian bump with diminishing returns toward a ceiling.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

SECONDS_PER_DAY = 86_400.0


class DecayFn(Protocol):
    """A decay function maps (base activation, elapsed seconds, rate) -> value."""

    def __call__(self, base: float, elapsed_seconds: float, rate: float) -> float: ...


def exponential_decay(base: float, elapsed_seconds: float, rate: float) -> float:
    """Ebbinghaus-style exponential forgetting curve.

    ``rate`` is expressed per day. With ``rate=0`` nothing is forgotten.
    """
    if rate <= 0.0 or elapsed_seconds <= 0.0:
        return base
    days = elapsed_seconds / SECONDS_PER_DAY
    return base * math.exp(-rate * days)


def power_law_decay(base: float, elapsed_seconds: float, rate: float) -> float:
    """Power-law forgetting: retention = (1 + days) ** -rate.

    Fits human recall over long horizons better than pure exponential; offered
    as an alternative pluggable curve.
    """
    if rate <= 0.0 or elapsed_seconds <= 0.0:
        return base
    days = elapsed_seconds / SECONDS_PER_DAY
    return base * (1.0 + days) ** (-rate)


def linear_decay(base: float, elapsed_seconds: float, rate: float) -> float:
    """Simple linear decay, floored at zero. Useful for tests / debugging."""
    days = elapsed_seconds / SECONDS_PER_DAY
    return max(0.0, base - rate * days)


# Registry of named curves so the CLI / config can select one by string.
CURVES: dict[str, DecayFn] = {
    "exponential": exponential_decay,
    "power_law": power_law_decay,
    "linear": linear_decay,
}


def get_curve(name: str) -> DecayFn:
    try:
        return CURVES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown decay curve {name!r}; choose from {sorted(CURVES)}"
        ) from exc


def current_activation(
    base: float,
    elapsed_seconds: float,
    rate: float,
    curve: DecayFn = exponential_decay,
) -> float:
    """Activation *now*, given its value at the last update and elapsed time."""
    return curve(base, elapsed_seconds, rate)


def reinforce(
    activation: float,
    amount: float = 0.5,
    ceiling: float = 4.0,
) -> float:
    """Hebbian reinforcement with diminishing returns.

    Accessing a memory boosts it, but with a soft ceiling so hot nodes don't run
    away. The closer to ``ceiling``, the smaller the bump.
    """
    headroom = max(0.0, ceiling - activation)
    return activation + amount * (headroom / ceiling)


def half_life_to_rate(half_life_days: float) -> float:
    """Convert a desired half-life (days) into an exponential decay rate."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return math.log(2.0) / half_life_days
