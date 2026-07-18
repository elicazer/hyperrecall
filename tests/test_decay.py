"""Tests for forgetting curves and Hebbian reinforcement."""

from __future__ import annotations

import time

from hyperrecall import Mesh, Node
from hyperrecall.decay import (
    SECONDS_PER_DAY,
    exponential_decay,
    half_life_to_rate,
    linear_decay,
    power_law_decay,
    reinforce,
)


def test_exponential_decay_decreases_over_time():
    fresh = exponential_decay(1.0, 0.0, 0.1)
    week = exponential_decay(1.0, 7 * SECONDS_PER_DAY, 0.1)
    assert fresh == 1.0
    assert week < fresh
    assert 0.0 < week < 1.0


def test_zero_rate_never_forgets():
    assert exponential_decay(1.0, 1000 * SECONDS_PER_DAY, 0.0) == 1.0


def test_half_life_matches_expected():
    rate = half_life_to_rate(10.0)
    val = exponential_decay(1.0, 10 * SECONDS_PER_DAY, rate)
    assert abs(val - 0.5) < 1e-6


def test_power_law_and_linear_curves_decrease():
    assert power_law_decay(1.0, 30 * SECONDS_PER_DAY, 0.5) < 1.0
    assert linear_decay(1.0, 30 * SECONDS_PER_DAY, 0.01) < 1.0
    assert linear_decay(1.0, 10_000 * SECONDS_PER_DAY, 1.0) == 0.0  # floored at zero


def test_reinforcement_boosts_but_respects_ceiling():
    boosted = reinforce(1.0, amount=0.5)
    assert boosted > 1.0
    # Repeated reinforcement approaches but never exceeds the ceiling.
    v = 1.0
    for _ in range(100):
        v = reinforce(v, amount=0.5, ceiling=4.0)
    assert v <= 4.0


def test_fresh_vs_decayed_activation_in_store():
    """A node whose activation timestamp is old should read lower than a fresh one."""
    mesh = Mesh(":memory:")
    fresh = mesh.add_node(Node(text="fresh memory", decay_rate=0.5, activation=1.0))
    old = mesh.add_node(Node(text="old memory", decay_rate=0.5, activation=1.0))

    # Backdate the old node's activation by 30 days.
    thirty_days_ago = time.time() - 30 * SECONDS_PER_DAY
    mesh.store.set_activation(old.id, base=1.0, updated_at=thirty_days_ago)

    fresh_act = mesh.store.live_activation(fresh.id)
    old_act = mesh.store.live_activation(old.id)
    assert fresh_act > old_act
    assert old_act < 1.0
    mesh.close()


def test_access_reinforces_activation():
    mesh = Mesh(":memory:")
    n = mesh.add_node(Node(text="accessed memory", decay_rate=0.0, activation=1.0))
    before = mesh.store.live_activation(n.id)
    mesh.store.reinforce_node(n.id, amount=0.5)
    after = mesh.store.live_activation(n.id)
    assert after > before
    assert mesh.store.access_count(n.id) == 1
    mesh.close()
