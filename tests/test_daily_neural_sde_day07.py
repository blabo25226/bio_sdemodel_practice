"""Day 7 tests: Oh, Lim & Kim (2024) Stable Neural SDEs, Geometric SDE
(numpy only).

These tests do not train the MLP teachers; they check the
ground-truth GSDE identities, the data generator, the thinning
construction, and the time-dependent simulator that the Day 7
pipeline relies on.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.common import fit_polynomial_field, nrmse
from src.daily_neural_sde.day07_oh_stable_gsde import (
    build_queries, build_thinned_transitions, build_transitions,
    generate_data, simulate, true_drift, true_g, true_gamma)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day07.json").read_text())


def _small_cfg():
    small = dict(CFG)
    small["data"] = dict(CFG["data"], n_trajectories=16)
    return small


def test_true_gsde_at_equilibrium_and_at_origin():
    # At the interior equilibrium (2, 1) the effective drift vanishes
    # and the diffusion levels are (0.30 * 2, 0.25 * 1).
    eq = np.array([[2.0, 1.0]])
    assert np.allclose(true_gamma(eq), 0.0)
    assert np.allclose(true_drift(eq), 0.0)
    assert np.allclose(true_g(eq), [[0.60, 0.25]])
    # At the origin the GSDE structure forces drift and diffusion to
    # vanish exactly (the property a polynomial fit can break).
    origin = np.array([[0.0, 0.0]])
    assert np.allclose(true_drift(origin), 0.0)
    assert np.allclose(true_g(origin), 0.0)


def test_generate_data_shape_positivity():
    data = generate_data(_small_cfg())
    assert data["obs"].shape == (16, 201, 2)
    assert np.isfinite(data["obs"]).all()
    assert (data["obs"] > 0).all()
    assert np.allclose(data["obs"][:, 0, :], data["x0"])


def test_transition_alignment_and_thinning():
    data = generate_data(_small_cfg())
    tr = build_transitions(data["obs"], 0.05)
    n_paths = 16
    assert len(tr["cur"]) == n_paths * 200
    assert np.allclose(tr["cur"][:n_paths], data["obs"][:, 0, :])
    assert np.allclose(tr["nxt"][:n_paths], data["obs"][:, 1, :])
    assert np.allclose(tr["dt"], 0.05)
    thin = build_thinned_transitions(data["obs"], 0.05, 0.5, 123)
    assert len(thin["cur"]) < len(tr["cur"])
    assert np.all(thin["dt"] >= 0.05)
    assert np.allclose(thin["dt"] / 0.05, np.round(thin["dt"] / 0.05))
    assert (thin["t"] >= 0).all()


def test_degree2_sindy_recovers_truth_drift():
    rng = np.random.default_rng(0)
    states = rng.uniform(0.3, 3.5, size=(4000, 2))
    field = fit_polynomial_field(states, true_drift(states), 2, 0.02)
    assert nrmse(true_drift(states), field.predict(states)) < 0.01


def test_simulate_truth_stays_positive_and_classifies_failures():
    rng = np.random.default_rng(0)
    x0 = np.stack([rng.lognormal(np.log(2.0), 0.4, size=8),
                   rng.lognormal(0.0, 0.4, size=8)], axis=1)
    inc = rng.standard_normal((8, 200, 2))
    sim = simulate(lambda tk, x: true_drift(x), lambda tk, x: true_g(x),
                   x0, 0.05, 200, inc, 30.0, 8.0)
    assert sim["failure_rate"] == 0.0
    assert sim["negativity_rate"] == 0.0
    assert np.isfinite(sim["states"][:, 0, :]).all()
    # A tiny failure radius forces outside_radius failures.
    sim2 = simulate(lambda tk, x: true_drift(x), lambda tk, x: true_g(x),
                    x0, 0.05, 200, inc, 0.5, 0.25)
    assert sim2["failure_rate"] > 0.0
    assert set(c for c in sim2["failure_class"] if c) <= {"outside_radius"}
