"""Day 5 tests: Zhang et al. (2025) END-nSDE circadian benchmark (numpy only).

These tests do not train the MLP teachers; they check the ground-truth
field identities, the data generator, and the full-matrix simulator
that the Day 5 pipeline relies on.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day05_zhang_circadian import (
    L_from_entries, generate_data, sigma_from_L, simulate_matrix,
    true_L, true_L_entries, true_drift, true_sigma_matrix, w2_loss_np)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day05.json").read_text())


def test_true_drift_and_sigma_at_known_state():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    drift = true_drift(x, CFG)
    # (-alpha x - beta y, beta x - alpha y) with alpha=0.19, beta=0.21
    assert np.allclose(drift[0], [-0.21, -0.19])
    assert np.allclose(drift[1], [-0.19, 0.21])
    sigma = true_sigma_matrix(CFG)
    # sigma0^2 [[1 + c^2, 2c], [2c, 1 + c^2]] with sigma0=0.2, c=0.5
    assert np.allclose(sigma, 0.04 * np.array([[1.25, 1.0], [1.0, 1.25]]))
    L = true_L(x, CFG)
    assert np.allclose(sigma_from_L(L)[0], sigma)


def test_L_entries_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    entries = true_L_entries(x, CFG)
    assert np.allclose(L_from_entries(entries), true_L(x, CFG))
    # The truth diffusion matrix is constant in the state.
    assert np.allclose(entries, entries[0])


def test_generate_data_shape_and_initial_condition():
    small = dict(CFG)
    small["data"] = dict(CFG["data"], n_trajectories=64)
    traj = generate_data(small)
    assert traj.shape == (64, 81, 2)
    assert np.isfinite(traj).all()
    assert (traj[:, 0, 0] == 0.0).all() and (traj[:, 0, 1] == 1.0).all()


def test_w2_loss_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(32, 5, 2))
    assert w2_loss_np(traj, traj) == 0.0


def test_simulate_matrix_truth_stays_finite():
    rng = np.random.default_rng(0)
    n, n_steps = 16, 400
    inc = rng.standard_normal((n, n_steps, 2))
    x0 = np.tile(np.array([0.0, 1.0]), (n, 1))
    sim = simulate_matrix(lambda x: true_drift(x, CFG),
                          lambda x: true_L(x, CFG), x0, 0.01, n_steps,
                          inc, failure_radius=5.0, domain_radius=2.0)
    assert sim["failure_rate"] == 0.0
    assert np.isfinite(sim["terminal"]).all()
    assert 0.0 <= sim["domain_excursion_rate"] <= 1.0
