"""Day 4 tests: Gierjatowicz et al. (2020) LSV benchmark (numpy only).

These tests do not train the MLP teacher; they check the ground-truth
field identities, the data generator, and the full-matrix simulator
that the Day 4 pipeline relies on.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day04_gierjatowicz_lsv import (
    L_from_entries, generate_data, regime_of, sigma_from_L, simulate_matrix,
    true_L, true_L_entries, true_drift)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day04.json").read_text())


def test_true_drift_and_sigma_at_known_state():
    x = np.array([[0.0, 0.04], [0.0, 0.0]])
    drift = true_drift(x, CFG)
    # r - 0.5 v, kappa (theta - v)
    assert drift[0, 0] == 0.02 - 0.5 * 0.04
    assert drift[0, 1] == 2.0 * (0.04 - 0.04)
    assert drift[1, 1] == 2.0 * 0.04
    sigma = sigma_from_L(true_L(x, CFG))
    # Sigma = [[v, rho xi v], [rho xi v, xi^2 v]]
    assert np.allclose(sigma[0], [[0.04, -0.7 * 0.3 * 0.04],
                                  [-0.7 * 0.3 * 0.04, 0.09 * 0.04]])
    assert np.allclose(sigma[1], np.zeros((2, 2)))


def test_L_entries_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2)) * np.array([0.3, 0.05]) + np.array([0.0, 0.04])
    entries = true_L_entries(x, CFG)
    assert entries[:, 1].max() == 0.0  # L12 fixed to zero
    assert np.allclose(L_from_entries(entries), true_L(x, CFG))


def test_generate_data_shape_and_regimes():
    small = dict(CFG)
    small["data"] = dict(CFG["data"], n_trajectories=64)
    traj = generate_data(small)
    assert traj.shape == (64, 21, 2)
    assert np.isfinite(traj).all()
    assert (traj[:, 0, 0] == 0.0).all() and (traj[:, 0, 1] == 0.04).all()
    labels = regime_of(traj.reshape(-1, 2))
    assert set(labels) <= {"low", "mid", "high"}
    assert "low" in set(labels) and "high" in set(labels)


def test_simulate_matrix_paired_and_failures():
    rng = np.random.default_rng(1)
    x0 = np.zeros((8, 2))
    x0[:, 1] = 0.04
    inc = rng.standard_normal((8, 10, 2))
    out = simulate_matrix(lambda x: true_drift(x, CFG),
                          lambda x: true_L(x, CFG),
                          x0, 0.01, 10, inc, 20.0)
    assert out["terminal"].shape == (8, 2)
    assert out["failure_rate"] == 0.0
    # Identical inputs reproduce identical terminals (paired determinism).
    out2 = simulate_matrix(lambda x: true_drift(x, CFG),
                           lambda x: true_L(x, CFG),
                           x0, 0.01, 10, inc, 20.0)
    assert np.allclose(out["terminal"], out2["terminal"])
