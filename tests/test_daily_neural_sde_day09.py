"""Day 9 tests: Yang et al. (2024) Neural McKean-Vlasov Processes, MV-SDE
(numpy only).

These tests do not train the MLP teachers; they check the ground-truth
McKean-Vlasov identities, the ensemble data generator, the
transition/m-mean alignment, the factorized-origin library (the Day 7
structural-constraint lesson: the expanded polynomial must genuinely
vanish at (x, m) = (0, 0)), and the self-consistent-mean simulator's
failure accounting.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day09_yang_mv import (
    FACT_ORIGIN, build_transitions, expand_to_original, fit_field_powers,
    generate_data, simulate_mv, true_drift, true_g)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day09.json").read_text())


def _small_cfg():
    small = dict(CFG)
    small["data"] = dict(CFG["data"], n_ensembles=8, n_particles=16)
    return small


def test_true_mv_identities():
    xm = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    # f = (x - x^3) + 0.9 (m - x) = 0.1 x - x^3 + 0.9 m
    assert np.allclose(true_drift(xm),
                       [[0.0], [-0.9], [-6.9], [0.9]])
    assert np.allclose(true_g(xm), 0.6)


def test_generate_data_shape_and_initial_means():
    data = generate_data(_small_cfg())
    assert data["obs"].shape == (8, 121, 16)
    assert np.isfinite(data["obs"]).all()
    assert np.allclose(data["obs"][:, 0, :], data["x0"])
    assert np.all((data["m0"] >= -1.2) & (data["m0"] <= 1.2))
    # Initial ensemble means track the drawn m0 (up to sampling noise).
    assert np.allclose(data["obs"].mean(axis=2)[:, 0], data["m0"], atol=0.4)


def test_transition_alignment_and_mean_field():
    data = generate_data(_small_cfg())
    tr = build_transitions(data["obs"], 0.05)
    assert len(tr["cur"]) == 8 * 120 * 16
    # First ensemble, first time: particle states and their mean.
    assert np.allclose(tr["cur"][:16, 0], data["obs"][0, 0, :])
    assert np.allclose(tr["cur"][:16, 1], data["obs"][0, 0, :].mean())
    assert np.allclose(tr["nxt"][:16, 0], data["obs"][0, 1, :])
    assert np.allclose(tr["dt"], 0.05)


def test_factorized_origin_fit_vanishes_at_origin():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 1)) * 0.7
    m = rng.standard_normal((3000, 1)) * 0.7
    xm = np.concatenate([x, m], axis=1)
    y = true_drift(xm) + rng.standard_normal((3000, 1)) * 0.01
    # Threshold 0.01 in RMS-normalized units: at threshold 0.05 the small
    # linear coefficient (0.1) is masked by the cubic-dominated field RMS,
    # which is itself the resolution limit the H3c discussion reports.
    field = fit_field_powers(xm, y, FACT_ORIGIN, 0.01, 1e-4, mean0=True)
    expanded = expand_to_original(field)[0]
    assert (0, 0) not in expanded
    assert abs(expanded[(1, 0)] - 0.1) < 0.1
    assert abs(expanded[(0, 1)] - 0.9) < 0.1
    assert abs(expanded[(3, 0)] + 1.0) < 0.15


def test_simulate_mv_constant_and_failure():
    zero = lambda xm: np.zeros((len(xm), 1))
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 8))
    dW = rng.standard_normal((4, 8, 20))
    sim = simulate_mv(zero, zero, x0, 0.05, 20, dW, 30.0, 8.0, 5.0)
    assert np.allclose(sim["states"][:, -1, :], x0)
    assert sim["failure_rate"] == 0.0
    assert sim["mean_excursion_rate"] == 0.0
    big = np.full((4, 8), 40.0)
    sim_big = simulate_mv(zero, zero, big, 0.05, 20, dW, 30.0, 8.0, 5.0)
    assert sim_big["failure_rate"] == 1.0
