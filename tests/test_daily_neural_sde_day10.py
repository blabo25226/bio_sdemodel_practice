"""Day 10 tests: Song et al. (2021) score-based reverse-time SDE (numpy only).

These tests do not train the MLP teacher; they check the analytic VP-SDE
identities, the perturbed score, the reverse-drift formula, the paired
reverse simulator's failure/excursion accounting, and the 1D Wasserstein
helper.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day10_song_score import (
    alpha, beta, forward_sample, pert_var, simulate_reverse, true_reverse_drift,
    true_reverse_g, true_score, w1_1d)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day10.json").read_text())


def test_vp_kernel_identities():
    assert np.isclose(alpha(np.array([0.0]))[0], 1.0)
    assert np.isclose(pert_var(np.array([0.0]))[0], 0.3 ** 2)
    # beta schedule endpoints from the paper's VP-SDE.
    assert np.isclose(beta(np.array([0.0]))[0], 0.1)
    assert np.isclose(beta(np.array([1.0]))[0], 20.0)
    # alpha(1) = exp(-0.5 * (0.1 + 9.95)).
    assert np.isclose(alpha(np.array([1.0]))[0], np.exp(-5.025), rtol=1e-9)


def test_true_score_symmetry_and_modes():
    # Symmetric mixture: score vanishes at x = 0 for every t.
    t = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(true_score(np.zeros(4), t), 0.0, atol=1e-12)
    # At t = 0 the score at a mode centre is (numerically) zero.
    assert abs(true_score(np.array([2.0]), np.array([0.0]))[0]) < 1e-6
    # At t = 1 the perturbed law is ~N(0, 1): score(1, 1) ~ -1.
    assert abs(true_score(np.array([1.0]), np.array([1.0]))[0] + 1.0) < 0.05


def test_reverse_drift_formula():
    xtau = np.array([[1.0, 0.0], [0.0, 0.5], [-2.0, 1.0]])
    s = 1.0 - xtau[:, 1]
    expected = 0.5 * beta(s) * xtau[:, 0] + beta(s) * true_score(xtau[:, 0], s)
    assert np.allclose(true_reverse_drift(xtau)[:, 0], expected)
    assert np.allclose(true_reverse_g(xtau)[:, 0], np.sqrt(beta(s)))


def test_forward_sample_matches_kernel():
    rng = np.random.default_rng(0)
    x0 = np.array([2.0] * 20000)
    t = np.full(20000, 0.5)
    xt = forward_sample(rng, x0, t)
    assert abs(xt.mean() - 2.0 * alpha(np.array([0.5]))[0]) < 0.02
    assert abs(xt.std() - np.sqrt(1.0 - alpha(np.array([0.5]))[0] ** 2)) < 0.02


def test_simulate_reverse_constant_and_failure():
    zero = lambda xa, tau: np.zeros((len(xa), 1))  # noqa: E731
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(6)
    dW = rng.standard_normal((6, 20))
    sim = simulate_reverse(zero, zero, x0, 0.05, 20, dW, 30.0)
    assert np.allclose(sim["final"], x0)
    assert sim["failure_rate"] == 0.0
    assert sim["excursion_rate"] == 0.0
    big = np.full(6, 40.0)
    sim_big = simulate_reverse(zero, zero, big, 0.05, 20, dW, 30.0)
    assert sim_big["failure_rate"] == 1.0
    assert sim_big["excursion_rate"] == 1.0


def test_w1_1d():
    a = np.array([0.0, 2.0])
    b = np.array([0.0, 2.0])
    assert w1_1d(a, b) == 0.0
    assert np.isclose(w1_1d(a, np.array([1.0, 3.0])), 1.0)


def test_config_thresholds_present():
    hyp = CFG["hypotheses"]
    assert hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"] == 0.3
    assert hyp["H4_simulation"]["w1_teacher_vs_truth_max"] == 0.35
    assert hyp["H5_paper_axis_reverse_vs_noscore_and_pfode"]["noscore_w1_vs_truth_min"] == 0.8
    assert hyp["H6_discretization_bias_reverse_step"]["w1_degradation_min"] == 0.03
