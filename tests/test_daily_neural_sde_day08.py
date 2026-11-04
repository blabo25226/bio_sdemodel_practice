"""Day 8 tests: Jia & Benson (2019) Neural Jump SDEs, jump-diffusion
(numpy only).

These tests do not train the MLP teachers; they check the
ground-truth jump-diffusion identities, the data generator, the
transition/coarse constructions, the origin-pinned library (the Day 7
structural-constraint lesson: the expanded polynomial must genuinely
vanish at x = 0), and the paired jump simulator's clipping count.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day08_jia_jump import (
    build_transitions, coarse_transitions, expand_to_original,
    fit_field_custom, generate_data, simulate_jump, true_drift, true_g,
    true_lambda, true_mu, true_sigma_j)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day08.json").read_text())


def _small_cfg():
    small = dict(CFG)
    small["data"] = dict(CFG["data"], n_trajectories=16)
    return small


def test_true_jump_diffusion_identities():
    x = np.array([[0.0], [2.0], [-2.0]])
    assert np.allclose(true_drift(x), [[0.0], [-1.2], [1.2]])
    assert np.allclose(true_g(x), 0.35)
    assert np.allclose(true_lambda(x), [[0.25], [0.57], [0.57]])
    assert np.allclose(true_mu(x), -0.80)
    assert np.allclose(true_sigma_j(x), 0.50)


def test_generate_data_shape_and_initial_states():
    data = generate_data(_small_cfg())
    assert data["obs"].shape == (16, 201, 1)
    assert np.isfinite(data["obs"]).all()
    assert np.allclose(data["obs"][:, 0, :], data["x0"])
    # Jumps are present: some observed increments must be large
    # relative to the diffusion-only step sd (0.35 * sqrt(0.05) = 0.078).
    increments = np.abs(np.diff(data["obs"][:, :, 0], axis=1))
    assert np.mean(increments > 0.5) > 0.0


def test_transition_alignment_and_coarsening():
    data = generate_data(_small_cfg())
    tr = build_transitions(data["obs"], 0.05)
    assert len(tr["cur"]) == 16 * 200
    assert np.allclose(tr["cur"][:16], data["obs"][:, 0, :])
    assert np.allclose(tr["nxt"][:16], data["obs"][:, 1, :])
    coarse = coarse_transitions(data["obs"], 0.05, 10)
    assert len(coarse["cur"]) == 16 * 20
    assert np.allclose(coarse["dt"], 0.5)
    assert np.allclose(coarse["cur"][:16], data["obs"][:, 0, :])


def test_origin_pinned_fit_has_no_intercept():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 1)) * 1.5
    y = true_drift(x) + rng.standard_normal((2000, 1)) * 0.01
    field = fit_field_custom(x, y, 2, 0.05, 1e-4, intercept=False, origin=True)
    expanded = expand_to_original(field)[0]
    assert (0,) not in expanded
    assert abs(expanded[(1,)] + 0.60) < 0.05


def test_simulate_jump_no_jump_and_negative_lambda_clip():
    n, steps = 8, 20
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((n, 1))
    dW = rng.standard_normal((n, steps))
    uniforms = rng.random((n, steps))
    jump_normals = rng.standard_normal((n, steps))
    zero = lambda x: np.zeros((len(x), 1))
    const_g = lambda x: np.full((len(x), 1), 0.35)
    neg_lam = lambda x: np.full((len(x), 1), -1.0)
    sim = simulate_jump(true_drift, const_g, neg_lam, true_mu, true_sigma_j,
                         x0, 0.05, steps, dW, uniforms, jump_normals, 30.0, 8.0)
    # Negative lambda is clipped at 0: no jumps occur, but every step
    # is counted as clipped.
    assert sim["jump_step_rate"] == 0.0
    assert sim["clip_rate"] == 1.0
    assert sim["failure_rate"] == 0.0
    lam0 = lambda x: np.zeros((len(x), 1))
    sim0 = simulate_jump(true_drift, const_g, lam0, true_mu, true_sigma_j,
                          x0, 0.05, steps, dW, uniforms, jump_normals, 30.0, 8.0)
    assert sim0["clip_rate"] == 0.0
    assert np.allclose(sim["states"], sim0["states"], equal_nan=True)
