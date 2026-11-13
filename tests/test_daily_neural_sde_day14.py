"""Day 14 tests: Koprulu, Djeumou & Topcu (ICLR 2025) NUNO utilities.

These tests do not train the teacher. They check the cubic-drag law, the
distance-aware diffusion, failure accounting, the NumPy replay, and the
frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import fit_polynomial_field
from src.daily_neural_sde.day14_koprulu_nuno import (
    NunoModel, behavior_action, coarse_increments, decide, load_config,
    make_dataset, numpy_callables, pack_weights, parity_error,
    polynomial_callables, sample_x0, simulate_controlled, strong_convexity_deficit,
    teacher_fields, true_diffusion, true_drift, vif_max)

CFG = load_config()


def test_true_drift_matches_the_cubic_drag_law():
    q = np.array([0.2, -0.4])
    v = np.array([0.5, 1.0])
    u = np.array([-0.3, 0.7])
    drift = true_drift(q, v, u, 1.0, 0.3, 0.8)
    assert drift.shape == (2, 2)
    assert np.allclose(drift[:, 0], v)
    expected_v = 1.0 * u - 0.3 * v - 0.8 * v ** 3
    assert np.allclose(drift[:, 1], expected_v)
    assert np.allclose(true_diffusion(q, 0.15), [0.15, 0.15])


def test_behavior_action_is_clipped():
    q = np.array([10.0, 0.0])
    v = np.array([0.0, 0.0])
    noise = np.array([0.0, 5.0])
    u = behavior_action(q, v, noise, CFG)
    assert u[0] == CFG["data"]["behavior"]["u_min"]
    assert u[1] == CFG["data"]["behavior"]["u_max"]


def test_position_drift_of_the_teacher_is_the_velocity():
    torch.manual_seed(0)
    model = NunoModel(CFG, epistemic=True)
    model.set_normalization(np.zeros(3), np.ones(3))
    q = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    v = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    u = torch.tensor([0.1, 0.0, -0.2], dtype=torch.float64)
    drift, diffusion, eta, residual = model.fields(q, v, u)
    assert drift.shape == (3, 2)
    assert diffusion.shape == (3,)
    assert eta.shape == (3, 1)
    assert residual.shape == (3,)
    assert torch.allclose(drift[:, 0], v)
    assert torch.isfinite(drift).all()
    assert torch.isfinite(diffusion).all()


def test_epistemic_diffusion_is_monotone_and_bounded():
    torch.manual_seed(1)
    model = NunoModel(CFG, epistemic=True)
    eta = torch.linspace(-8.0, 8.0, 17, dtype=torch.float64).reshape(-1, 1)
    h = model.epistemic_diffusion(eta)
    assert torch.all(h[1:] >= h[:-1] - 1e-12)
    assert torch.all(h > 0.0)
    assert torch.all(h < model.wmax)
    # A large positive eta saturates under the cap.
    h_hi = model.epistemic_diffusion(torch.tensor([[40.0]], dtype=torch.float64))
    assert torch.allclose(h_hi, torch.tensor([model.wmax], dtype=torch.float64), atol=1e-5)


def test_full_diffusion_adds_the_epistemic_term():
    torch.manual_seed(2)
    model = NunoModel(CFG, epistemic=True)
    model.set_normalization(np.zeros(3), np.ones(3))
    q = torch.zeros(4, dtype=torch.float64)
    v = torch.linspace(-0.2, 0.2, 4, dtype=torch.float64)
    u = torch.zeros(4, dtype=torch.float64)
    _drift, diffusion, eta, _res = model.fields(q, v, u)
    z = model._z(q, v, u)
    expected = model.aleatoric(z) + model.epistemic_diffusion(eta)
    assert torch.allclose(diffusion, expected)


def test_square_is_on_the_strong_convexity_boundary_at_unit_mu():
    x = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64)
    y = torch.tensor([[0.5, 0.4, -0.7]], dtype=torch.float64)
    # eta(z) = ||z||^2 has grad 2z and slack ||y - x||^2 - mu ||y - x||^2.
    eta_x = (x ** 2).sum(dim=-1, keepdim=True)
    eta_y = (y ** 2).sum(dim=-1, keepdim=True)
    grad = 2.0 * x
    delta = y - x
    mu = torch.ones(1, 1, dtype=torch.float64)
    deficit = strong_convexity_deficit(eta_y, eta_x, grad, delta, mu)
    assert torch.allclose(deficit, torch.zeros_like(deficit), atol=1e-12)
    mu_tight = torch.full((1, 1), 2.0, dtype=torch.float64)
    violated = strong_convexity_deficit(eta_y, eta_x, grad, delta, mu_tight)
    assert torch.all(violated < 0.0)


def test_failure_is_latched_and_counted():
    x0 = np.array([[0.0, 0.0], [0.0, 0.0], [12.0, 0.0]])
    actions = np.zeros((3, 1))
    increments = np.zeros((3, 1))
    increments[1, 0] = 1.0

    def drift(x, u):
        return np.zeros((len(x), 2))

    def diffusion(x, u):
        out = np.zeros(len(x))
        # The third row is already outside the radius, so the active batch is
        # the first two rows. Poison the second of those.
        out[-1] = np.nan
        return out

    sim = simulate_controlled(drift, diffusion, x0, actions, 0.05, increments, 8.0)
    assert sim["failure_class"][0] is None
    assert sim["failure_class"][1] == "non_finite"
    assert sim["failure_class"][2] == "outside_radius"
    assert sim["failure_rate"] == 2.0 / 3.0
    assert np.allclose(sim["states"][0, -1], [0.0, 0.0])
    assert not np.isfinite(sim["states"][1, -1]).any()


def test_same_seed_logs_match_and_splits_are_disjoint_lengths():
    first = make_dataset(CFG)
    second = make_dataset(CFG)
    assert first["train"]["states"].shape[0] == CFG["split"]["train"]
    assert first["validation"]["states"].shape[0] == CFG["split"]["validation"]
    assert first["test"]["states"].shape[0] == CFG["split"]["test"]
    assert np.allclose(first["train"]["states"], second["train"]["states"])
    assert np.allclose(first["test"]["actions"], second["test"]["actions"])
    # A trajectory's terminal state is not a query row of another split's start.
    assert not np.allclose(first["train"]["states"][0, 0], first["test"]["states"][0, 0])


def test_numpy_replay_matches_torch_fields():
    torch.manual_seed(4)
    model = NunoModel(CFG, epistemic=True)
    model.set_normalization(np.array([0.1, -0.2, 0.3]), np.array([1.2, 0.8, 1.1]))
    weights = pack_weights(model)
    assert parity_error(model, weights, seed=11) < 1e-10
    drift_fn, diffusion_fn = numpy_callables(weights)
    x = np.array([[0.2, -0.4], [0.0, 0.1]])
    u = np.array([0.3, -0.2])
    assert drift_fn(x, u).shape == (2, 2)
    assert diffusion_fn(x, u).shape == (2,)
    assert np.isfinite(drift_fn(x, u)).all()
    assert np.isfinite(diffusion_fn(x, u)).all()


def test_same_seed_simulation_is_reproducible():
    drift, diffusion = numpy_callables(pack_weights(NunoModel(CFG, epistemic=False)))
    # The aleatoric model above is a fresh init. Rebuild with a fixed seed.
    torch.manual_seed(5)
    weights = pack_weights(NunoModel(CFG, epistemic=False))
    weights["mean"] = [0.0, 0.0, 0.0]
    weights["std"] = [1.0, 1.0, 1.0]
    drift, diffusion = numpy_callables(weights)
    x0 = np.array([[0.1, 0.2], [-0.3, 0.0]])
    actions = np.array([[0.2, -0.1], [0.0, 0.4]])
    increments = np.array([[0.3, -0.2], [0.1, 0.5]])
    first = simulate_controlled(drift, diffusion, x0, actions, 0.05, increments, 8.0)
    second = simulate_controlled(drift, diffusion, x0, actions, 0.05, increments, 8.0)
    assert np.allclose(first["states"], second["states"])
    assert first["failure_rate"] == 0.0


def test_coarse_increments_have_unit_variance_in_expectation():
    rng = np.random.default_rng(0)
    fine = rng.standard_normal((20000, 3, 5))
    coarse = coarse_increments(fine)
    assert coarse.shape == (20000, 3)
    assert abs(float(coarse.var()) - 1.0) < 0.02


def test_symbolic_callbacks_match_the_polynomial_field():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    y = np.column_stack([x[:, 1], x[:, 2] - 0.3 * x[:, 1]])
    g = np.full((40, 1), 0.15)
    drift_field = fit_polynomial_field(x, y, 1, 0.0, 1e-8)
    g_field = fit_polynomial_field(x, g, 1, 0.0, 1e-8)
    drift_fn, diffusion_fn = polynomial_callables(drift_field, g_field)
    state = np.array([[0.2, -0.1], [0.0, 0.4]])
    action = np.array([0.3, -0.2])
    assert drift_fn(state, action).shape == (2, 2)
    assert diffusion_fn(state, action).shape == (2,)
    direct = drift_field.predict(np.column_stack([state, action]))
    assert np.allclose(drift_fn(state, action), direct)


def test_vif_excludes_the_constant_and_stays_finite():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(300, 3))
    value = vif_max(features, 3)
    assert math.isfinite(value)
    assert value < 100.0


def test_dataset_initial_box_is_respected():
    rng = np.random.default_rng(3)
    x0 = sample_x0(rng, 200, CFG)
    q_lo, q_hi = CFG["data"]["x0_q"]
    v_lo, v_hi = CFG["data"]["x0_v"]
    assert np.all(x0[:, 0] >= q_lo) and np.all(x0[:, 0] <= q_hi)
    assert np.all(x0[:, 1] >= v_lo) and np.all(x0[:, 1] <= v_hi)


def _metrics(**overrides):
    base = {
        "drift_v_sym_teacher": 0.1,
        "drift_v_teacher_truth": 0.2,
        "drift_v_sym_truth": 0.25,
        "drift_q_sym_teacher": 0.05,
        "g_sym_teacher": 0.1,
        "g_teacher_truth": 0.2,
        "g_sym_truth": 0.25,
        "sigma_sym_teacher": 0.2,
        "fraction_g_nonpositive": 0.0,
        "vif": 4.0,
        "fail_truth": 0.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "mean_sym_teacher": 0.05,
        "mean_teacher_truth": 0.08,
        "cov_sym_teacher": 0.1,
        "cov_teacher_truth": 0.2,
        "disc_rmse": 0.05,
        "disc_retained": 1.0,
        "law_fail_truth": 0.0,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "law_mean_teacher": 0.05,
        "law_mean_symbolic": 0.08,
        "law_cov_teacher": 0.1,
        "law_cov_symbolic": 0.2,
        "floor_mean": 0.02,
        "floor_cov": 0.05,
        "g_full_truth": 0.2,
        "g_aleatoric_truth": 0.3,
        "h_ood_minus_id": 0.1,
        "g_lift_full": 0.12,
        "g_lift_aleatoric": 0.01,
    }
    base.update(overrides)
    return base


def test_gates_pass_on_a_clean_metric_card():
    verdicts = decide(CFG, _metrics())
    assert verdicts["H1"] == "pass"
    assert verdicts["H2"] == "pass"
    assert verdicts["H4"] == "pass"
    assert verdicts["H4_path_match_without_field_recovery"] is False
    assert verdicts["H5"] == "pass"
    assert verdicts["H6"] == "pass"
    assert verdicts["H6_shared_floor"] is False
    assert verdicts["H7"] == "pass"


def test_path_match_without_drift_recovery_fails_h4():
    verdicts = decide(CFG, _metrics(drift_v_teacher_truth=0.9))
    assert verdicts["H1"] == "fail"
    assert verdicts["H4"] == "fail"
    assert verdicts["H4_path_match_without_field_recovery"] is True


def test_high_failure_makes_h4_indeterminate():
    verdicts = decide(CFG, _metrics(fail_symbolic=0.2))
    assert verdicts["H4"] == "indeterminate"
    assert verdicts["H4_failure"] == "fail"


def test_floor_above_the_threshold_makes_h5_indeterminate():
    verdicts = decide(CFG, _metrics(floor_mean=0.5))
    assert verdicts["H5"] == "indeterminate"


def test_shared_floor_is_not_an_h6_win():
    verdicts = decide(CFG, _metrics(g_full_truth=0.9, g_aleatoric_truth=0.91,
                                    h_ood_minus_id=0.2, g_lift_full=0.3,
                                    g_lift_aleatoric=0.0))
    assert verdicts["H6_shared_floor"] is True
    assert verdicts["H6"] == "fail"


def test_vif_above_the_gate_is_indeterminate():
    verdicts = decide(CFG, _metrics(vif=25.0))
    assert verdicts["H3"] == "indeterminate"


def test_teacher_query_shapes_from_packed_weights():
    torch.manual_seed(6)
    weights = pack_weights(NunoModel(CFG, epistemic=True))
    drift, diffusion, eta = teacher_fields(weights, np.zeros(5), np.ones(5), np.full(5, 0.2))
    assert drift.shape == (5, 2)
    assert diffusion.shape == (5,)
    assert eta.shape == (5,)
    assert np.allclose(drift[:, 0], 1.0)
