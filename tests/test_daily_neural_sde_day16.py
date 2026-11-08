"""Day 16 tests: Liu et al. (ICML 2026) LieStoNet utilities.

These tests do not train the full teacher. They check the Example-4 SDE,
the scaling identity, the Cholesky surrogate, failure accounting, and the
frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import fit_polynomial_field, nrmse
from src.daily_neural_sde.day16_liu_liestonet import (
    FrozenTeacher, Surrogate, build_model, central_jacobian, chol_parts,
    covariance_from_chol, decide, diffusion_selection_score, fit_standardizer,
    gaussian_nll_with_constant, glorot_, identity_stats, ito_x_squared_drift,
    load_config, make_dataset, mean_abs, notebook_nll, numpy_forward,
    parts_to_chol, scaling_residual, simulate_matrix, state_dict_numpy,
    train_surrogate, true_chol, true_drift, vif_max)

CFG = load_config()


def test_frozen_thresholds_match_the_pre_registration():
    assert CFG["paper"]["venue"] == "ICML"
    assert CFG["paper"]["year"] == 2026
    assert CFG["data"]["a1"] == 1.0
    assert CFG["data"]["a2"] == 1.0
    assert CFG["hypotheses"]["H1_drift"]["teacher_vs_truth_test_nrmse_max"] == 0.30
    assert CFG["hypotheses"]["H1_drift"]["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert CFG["hypotheses"]["H6_scaling"]["v3_teacher_mean_abs_max"] == 0.15
    assert CFG["hypotheses"]["H6_scaling"]["v3_symbolic_mean_abs_max"] == 0.15
    assert CFG["interpretation"]["coefficients_are_closed_form"] is False


def test_example4_drift_and_ito_correction_for_x_squared():
    state = np.array([[1.5, -0.2], [2.0, 0.4], [0.8, 1.0]])
    drift = true_drift(state, a1=1.0, a2=1.0)
    assert drift.shape == (3, 2)
    assert np.allclose(drift[:, 0], 1.0 / state[:, 0])
    assert np.allclose(drift[:, 1], 1.0)
    # d(x^2) = 2 x (a1/x) dt + sigma^2 dt + martingale = (2 a1 + sigma^2) dt.
    assert np.allclose(ito_x_squared_drift(state), 3.0)


def test_truth_satisfies_the_scaling_identity_and_a_cubic_does_not():
    rng = np.random.default_rng(0)
    state = np.stack([rng.uniform(0.6, 3.5, 500), rng.uniform(-1.0, 3.0, 500)], axis=1)
    target = true_drift(state)
    field = fit_polynomial_field(state, target, 3, 0.0)
    prediction = field.predict(state)
    assert prediction.shape == (500, 2)
    assert nrmse(target[:, 0], prediction[:, 0]) < 0.1
    times = np.ones(len(state))
    truth = identity_stats(lambda points: true_drift(points), state, times, 1.0, 1e-5)
    symbolic = identity_stats(field.predict, state, times, 1.0, 1e-5)
    assert truth["v3"] < 1e-6
    assert truth["v2"] < 1e-6
    assert symbolic["v3"] > 0.05


def test_cholesky_of_the_identity_is_psd_and_round_trips():
    state = np.ones((4, 2))
    factor = true_chol(state, sigma=1.0)
    assert factor.shape == (4, 2, 2)
    sigma = covariance_from_chol(factor)
    assert np.allclose(sigma, np.eye(2)[None, :, :])
    assert np.all(np.linalg.eigvalsh(sigma) >= -1e-12)
    assert np.allclose(parts_to_chol(chol_parts(factor)), factor)


def test_fields_stay_finite_on_a_safe_box():
    axis_x = np.linspace(0.3, 4.0, 6)
    axis_y = np.linspace(-2.0, 2.0, 5)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y, indexing="ij")
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    drift = true_drift(grid)
    factor = true_chol(grid)
    assert np.isfinite(drift).all()
    assert np.isfinite(factor).all()
    assert np.all(grid[:, 0] > 0.0)


def test_notebook_nll_drops_one_log_two_pi():
    rng = np.random.default_rng(1)
    increment = rng.normal(scale=0.1, size=(6, 2))
    drift = true_drift(np.stack([rng.uniform(0.5, 2.0, 6), rng.normal(size=6)], axis=1))
    factor = true_chol(drift)
    dt = 0.01
    gap = gaussian_nll_with_constant(increment, drift, factor, dt) - notebook_nll(
        increment, drift, factor, dt)
    assert np.allclose(gap, math.log(2.0 * math.pi))


def test_surrogate_shapes_diffusion_floor_and_numpy_replay():
    torch.manual_seed(0)
    model = Surrogate(hidden=8, depth=2, sigma_floor=0.001).double()
    glorot_(model)
    state = torch.randn(5, 2, dtype=torch.float64)
    drift, factor = model(state)
    assert drift.shape == (5, 2)
    assert factor.shape == (5, 2, 2)
    assert torch.all(factor[:, 0, 0] >= 0.001)
    assert torch.all(factor[:, 1, 1] >= 0.001)
    assert torch.all(factor[:, 0, 1] == 0.0)
    mean = np.array([0.1, -0.2])
    scale = np.array([1.2, 0.8])
    raw = np.random.default_rng(2).normal(size=(6, 2))
    replay_drift, replay_factor = numpy_forward(
        state_dict_numpy(model), raw, mean, scale, 0.001, depth=2)
    with torch.no_grad():
        torch_drift, torch_factor = model(torch.as_tensor((raw - mean) / scale, dtype=torch.float64))
    assert np.allclose(replay_drift, torch_drift.numpy(), atol=1e-10)
    assert np.allclose(replay_factor, torch_factor.numpy(), atol=1e-10)
    teacher = FrozenTeacher(model, mean, scale)
    out_drift, out_factor, parts = teacher.predict(raw)
    assert out_drift.shape == (6, 2)
    assert out_factor.shape == (6, 2, 2)
    assert parts.shape == (6, 3)


def test_symbolic_polynomial_matches_batch_shapes():
    rng = np.random.default_rng(3)
    state = rng.normal(size=(30, 2))
    target = np.stack([state[:, 0], np.ones(30)], axis=1)
    field = fit_polynomial_field(state, target, 2, 0.0)
    prediction = field.predict(state[:7])
    assert prediction.shape == (7, 2)
    assert np.isfinite(prediction).all()


def test_failure_classes_latch_and_keep_their_mass():
    dt = 0.01
    radius = 12.0
    floor = 0.2
    below = simulate_matrix(
        lambda state: true_drift(state), lambda state: true_chol(state),
        np.array([[0.25, 0.0]]), dt, np.array([[[-5.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        radius, floor)
    assert below["failure_rate"] == 1.0
    assert below["failure_class"] == ["below_floor"]
    assert np.isnan(below["states"][0, -1]).all()

    def negative_diag(state: np.ndarray) -> np.ndarray:
        factor = true_chol(state)
        factor[:, 0, 0] = -1.0
        return factor

    chol = simulate_matrix(
        lambda state: np.zeros((len(state), 2)), negative_diag,
        np.array([[1.0, 0.0]]), dt, np.zeros((1, 2, 2)), radius, floor)
    assert chol["failure_class"] == ["nonpositive_chol"]
    assert np.allclose(chol["states"][0, 0], [1.0, 0.0])
    assert np.isnan(chol["states"][0, 1]).all()

    escaped = simulate_matrix(
        lambda state: np.full((len(state), 2), 1.0e6), lambda state: true_chol(state),
        np.array([[1.0, 0.0]]), dt, np.zeros((1, 1, 2)), radius, floor)
    assert escaped["failure_class"] == ["outside_radius"]
    assert escaped["failure_rate"] == 1.0

    broken = simulate_matrix(
        lambda state: np.full((len(state), 2), np.nan), lambda state: true_chol(state),
        np.array([[1.0, 0.0]]), dt, np.zeros((1, 1, 2)), radius, floor)
    assert broken["failure_class"] == ["non_finite"]


def test_same_increments_reproduce_the_analytic_path():
    x0 = np.array([[1.2, -0.3], [0.9, 0.4]])
    increments = np.random.default_rng(4).normal(size=(2, 5, 2))
    kwargs = dict(dt=0.01, failure_radius=12.0, x_floor=0.2)
    left = simulate_matrix(lambda state: true_drift(state), lambda state: true_chol(state),
                           x0, increments=increments, **kwargs)
    right = simulate_matrix(lambda state: true_drift(state), lambda state: true_chol(state),
                            x0, increments=increments, **kwargs)
    assert np.allclose(left["states"], right["states"])
    assert left["failure_class"] == right["failure_class"]


def test_vif_excludes_the_constant_and_stays_finite():
    state = np.random.default_rng(5).normal(size=(200, 2))
    value = vif_max(state, 3)
    assert math.isfinite(value)
    assert 1.0 <= value < 100.0


def test_standardizer_uses_only_the_supplied_nonterminal_states():
    states = np.array([[[0.0, 0.0], [2.0, 4.0], [9.0, 9.0]]])
    mean, scale = fit_standardizer(states)
    assert np.allclose(mean, [1.0, 2.0])
    assert np.allclose(scale, [1.0, 2.0])


def test_surviving_dataset_respects_the_floor_and_the_split_sizes():
    cfg = load_config()
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["n_train"] = 3
    cfg["data"]["n_val"] = 2
    cfg["data"]["n_test"] = 2
    dataset = make_dataset(cfg, np.random.default_rng(6))
    for name, count in (("train", 3), ("val", 2), ("test", 2)):
        states = dataset[name]["states"]
        assert states.shape[0] == count
        assert states.shape[-1] == 2
        assert np.isfinite(states).all()
        assert np.all(states[:, :, 0] >= cfg["data"]["x_floor"])
        assert dataset[name]["attempts"] >= count


def test_diffusion_score_does_not_divide_by_a_zero_off_diagonal():
    teacher = np.array([[1.0, 0.0, 1.0], [1.1, 0.0, 0.9]])
    symbolic = np.array([[1.0, 0.05, 1.0], [1.1, -0.05, 0.9]])
    score = diffusion_selection_score(teacher, symbolic)
    assert math.isfinite(score)
    assert score > 0.0


def test_one_alternating_step_has_a_finite_validation_nll():
    cfg = load_config()
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["outer_steps"] = 2
    cfg["model"]["eval_every"] = 1
    cfg["model"]["batch_size"] = 16
    cfg["data"] = dict(cfg["data"])
    rng = np.random.default_rng(7)
    state = np.stack([rng.uniform(0.5, 3.0, 40), rng.uniform(-1.0, 1.0, 40)], axis=1)
    increment = rng.normal(scale=0.1, size=(40, 2))
    model = Surrogate(hidden=8, depth=1, sigma_floor=0.001).double()
    glorot_(model)
    info = train_surrogate(model, state[:30], increment[:30], state[30:], increment[30:],
                           state[:30].mean(axis=0), state[:30].std(axis=0), cfg)
    assert math.isfinite(info["best_val_nll"])


def _passing_metrics() -> dict:
    return {
        "drift_sym_teacher": [0.1, 0.1],
        "drift_teacher_truth": [0.1, 0.1],
        "drift_sym_truth": [0.1, 0.1],
        "diag_sym_teacher": [0.05, 0.05],
        "diag_teacher_truth": [0.05, 0.05],
        "diag_sym_truth": [0.05, 0.05],
        "l21_rms_teacher": 0.01,
        "l21_rms_sym_teacher": 0.01,
        "l21_rms_sym_truth": 0.01,
        "fraction_nonpositive": 0.0,
        "vif": 3.0,
        "fail_truth": 0.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "paired_sym_teacher": 0.1,
        "paired_teacher_truth": 0.1,
        "cov_sym_teacher": 0.1,
        "cov_teacher_truth": 0.1,
        "disc_rmse": 0.05,
        "disc_retained": 1.0,
        "law_fail_truth": 0.0,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "law_mean_teacher": 0.2,
        "law_mean_symbolic": 0.2,
        "law_cov_teacher": 0.2,
        "law_cov_symbolic": 0.2,
        "floor_mean": 0.15,
        "floor_cov": 0.2,
        "v2_teacher": 0.01,
        "v3_teacher": 0.05,
        "v3_symbolic": 0.05,
        "v3_oracle": 0.2,
    }


def test_value_match_can_pass_while_the_scaling_identity_fails():
    metrics = _passing_metrics()
    metrics["v3_symbolic"] = 0.30
    metrics["v3_oracle"] = 0.28
    verdicts = decide(CFG, metrics)
    assert verdicts["H1"] == "pass"
    assert verdicts["H6"] == "fail"
    assert verdicts["H6_reason"] == "library_breaks_scaling_identity"
    assert verdicts["coefficients_are_closed_form"] is False


def test_high_failure_rate_makes_the_path_comparison_indeterminate():
    metrics = _passing_metrics()
    metrics["fail_truth"] = 0.2
    metrics["fail_teacher"] = 0.2
    metrics["fail_symbolic"] = 0.2
    verdicts = decide(CFG, metrics)
    assert verdicts["H4_failure"] == "fail"
    assert verdicts["H4"] == "indeterminate"


def test_floor_above_the_teacher_gate_makes_h5_indeterminate():
    metrics = _passing_metrics()
    metrics["floor_mean"] = 0.5
    verdicts = decide(CFG, metrics)
    assert verdicts["H5"] == "indeterminate"


def test_central_jacobian_of_a_linear_field_is_exact():
    def field(state: np.ndarray) -> np.ndarray:
        return np.stack([2.0 * state[:, 0] + state[:, 1], -state[:, 0]], axis=1)

    state = np.array([[0.4, -0.2], [1.2, 0.3]])
    jacobian = central_jacobian(field, state, 1e-6)
    assert jacobian.shape == (2, 2, 2)
    assert np.allclose(jacobian[:, 0, 0], 2.0)
    assert np.allclose(jacobian[:, 0, 1], 1.0)
    assert np.allclose(jacobian[:, 1, 0], -1.0)
    residual = scaling_residual(field(state), jacobian, state, np.array([0.5, 0.5]), 1.0)
    # f = (2x + y, -x) is not the reciprocal field, so the residual is nonzero.
    assert mean_abs(residual) > 0.1


def test_build_model_uses_the_frozen_width():
    model = build_model(CFG)
    assert model.sigma_floor == 0.001
    drift, factor = model(torch.zeros(3, 2, dtype=torch.float64))
    assert drift.shape == (3, 2)
    assert factor.shape == (3, 2, 2)
    assert torch.isfinite(drift).all()
    assert torch.isfinite(factor).all()
