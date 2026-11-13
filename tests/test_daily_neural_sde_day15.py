"""Day 15 tests: Seifner et al. (NeurIPS 2025) FIM-SDE utilities.

These tests do not train the teacher. They check the synthetic SDE, the
recognition-model shapes, the NumPy replay, failure accounting, and the
frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import fit_polynomial_field, simulate_em
from src.daily_neural_sde.day15_seifner_fim import (
    FIM, choose_candidate, decide, drift_groups, evaluate_coefficient_field,
    fit_standardizer, gaussian_nll, heteroscedastic_parts, load_config,
    make_dataset, nll_constant_gap, numpy_forward, paper_eq34, state_dict_numpy,
    target_drift_coefficients, target_variance_coefficients, transition_features,
    true_diffusion, true_drift, vif_max)

CFG = load_config()


def test_unit_circle_drift_is_a_rotation():
    theta = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    state = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    drift = true_drift(state)
    assert drift.shape == (12, 2)
    assert np.allclose(drift, np.stack([-np.sin(theta), np.cos(theta)], axis=1))


def test_closed_form_matches_the_monomial_coefficients():
    state = np.random.default_rng(0).normal(size=(15, 2))
    assert np.allclose(true_drift(state),
                       evaluate_coefficient_field(state, target_drift_coefficients(), 3))
    variance = evaluate_coefficient_field(state, target_variance_coefficients(), 2)
    assert np.allclose(variance, true_diffusion(state) ** 2)
    assert np.all(true_diffusion(state) > 0.0)


def test_linear_plus_cubic_and_radial_plus_rotation_rebuild_the_drift():
    state = np.random.default_rng(1).normal(size=(10, 2))
    groups = drift_groups(state)
    assert np.allclose(groups["linear"] + groups["cubic"], true_drift(state))
    assert np.allclose(groups["radial"] + groups["rotational"], true_drift(state))


def test_fields_stay_finite_on_a_box():
    axis = np.linspace(-1.5, 1.5, 7)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    assert np.isfinite(true_drift(grid)).all()
    diffusion = true_diffusion(grid)
    assert np.isfinite(diffusion).all()
    assert np.all(diffusion > 0.0)


def test_equation_34_differs_from_the_gaussian_nll_by_a_constant():
    rng = np.random.default_rng(2)
    increment = rng.normal(size=(6, 2))
    drift = rng.normal(size=(6, 2))
    sigma = np.abs(rng.normal(size=(6, 2))) + 0.2
    dt = 0.02
    gap = gaussian_nll(increment, drift, sigma, dt) - paper_eq34(increment, drift, sigma, dt)
    assert np.allclose(gap, nll_constant_gap(dt, dimension=2))


def test_recognition_model_shapes_and_zero_initial_uncertainty():
    torch.manual_seed(0)
    model = FIM(hidden=8, sigma_floor=0.02).double()
    state = torch.randn(4, 2, dtype=torch.float64)
    context = torch.randn(5, 7, dtype=torch.float64)
    drift, sigma, uncertainty = model(state, context)
    assert drift.shape == (4, 2)
    assert sigma.shape == (4, 2)
    assert uncertainty.shape == (4,)
    assert torch.allclose(uncertainty, torch.zeros(4, dtype=torch.float64))
    assert torch.all(sigma >= 0.02)
    assert torch.isfinite(drift).all()


def test_numpy_replay_matches_torch():
    torch.manual_seed(1)
    model = FIM(hidden=8, sigma_floor=0.02).double()
    state = torch.randn(3, 2, dtype=torch.float64)
    context = torch.randn(6, 7, dtype=torch.float64)
    with torch.no_grad():
        drift, sigma, uncertainty = model(state, context)
    replay = numpy_forward(state.numpy(), context.numpy(), state_dict_numpy(model), 0.02)
    assert np.allclose(replay[0], drift.numpy(), atol=1e-8)
    assert np.allclose(replay[1], sigma.numpy(), atol=1e-8)
    assert np.allclose(replay[2], uncertainty.numpy(), atol=1e-8)


def test_heteroscedastic_stationary_point_has_zero_u_gradient():
    level = 0.4
    drift_hat = torch.zeros(3, 2, dtype=torch.float64)
    drift_true = torch.full((3, 2), math.sqrt(level / 2.0), dtype=torch.float64)
    sigma_hat = torch.zeros(3, 2, dtype=torch.float64)
    sigma_true = torch.full((3, 2), math.sqrt(level / 2.0), dtype=torch.float64)
    uncertainty = torch.full((3,), math.log(level), dtype=torch.float64, requires_grad=True)
    loss = heteroscedastic_parts(drift_hat, drift_true, sigma_hat, sigma_true, uncertainty)
    loss.backward()
    assert torch.allclose(uncertainty.grad, torch.zeros_like(uncertainty), atol=1e-8)


def test_standardizer_uses_only_the_supplied_rows():
    train = np.array([[0.0, 0.0], [2.0, 0.0]])
    mean, scale = fit_standardizer(train)
    assert np.allclose(mean, [1.0, 0.0])
    assert np.allclose(scale, [1.0, 1.0])
    dirty = np.array([[0.0, 0.0], [np.nan, 4.0], [2.0, 0.0]])
    mean_dirty, _scale_dirty = fit_standardizer(dirty)
    assert np.allclose(mean_dirty, [1.0, 0.0])


def test_transition_features_have_seven_columns():
    state = np.array([[0.0, 1.0], [1.0, -1.0]])
    increment = np.array([[0.1, -0.2], [0.0, 0.3]])
    features = transition_features(state, increment, 0.02, np.zeros(2), np.ones(2))
    assert features.shape == (2, 7)
    assert np.allclose(features[:, -1], 0.02)


def test_same_seed_paths_match_and_splits_are_disjoint():
    first = make_dataset(CFG, np.random.default_rng(4))
    second = make_dataset(CFG, np.random.default_rng(4))
    assert np.allclose(first["train"]["states"], second["train"]["states"])
    assert first["train"]["x0"].shape[0] == CFG["data"]["n_train"]
    stacked = np.concatenate([first[name]["x0"] for name in ("train", "val", "test")], axis=0)
    rounded = np.round(stacked, decimals=8)
    assert len(np.unique(rounded, axis=0)) == len(rounded)


def test_symbolic_field_shape_and_no_nan_on_a_box():
    rng = np.random.default_rng(5)
    state = rng.normal(size=(30, 2))
    target = true_drift(state)
    field = fit_polynomial_field(state, target, degree=2, threshold=0.05)
    grid = rng.normal(size=(8, 2))
    prediction = field.predict(grid)
    assert prediction.shape == (8, 2)
    assert np.isfinite(prediction).all()


def test_failure_is_latched_and_counted():
    x0 = np.zeros((3, 2))
    increments = np.zeros((3, 2, 2))
    increments[2, 0, 0] = 100.0

    def drift(x):
        return np.zeros((len(x), 2))

    def sigma(x):
        out = np.ones((len(x), 2))
        if len(x) >= 2:
            out[1] = np.nan
        return out

    simulated = simulate_em(drift, sigma, x0, 0.02, 2, increments, 8.0)
    assert simulated["failure_class"][0] is None
    assert simulated["failure_class"][1] == "non_finite"
    assert simulated["failure_class"][2] == "outside_radius"
    assert simulated["failure_rate"] == 2.0 / 3.0
    assert np.isfinite(simulated["states"][0, -1]).all()
    assert np.isnan(simulated["states"][1, -1]).all()
    assert np.isnan(simulated["states"][2, -1]).all()


def test_vif_excludes_the_constant_and_stays_finite():
    state = np.random.default_rng(6).normal(size=(80, 2))
    value = vif_max(state, degree=3)
    assert math.isfinite(value)
    assert value < 1e6


def test_candidate_ties_follow_priority_and_a_real_gap_does_not():
    scores = {"finetuned": 1.0, "zero_shot": 1.00005, "scratch": 0.5}
    assert choose_candidate(scores, ["finetuned", "zero_shot", "scratch"], 1e-4) == "scratch"
    tied = {"finetuned": 1.0, "zero_shot": 1.00005, "scratch": 1.00002}
    assert choose_candidate(tied, ["finetuned", "zero_shot", "scratch"], 1e-4) == "finetuned"


def _metrics(**overrides):
    base = {
        "drift_sym_teacher": [0.1, 0.1],
        "drift_teacher_truth": [0.1, 0.1],
        "drift_sym_truth": [0.1, 0.1],
        "sigma_sym_teacher": [0.1, 0.1],
        "sigma_teacher_truth": [0.1, 0.1],
        "sigma_sym_truth": [0.1, 0.1],
        "variance_sym_teacher": 0.1,
        "fraction_sigma_nonpositive": 0.0,
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
        "law_mean_teacher": 0.1,
        "law_mean_symbolic": 0.1,
        "law_cov_teacher": 0.1,
        "law_cov_symbolic": 0.1,
        "floor_mean": 0.05,
        "floor_cov": 0.05,
        "u_rise": 0.5,
        "penalty_ratio": 0.2,
    }
    base.update(overrides)
    return base


def test_decide_passes_when_every_gate_holds():
    verdicts = decide(CFG, _metrics())
    assert verdicts["H1"] == "pass"
    assert verdicts["H2"] == "pass"
    assert verdicts["H4"] == "pass"
    assert verdicts["H5"] == "pass"
    assert verdicts["H6"] == "pass"
    assert verdicts["H6_reason"] == "pretrained_rise"
    assert verdicts["coefficients_are_closed_form"] is False
    assert verdicts["H4_path_match_without_field_recovery"] is False


def test_decide_flags_path_match_without_field_recovery():
    verdicts = decide(CFG, _metrics(drift_teacher_truth=[0.8, 0.1]))
    assert verdicts["H1"] == "fail"
    assert verdicts["H4"] == "fail"
    assert verdicts["H4_path_match_without_field_recovery"] is True


def test_decide_makes_h5_indeterminate_when_the_floor_exceeds_the_gate():
    verdicts = decide(CFG, _metrics(floor_mean=0.9))
    assert verdicts["H5"] == "indeterminate"


def test_decide_does_not_treat_a_negligible_uncertainty_term_as_recovery():
    verdicts = decide(CFG, _metrics(penalty_ratio=0.001, u_rise=1.0))
    assert verdicts["H6"] == "fail"
    assert verdicts["H6_reason"] == "uncertainty_term_negligible"


def test_decide_fails_a_flat_uncertainty_head():
    verdicts = decide(CFG, _metrics(u_rise=1e-4, penalty_ratio=0.2))
    assert verdicts["H6"] == "fail"
    assert verdicts["H6_reason"] == "no_ood_rise"


def test_high_vif_is_indeterminate_and_not_a_closed_form():
    verdicts = decide(CFG, _metrics(vif=25.0))
    assert verdicts["H3"] == "indeterminate"
    assert verdicts["H3_closed_form"] is False
