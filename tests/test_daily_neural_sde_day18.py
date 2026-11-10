"""Day 18 tests: Guan et al. (AISTATS 2026) nn-APPEX utilities.

These tests do not train the full teacher. They check the Styblinski drift,
the multi-marginal coupling, the polynomial ceiling, failure accounting, and
the frozen hypothesis gates.
"""
import math

import numpy as np

from src.daily_neural_sde.common import evaluate_expanded, nrmse
from src.daily_neural_sde.day18_guan_nnappex import (
    decide, diffusion_prior, expand_gradient, fit_gradient_field, fit_polynomial_field,
    gaussian_cost, load_config, mmos_sinkhorn, run_simulation, sigma2_mle, true_drift,
    true_sigma, vif_max, zero_drift)

CFG = load_config()


def _metrics(**overrides) -> dict:
    drift = {"sym_teacher_nrmse": 0.10, "teacher_truth_nrmse": 0.20,
             "sym_truth_nrmse": 0.22, "teacher_truth_rms": 1.0}
    base = {
        "drift_test": drift,
        "sigma_test_nrmse": 0.01,
        "sigma_nonpositive_fraction": 0.0,
        "sigma2_relative_error": 0.05,
        "sigma_grid_nrmse": 0.01,
        "vif_degree3": 4.0,
        "fail_truth": 0.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "paired_sym_teacher": 0.02,
        "paired_teacher_truth": 0.03,
        "var_sym_teacher": 0.1,
        "var_teacher_truth": 0.1,
        "law_fail_truth": 0.0,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "floor_mean": 0.02,
        "floor_var": 0.02,
        "law_mean_teacher": 0.04,
        "law_mean_symbolic": 0.05,
        "law_var_teacher": 0.1,
        "law_var_symbolic": 0.1,
        "shuffle_truth": 1.1,
        "shuffle_teacher": 1.05,
        "shuffle_symbolic": 1.15,
        "grid_teacher_truth_nrmse": 0.3,
        "grid_sym_teacher_nrmse": 0.2,
        "grid_teacher_truth_rms": 1.5,
        "disc_rmse": 0.1,
        "disc_retained": 1.0,
        "fixed_teacher_truth_nrmse": 0.40,
    }
    base.update(overrides)
    return base


def test_frozen_thresholds_match_the_pre_registration():
    assert CFG["paper"]["venue"] == "AISTATS"
    assert CFG["paper"]["year"] == 2026
    assert CFG["data"]["sigma2"] == 0.2
    assert CFG["data"]["dt"] == 0.01
    assert CFG["data"]["n_steps"] == 2
    assert CFG["model"]["width"] == 128
    assert CFG["model"]["depth"] == 2
    assert CFG["model"]["conservative"] is True
    assert CFG["hypotheses"]["H1_drift"]["teacher_vs_truth_test_nrmse_max"] == 0.4
    assert CFG["hypotheses"]["H1_drift"]["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert CFG["hypotheses"]["H2_diffusion"]["sigma2_relative_error_max"] == 0.2
    assert CFG["hypotheses"]["H3_collinearity_gate"]["vif_max"] == 10.0
    assert CFG["hypotheses"]["H7_discretization"]["terminal_rmse_max"] == 0.4
    assert CFG["threshold_basis"]["probe_seed"] == 18099
    assert CFG["interpretation"]["diffusion_teacher_is_neural"] is False
    assert CFG["interpretation"]["coefficients_are_closed_form"] is False
    assert CFG["interpretation"]["distillation_discovers_true_sde"] is False
    prior = diffusion_prior(CFG)
    assert abs(prior - 0.1257886725699169) < 1e-12
    assert abs(prior - CFG["data"]["sigma2"]) / CFG["data"]["sigma2"] > 0.2


def test_styblinski_drift_is_the_negative_gradient():
    point = np.array([[1.0, 2.0]])
    expected = np.array([[11.5, 13.5]])
    np.testing.assert_allclose(true_drift(point), expected)
    step = 1e-6
    numerical = np.zeros(2)
    for axis in range(2):
        plus = point.copy()
        minus = point.copy()
        plus[0, axis] += step
        minus[0, axis] -= step
        potential = lambda z: float(0.5 * np.sum(z ** 4 - 16.0 * z ** 2 + 5.0 * z))
        numerical[axis] = (potential(plus) - potential(minus)) / (2 * step)
    np.testing.assert_allclose(true_drift(point)[0], -numerical, atol=1e-6)
    assert abs(true_sigma(0.2) - math.sqrt(0.2)) < 1e-15


def test_degree3_library_recovers_the_cubic_drift():
    rng = np.random.default_rng(18)
    train = rng.normal(size=(300, 2))
    hold = rng.normal(size=(60, 2))
    target = true_drift(train)
    gradient = fit_gradient_field(train, target, 3, 0.0, 0.0)
    free = fit_polynomial_field(train, target, 3, 0.0, 0.0)
    assert nrmse(true_drift(hold), gradient.predict(hold)) < 1e-8
    assert nrmse(true_drift(hold), free.predict(hold)) < 1e-8
    expanded = expand_gradient(gradient)
    assert nrmse(gradient.predict(hold), evaluate_expanded(expanded, hold)) < 1e-8


def test_sinkhorn_paths_match_the_marginal_shape():
    rng = np.random.default_rng(4)
    states = rng.normal(size=(24, 3, 2))
    out = mmos_sinkhorn(states, 0.01, zero_drift, 0.2, rng, 80, 1e-5)
    assert out["paths"].shape == (24, 3, 2)
    assert np.isfinite(out["paths"]).all()
    assert out["marginal_error"] < 1e-4
    cost = gaussian_cost(states[:, 0], states[:, 0], np.zeros_like(states[:, 0]), np.eye(2))
    np.testing.assert_allclose(np.diag(cost), 0.0)


def test_sigma2_mle_on_true_paths_is_near_the_diffusivity():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1500, 2))
    dt = 0.01
    sigma = true_sigma(0.2)
    increment = true_drift(x) * dt + sigma * math.sqrt(dt) * rng.standard_normal(x.shape)
    paths = np.stack([x, x + increment], axis=1)
    estimate = sigma2_mle(paths, dt, true_drift)
    assert abs(estimate - 0.2) / 0.2 < 0.15


def test_vif_excludes_the_constant_column():
    rng = np.random.default_rng(6)
    cloud = rng.normal(size=(200, 2))
    value = vif_max(cloud, 3)
    assert math.isfinite(value)
    assert value < 1e6


def test_failure_mass_is_retained_outside_the_radius():
    rng = np.random.default_rng(7)
    x0 = np.zeros((16, 2))
    increments = rng.standard_normal((16, 2, 2))
    exploding = lambda x: np.full_like(x, 1.0e4)  # noqa: E731
    out = run_simulation(exploding, true_sigma(0.2), x0, increments, 0.01, 12.0)
    assert out["failure_rate"] == 1.0
    assert all(item == "outside_radius" for item in out["failure_class"])
    assert np.isnan(out["states"][:, -1]).all()


def test_gates_pass_only_inside_the_frozen_thresholds():
    good = decide(CFG, _metrics())
    assert good["H1"] == "pass"
    assert good["H2_distill"] == "pass"
    assert good["H2_truth"] == "pass"
    assert good["H3"] == "pass"
    assert good["H3_closed_form"] is False
    assert good["H4"] == "pass"
    assert good["H5"] == "pass"
    assert good["H6"] == "pass"
    assert good["H7"] == "pass"
    assert good["H8"] == "pass"
    bad_field = decide(CFG, _metrics(drift_test={
        "sym_teacher_nrmse": 0.9, "teacher_truth_nrmse": 0.9,
        "sym_truth_nrmse": 0.9, "teacher_truth_rms": 4.0}))
    assert bad_field["H1"] == "fail"
    assert bad_field["H4"] == "fail"
    collinear = decide(CFG, _metrics(vif_degree3=15.3))
    assert collinear["H3"] == "indeterminate"
    buried = decide(CFG, _metrics(floor_mean=0.2))
    assert buried["H5"] == "indeterminate"
    unstable = decide(CFG, _metrics(fail_teacher=0.2))
    assert unstable["H4"] == "indeterminate"
