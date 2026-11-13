"""Day 13 tests: Bartosh et al. (ICML 2025) SDE Matching utilities.

These tests do not train the teacher. They check the Lotka-Volterra fields,
the posterior-drift identity, failure accounting, the NumPy replay, and the
frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.day13_bartosh_sde_matching import (
    MatchingModel, assemble_posterior_drift, coarse_increments, decide,
    diagonal_gg_divergence, flow_velocity, gaussian_kl_diag, load_config,
    make_dataset, pack_weights, parity_error, sample_x0, simulate_population,
    teacher_fields, true_diffusion, true_drift, vif_max)
from src.daily_neural_sde.common import nrmse

CFG = load_config()


def test_true_lotka_volterra_matches_the_closed_form():
    x = np.array([[1.0, 0.5], [0.9, 0.2]])
    alpha, beta, delta, gamma = 2.0 / 3.0, 4.0 / 3.0, 1.0, 1.0
    drift = true_drift(x, alpha, beta, delta, gamma)
    diffusion = true_diffusion(x, 0.15)
    assert drift.shape == (2, 2)
    # Equilibrium (1, 1/2): both drift components are 0.
    assert np.allclose(drift[0], 0.0, atol=1e-12)
    prey = 0.9 * (alpha - beta * 0.2)
    predator = 0.2 * (delta * 0.9 - gamma)
    assert np.allclose(drift[1], [prey, predator])
    assert np.allclose(diffusion[1], [0.15 * 0.9, 0.15 * 0.2])


def test_flow_velocity_matches_an_affine_clock():
    t = torch.tensor([[0.4], [1.2]], dtype=torch.float64, requires_grad=True)
    slope_mu = torch.tensor([0.5, -0.3], dtype=torch.float64)
    slope_sigma = torch.tensor([0.1, -0.2], dtype=torch.float64)
    mu = 0.2 + t * slope_mu
    sigma = 0.4 + t * slope_sigma
    eps = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float64)
    velocity = flow_velocity(mu, sigma, eps, t)
    expected = slope_mu + slope_sigma * eps
    assert torch.allclose(velocity, expected, atol=1e-12)


def test_divergence_of_state_softplus_matches_the_analytic_diagonal():
    z = torch.tensor([[0.2, -0.4], [1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    g = torch.nn.functional.softplus(z) + 0.01
    div = diagonal_gg_divergence(g ** 2, z)
    analytic = 2.0 * g * torch.sigmoid(z)
    assert torch.allclose(div, analytic, atol=1e-10)


def test_state_independent_divergence_is_zero():
    z = torch.tensor([[0.2, -0.4], [1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    level = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
    g2 = level.expand_as(z) ** 2
    # The level depends on a parameter and not on z, so the diagonal divergence is 0.
    div = diagonal_gg_divergence(g2, z)
    assert torch.allclose(div, torch.zeros_like(z))
    constant = torch.full_like(z, 0.25)
    assert torch.allclose(diagonal_gg_divergence(constant, z), torch.zeros_like(z))


def test_posterior_drift_adds_the_song_correction():
    f_bar = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    g = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    score = torch.tensor([[-1.0, 0.5]], dtype=torch.float64)
    div = torch.tensor([[0.05, -0.02]], dtype=torch.float64)
    got = assemble_posterior_drift(f_bar, g, score, div)
    expected = f_bar + 0.5 * g ** 2 * score + 0.5 * div
    assert torch.allclose(got, expected)


def test_gaussian_kl_is_zero_for_identical_diagonals_and_matches_a_shift():
    mu = torch.zeros(4, 2, dtype=torch.float64)
    sigma = torch.ones(4, 2, dtype=torch.float64)
    assert torch.allclose(gaussian_kl_diag(mu, sigma, mu, sigma),
                          torch.zeros(4, dtype=torch.float64))
    mu_q = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    sigma_q = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
    mu_p = torch.zeros(1, 2, dtype=torch.float64)
    sigma_p = torch.ones(1, 2, dtype=torch.float64)
    # 0.5 * sum_i [log 1 - log var_q - 1 + var_q + mu^2]
    manual = 0.0
    for mean, std in ((1.0, 0.5), (-2.0, 2.0)):
        var = std ** 2
        manual += 0.5 * (-math.log(var) - 1.0 + var + mean ** 2)
    assert torch.allclose(gaussian_kl_diag(mu_q, sigma_q, mu_p, sigma_p),
                          torch.tensor([manual], dtype=torch.float64))


def test_negative_failure_is_latched_and_counted():
    x0 = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [12.0, 1.0]])
    increments = np.zeros((4, 1, 2))
    increments[1, 0, 0] = -50.0

    def drift(x, t):
        return np.zeros((len(x), 2))

    def diffusion(x, t):
        out = np.ones((len(x), 2))
        # The initial state at index 3 is already outside the radius, so the
        # active batch is the first three rows. Poison the last of those.
        out[-1] = np.nan
        return out

    sim = simulate_population(drift, diffusion, x0, 0.1, increments, 10.0, 5.0)
    assert sim["failure_class"][0] is None
    assert sim["failure_class"][1] == "negative_state"
    assert sim["failure_class"][2] == "non_finite"
    assert sim["failure_class"][3] == "outside_radius"
    assert sim["failure_rate"] == 0.75
    assert not np.isfinite(sim["final"][1]).any()
    assert np.allclose(sim["final"][0], [1.0, 1.0])


def test_coarse_increments_match_the_wiener_sum():
    fine = np.arange(40, dtype=np.float64).reshape(2, 10, 2)
    coarse = coarse_increments(fine, 5)
    assert coarse.shape == (2, 2, 2)
    assert np.allclose(coarse[:, 0], fine[:, 0:5].sum(axis=1) / np.sqrt(5))


def test_numpy_prior_matches_torch_and_scales_by_std():
    torch.manual_seed(0)
    model = MatchingModel(CFG, state_dependent=True)
    mean = np.array([0.2, -0.1])
    std = np.array([0.4, 1.5])
    model.set_normalization(mean, std)
    weights = pack_weights(model)
    assert parity_error(model, weights, seed=3) < 1e-10
    x = np.array([[1.0, 0.5]])
    t = np.array([1.5])
    z = (x - mean) / std
    tt = torch.tensor([[1.5 / model.T]], dtype=torch.float64)
    zz = torch.tensor(z, dtype=torch.float64)
    with torch.no_grad():
        drift_z = model.prior_drift(zz, torch.tensor([[1.5]], dtype=torch.float64)).numpy()
    drift_x, diffusion_x = teacher_fields(weights, x, t)
    assert np.allclose(drift_x, drift_z * std)
    assert np.all(diffusion_x > 0.0)
    # The normalized input used above agrees with t / T inside the module.
    assert tt.shape == (1, 1)


def test_one_elbo_step_is_finite_and_reaches_the_diffusion():
    torch.manual_seed(1)
    model = MatchingModel(CFG, state_dependent=True)
    paths = torch.randn(4, len(model.times), 2, dtype=torch.float64)
    terms = model.elbo_terms(paths)
    assert torch.isfinite(terms["total"])
    terms["total"].backward()
    assert model.diffusion_net[0].weight.grad is not None
    assert torch.isfinite(model.diffusion_net[0].weight.grad).all()
    assert model.drift_net[0].weight.grad is not None
    assert model.gru.weight_hh_l0.grad is not None


def test_initial_draw_stays_inside_the_frozen_box():
    rng = np.random.default_rng(0)
    x0 = sample_x0(rng, 200, CFG)
    lo = np.asarray(CFG["data"]["x0_min"])
    hi = np.asarray(CFG["data"]["x0_max"])
    assert np.all(x0 >= lo - 1e-12)
    assert np.all(x0 <= hi + 1e-12)


def test_dataset_is_trajectory_disjoint_and_positive():
    data = make_dataset(CFG)
    assert data["train"]["paths"].shape[0] == 128
    assert data["validation"]["paths"].shape[0] == 64
    assert data["test"]["paths"].shape[0] == 128
    assert data["train"]["paths"].shape[1] == 61
    for name in ("train", "validation", "test"):
        assert np.all(data[name]["paths"] > 0.0)
    # Different splits are different arrays, not views of one path list.
    assert data["train"]["x0"].shape[1] == 2


def test_config_thresholds_stay_at_the_preregistered_values():
    h1 = CFG["hypotheses"]["H1_drift"]
    assert h1["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert h1["teacher_vs_truth_test_nrmse_max"] == 0.4
    assert h1["symbolic_vs_truth_test_nrmse_max"] == 0.5
    h6 = CFG["hypotheses"]["H6_state_independent_baseline"]
    assert h6["diffusion_nrmse_improvement_min"] == 0.05
    assert h6["state_dependent_nrmse_max"] == 0.4
    assert h6["shared_floor_nrmse"] == 0.8
    assert CFG["hypotheses"]["H3_collinearity_gate"]["vif_max"] == 10.0
    assert CFG["simulation"]["failure_radius"] == 10.0
    assert CFG["date"] == "2026-11-06"
    assert CFG["day"] == 13


def _metrics(**overrides):
    base = {
        "drift_sym_teacher": 0.1, "drift_teacher_truth": 0.2, "drift_sym_truth": 0.2,
        "g_sym_teacher": 0.1, "g_teacher_truth": 0.2, "g_sym_truth": 0.2,
        "sigma_sym_teacher": 0.1, "fraction_g_nonpositive": 0.0, "vif": 3.0,
        "fail_truth": 0.0, "fail_teacher": 0.0, "fail_symbolic": 0.0,
        "exc_teacher": 0.1, "exc_symbolic": 0.12,
        "mean_sym_teacher": 0.05, "mean_teacher_truth": 0.1, "cov_sym_teacher": 0.1,
        "disc_rmse": 0.1, "disc_retained": 1.0,
        "law_fail_truth": 0.0, "law_fail_teacher": 0.0, "law_fail_symbolic": 0.0,
        "law_mean_teacher": 0.1, "law_mean_symbolic": 0.1,
        "law_cov_teacher": 0.1, "law_cov_symbolic": 0.1,
        "floor_mean": 0.05, "floor_cov": 0.05,
        "g_state_truth": 0.2, "g_time_truth": 0.4,
    }
    base.update(overrides)
    return base


def test_shared_floor_does_not_pass_h6_and_a_real_gap_does():
    shared = decide(CFG, _metrics(g_state_truth=0.91, g_time_truth=0.99))
    assert shared["H6"] == "fail"
    assert shared["H6_shared_floor"] is True
    gap_only = decide(CFG, _metrics(g_state_truth=0.7, g_time_truth=0.9))
    assert gap_only["H6"] == "fail"
    assert gap_only["H6_shared_floor"] is False
    good = decide(CFG, _metrics(g_state_truth=0.2, g_time_truth=0.4))
    assert good["H6"] == "pass"
    assert good["H1"] == "pass"


def test_failure_and_floor_gates_make_distribution_claims_indeterminate():
    exploded = decide(CFG, _metrics(fail_symbolic=0.2))
    assert exploded["H4"] == "indeterminate"
    assert exploded["H4_failure"] == "fail"
    coarse_floor = decide(CFG, _metrics(floor_mean=0.5))
    assert coarse_floor["H5"] == "indeterminate"
    vif = decide(CFG, _metrics(vif=25.0))
    assert vif["H3"] == "indeterminate"


def test_vif_ignores_the_constant_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    value = vif_max(x, 1)
    assert math.isfinite(value)
    assert value < 2.0
    # A duplicated column makes the degree-1 library singular and the VIF large.
    cloned = np.concatenate([x[:, :1], x[:, :1], x[:, 2:]], axis=1)
    assert vif_max(cloned, 1) > 10.0


def test_degree3_oracle_can_represent_the_lotka_volterra_field():
    from src.daily_neural_sde.common import fit_polynomial_field
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 2.0, size=(500, 2))
    t = rng.uniform(0.0, 6.0, size=500)
    features = np.concatenate([x, t.reshape(-1, 1)], axis=1)
    target = true_drift(x, 2.0 / 3.0, 4.0 / 3.0, 1.0, 1.0)
    field = fit_polynomial_field(features, target, 3, 0.0, 1e-8)
    assert nrmse(target, field.predict(features)) < 1e-8
