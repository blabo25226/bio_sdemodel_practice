"""Day 19 tests: Gao et al. (NeurIPS 2025) Neural MJD utilities.

These tests do not train the full teacher. They check the Merton step, the
truncated transition, the polynomial ceiling, failure accounting, and the
frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import nrmse
from src.daily_neural_sde.day19_gao_neural_mjd import (
    RateMLP, compensator, decide, design_matrix, fit_polynomial_field, future_times,
    generate_windows, jump_sum, load_config, make_noise, query_rates, repeat_targets,
    sample_parameters, simulate_forecast, simulate_log_paths, transition_log_prob,
    uniform_constant_floor, vif_max)

CFG = load_config()


def _metrics(**overrides) -> dict:
    pack = {"sym_teacher_nrmse": 0.1, "teacher_truth_nrmse": 0.2, "sym_truth_nrmse": 0.22,
            "sym_teacher_rms": 0.02, "teacher_truth_rms": 0.04, "teacher_rms": 0.3, "truth_rms": 0.32}
    base = {
        "mu_test": pack,
        "sigma_test": dict(pack),
        "sigma_test_nrmse": 0.1,
        "sigma_nonpositive_fraction": 0.0,
        "sigma_mean_relative_error": 0.05,
        "sigma_slice_nrmse": 0.2,
        "vif_degree3": 4.0,
        "shuffle_teacher": 0.05,
        "shuffle_symbolic": 0.08,
        "time_teacher": 0.04,
        "time_symbolic": 0.06,
        "slice_sym_teacher_nrmse": 0.2,
        "slice_sym_teacher_rms": 0.03,
        "fail_truth": 0.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "paired_sym_teacher": 0.02,
        "paired_teacher_truth": 0.03,
        "var_sym_teacher": 0.1,
        "var_teacher_truth": 0.1,
        "field_teacher_truth": 0.2,
        "disc_rmse": 0.05,
        "disc_retained": 1.0,
        "bs_sigma_nrmse": 0.4,
        "law_fail_truth": 0.0,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "floor_mean": 0.02,
        "floor_var": 0.05,
        "law_mean_teacher": 0.04,
        "law_mean_symbolic": 0.05,
        "law_var_teacher": 0.1,
        "law_var_symbolic": 0.12,
        "cross_ratio_teacher": 0.1,
    }
    base.update(overrides)
    return base


def test_frozen_thresholds_match_the_pre_registration():
    assert CFG["paper"]["venue"] == "NeurIPS"
    assert CFG["paper"]["year"] == 2025
    assert CFG["data"]["dt"] == 0.01
    assert CFG["data"]["context_prices"] == 10
    assert CFG["data"]["future_steps"] == 10
    assert CFG["model"]["kappa"] == 5
    assert CFG["model"]["omega"] == 1.0
    assert CFG["model"]["width"] == 64
    assert CFG["hypotheses"]["H1_fields"]["teacher_vs_truth_test_nrmse_max"] == 0.3
    assert CFG["hypotheses"]["H1_fields"]["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert CFG["hypotheses"]["H3_collinearity_gate"]["vif_max"] == 10.0
    assert CFG["hypotheses"]["H8_jump_versus_black_scholes"]["sigma_nrmse_improvement_min"] == 0.05
    assert CFG["threshold_basis"]["probe_seed"] == 19099
    assert CFG["seeds"]["law_noise"] == 19027
    assert len({CFG["seeds"]["law_truth"], CFG["seeds"]["law_noise"], CFG["seeds"]["law_floor"]}) == 3
    assert CFG["interpretation"]["diffusion_teacher_is_neural"] is True
    assert CFG["interpretation"]["coefficients_are_closed_form"] is False
    assert CFG["interpretation"]["distillation_discovers_true_sde"] is False
    floor = uniform_constant_floor(0.1, 0.5)
    assert abs(floor - CFG["threshold_basis"]["constant_floor_mu_sigma"]) < 1e-6
    assert floor > CFG["hypotheses"]["H1_fields"]["teacher_vs_truth_test_nrmse_max"]


def test_pure_drift_step_is_exact_and_jumps_match_the_compensator():
    dt = 0.01
    mu, sigma, lam, nu, gamma = 0.3, 0.0, 0.0, 0.0, 0.75
    params = {key: np.array([value]) for key, value in
              {"mu": mu, "sigma": sigma, "lam": lam, "nu": nu, "gamma": gamma}.items()}
    out = simulate_log_paths(params, 4, dt, np.random.default_rng(0), 4)
    np.testing.assert_allclose(out["log_s"][0, -1], mu * 4 * dt)
    assert compensator(0.0, 0.75) == math.exp(0.5 * 0.75 ** 2) - 1.0
    rng = np.random.default_rng(1)
    n = 4000
    sigma = np.full(n, 0.2)
    lam = np.full(n, 4.0)
    nu = np.full(n, 0.0)
    gamma = np.full(n, 0.3)
    mu = np.full(n, 0.4)
    z = rng.standard_normal(n)
    exp_unit = rng.exponential(1.0, size=(n, 6))
    gauss = rng.standard_normal((n, 6))
    jump, cap = jump_sum(exp_unit, gauss, lam, nu, gamma, dt)
    assert not cap.any()
    k = float(compensator(0.0, 0.3))
    delta = (mu - lam * k - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z + jump
    expected = (0.4 - 4.0 * k - 0.5 * 0.04) * dt
    assert abs(float(delta.mean()) - expected) < 0.002
    ratio = np.exp(delta)
    assert abs(float(ratio.mean()) - math.exp(0.4 * dt)) < 0.03


def test_truncated_density_matches_the_gaussian_when_there_are_no_jumps():
    log_cond = torch.zeros(3, dtype=torch.float64)
    log_next = torch.tensor([0.01, -0.02, 0.0], dtype=torch.float64)
    mu = torch.full((3,), 0.3, dtype=torch.float64)
    sigma = torch.full((3,), 0.2, dtype=torch.float64)
    lam = torch.zeros(3, dtype=torch.float64)
    nu = torch.zeros(3, dtype=torch.float64)
    gamma = torch.ones(3, dtype=torch.float64)
    dt = 0.01
    got = transition_log_prob(log_cond, log_next, mu, sigma, lam, nu, gamma, dt, 5)
    mean = (mu - 0.5 * sigma ** 2) * dt
    var = sigma ** 2 * dt
    expected = -0.5 * (math.log(2 * math.pi) + torch.log(var) + (log_next - mean) ** 2 / var)
    torch.testing.assert_close(got, expected, atol=1e-8, rtol=1e-8)


def test_degree3_library_recovers_a_constant_rate():
    rng = np.random.default_rng(19)
    returns = rng.normal(scale=0.2, size=(320, 9))
    train = design_matrix(returns[:280], 10)
    hold = design_matrix(returns[280:], 10)
    assert train.shape[0] > 286
    field = fit_polynomial_field(train, np.full((len(train), 1), 0.3), 3, 0.0, 0.0)
    assert nrmse(np.full(len(hold), 0.3), field.predict(hold)[:, 0]) < 1e-8
    assert abs(uniform_constant_floor(0.1, 0.5) - 0.359211) < 1e-5


def test_window_shapes_and_shared_noise_reproducibility():
    cfg = load_config()
    windows = generate_windows(cfg, np.random.default_rng(4), 6)
    assert windows["returns"].shape == (6, 9)
    assert windows["future"].shape == (6, 10)
    assert np.isfinite(windows["log_s"]).all()
    features = design_matrix(windows["returns"], 10)
    assert features.shape == (60, 10)
    assert np.allclose(features[:10, -1], future_times(10))
    noise = make_noise(6, 10, 4, np.random.default_rng(5))
    left = simulate_forecast(lambda _step: windows["params"], windows["present"], noise, 0.01, 12.0)
    right = simulate_forecast(lambda _step: windows["params"], windows["present"], noise, 0.01, 12.0)
    np.testing.assert_allclose(left["log_s"], right["log_s"])
    assert left["failure_rate"] == 0.0
    model = RateMLP(8, 1, True).to(dtype=torch.float64)
    rates = query_rates(model, features[:4])
    assert rates["sigma"].shape == (4,)
    assert np.all(rates["sigma"] > 0.0)
    assert np.all(rates["lam"] > 0.0)
    assert np.all(rates["gamma"] > 0.0)


def test_negative_diffusion_is_a_counted_failure_and_vif_ignores_the_constant():
    present = np.zeros(5)
    noise = make_noise(5, 2, 3, np.random.default_rng(6))
    bad = {"mu": np.zeros(5), "sigma": np.full(5, -0.1), "lam": np.ones(5),
           "nu": np.zeros(5), "gamma": np.ones(5)}
    out = simulate_forecast(lambda _step: bad, present, noise, 0.01, 12.0)
    assert out["failure_rate"] == 1.0
    assert out["failure_class"].count("nonpositive_coefficient") == 5
    assert np.isnan(out["log_s"][:, 1]).all()
    features = design_matrix(np.random.default_rng(8).normal(scale=0.2, size=(40, 9)), 4)
    value = vif_max(features, 1)
    assert math.isfinite(value)
    assert value < 10.0


def test_decide_pass_fail_and_indeterminate_failure_mass():
    assert decide(CFG, _metrics())["H1"] == "pass"
    assert decide(CFG, _metrics())["H8"] == "pass"
    failed = _metrics()
    failed["mu_test"] = dict(failed["mu_test"])
    failed["mu_test"]["teacher_truth_nrmse"] = 0.36
    failed["field_teacher_truth"] = 0.36
    verdict = decide(CFG, failed)
    assert verdict["H1_identify"] == "fail"
    assert verdict["H1"] == "fail"
    assert verdict["H4"] == "fail"
    dropped = _metrics(fail_teacher=0.2, fail_symbolic=0.2, fail_truth=0.2)
    assert decide(CFG, dropped)["H4"] == "indeterminate"
    assert decide(CFG, _metrics(floor_mean=0.5))["H5"] == "indeterminate"
    assert decide(CFG, _metrics(vif_degree3=12.0))["H3"] == "indeterminate"
    assert decide(CFG, _metrics())["H3_closed_form"] is False


def test_parameter_ranges_match_the_appendix():
    rng = np.random.default_rng(7)
    params = sample_parameters(2000, CFG, rng)
    assert params["mu"].min() >= 0.1 and params["mu"].max() <= 0.5
    assert params["lam"].min() >= 3.0 and params["lam"].max() <= 10.0
    assert params["nu"].min() >= -0.1 and params["nu"].max() <= 0.1
    assert abs(float(params["gamma"].mean()) - 0.75) < 0.02
    repeated = repeat_targets(params["mu"][:3], 4)
    assert repeated.shape == (12,)
    assert np.allclose(repeated[::4], params["mu"][:3])
