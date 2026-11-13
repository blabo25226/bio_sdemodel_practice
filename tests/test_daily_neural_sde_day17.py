"""Day 17 tests: Du et al. (UAI 2026) neural diffusion intensity utilities.

These tests do not train the full teacher. They check the CIR data law, the
ELBO rollout, failure accounting, the power-library ceiling for sqrt(z), and
the frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import fit_polynomial_field, nrmse
from src.daily_neural_sde.day17_du_ndm import (
    build_model, decide, drift_numpy, gaussian_nll, load_config, make_dataset,
    paper_sigma_fn, remaining_counts, rollout, select_field, simulate_scalar,
    time_sensitivity, true_drift, true_sigma, vif_max)

CFG = load_config()


def _passing_metrics() -> dict:
    drift = {"sym_teacher_nrmse": 0.05, "teacher_truth_nrmse": 0.10, "sym_truth_nrmse": 0.12}
    return {
        "drift_test": drift,
        "sigma_test_nrmse": 0.01,
        "sigma_nonpositive_fraction": 0.0,
        "sigma_low_z_nrmse": 0.30,
        "vif_selected": 4.0,
        "fail_truth": 0.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "fail_poly": 0.0,
        "paired_sym_teacher": 1.0,
        "paired_teacher_truth": 1.5,
        "paired_poly_teacher": 1.2,
        "var_sym_teacher": 0.1,
        "var_teacher_truth": 0.1,
        "var_poly_teacher": 0.1,
        "disc_rmse": 0.2,
        "disc_retained": 1.0,
        "law_fail_truth": 0.0,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "floor_mean": 0.9,
        "floor_var": 0.08,
        "law_mean_teacher": 1.0,
        "law_mean_symbolic": 1.2,
        "law_var_teacher": 0.1,
        "law_var_symbolic": 0.1,
        "time_teacher": 0.02,
        "time_symbolic": 0.02,
        "grid_teacher_truth_nrmse": 0.2,
        "grid_sym_teacher_nrmse": 0.1,
        "post_fail": 0.0,
        "prior_fail": 0.0,
        "post_rmse": 3.0,
        "prior_rmse": 6.0,
        "step_ratio": 0.55,
        "nll_teacher": 1.0,
        "nll_truth": 0.99,
    }


def test_frozen_thresholds_match_the_pre_registration():
    assert CFG["paper"]["venue"] == "UAI"
    assert CFG["paper"]["year"] == 2026
    assert CFG["data"]["kappa"] == 0.3
    assert CFG["data"]["theta"] == 80.0
    assert CFG["data"]["sigma_scale"] == 1.0
    assert CFG["data"]["z0"] == 5.0
    assert CFG["hypotheses"]["H1_drift"]["teacher_vs_truth_test_nrmse_max"] == 0.30
    assert CFG["hypotheses"]["H1_drift"]["symbolic_vs_teacher_test_nrmse_max"] == 0.20
    assert CFG["hypotheses"]["H2_diffusion_probe"]["symbolic_vs_sigma_test_nrmse_max"] == 0.05
    assert CFG["interpretation"]["coefficients_are_closed_form"] is False
    assert CFG["interpretation"]["diffusion_teacher_is_neural"] is False
    # Feller's condition for this CIR diffusion: 2 kappa theta > sigma^2.
    assert 2 * CFG["data"]["kappa"] * CFG["data"]["theta"] > CFG["data"]["sigma_scale"] ** 2


def test_cir_drift_and_one_step_mean():
    state = np.array([5.0, 80.0, 20.0])
    drift = true_drift(state, kappa=0.3, theta=80.0)
    assert drift.shape == (3,)
    assert np.allclose(drift, [22.5, 0.0, 18.0])
    sigma = true_sigma(np.array([4.0, 9.0]))
    assert np.allclose(sigma, [2.0, 3.0])
    rng = np.random.default_rng(0)
    z0 = np.full(20000, 5.0)
    dt = 0.04
    shock = rng.normal(0.0, 1.0, size=len(z0))
    nxt = z0 + true_drift(z0) * dt + true_sigma(z0) * math.sqrt(dt) * shock
    assert abs(float(nxt.mean()) - (5.0 + 22.5 * dt)) < 0.02


def test_dataset_is_reproducible_and_stays_above_the_floor():
    first = make_dataset(CFG)
    second = make_dataset(CFG)
    assert first["train"]["left"].shape == (256, 100)
    assert first["val"]["left"].shape == (64, 100)
    assert first["test"]["counts"].shape == (64, 100)
    assert np.array_equal(first["train"]["counts"], second["train"]["counts"])
    assert np.array_equal(first["test"]["left"], second["test"]["left"])
    assert float(np.min(first["train"]["left"])) > CFG["data"]["failure_floor"]
    remain = remaining_counts(first["train"]["counts"])
    assert remain.shape == first["train"]["counts"].shape
    assert np.all(remain[:, 0] == first["train"]["counts"].sum(axis=1))
    assert np.all(np.diff(remain, axis=1) <= 0)


def test_power_library_recovers_the_linear_drift_and_not_the_square_root_tail():
    z_axis = np.linspace(8.0, 70.0, 30)
    t_axis = np.linspace(0.0, 4.0, 20)
    zz, tt = np.meshgrid(z_axis, t_axis, indexing="ij")
    state = np.stack([zz.ravel(), tt.ravel()], axis=1)
    target = true_drift(state[:, 0])
    field, search = select_field(state[:400], target[:400], state[400:], target[400:], CFG)
    assert search["selected"]["degree"] == 1
    assert field.predict(state).shape == (len(state), 1)
    assert nrmse(target, field.predict(state)[:, 0]) < 1e-8
    cloud = np.linspace(8.0, 70.0, 200)
    sigma_field = fit_polynomial_field(cloud.reshape(-1, 1), np.sqrt(cloud).reshape(-1, 1), 3, 0.0)
    low = np.linspace(0.5, 4.0, 40)
    assert nrmse(np.sqrt(cloud), sigma_field.predict(cloud.reshape(-1, 1))[:, 0]) < 0.02
    assert nrmse(np.sqrt(low), sigma_field.predict(low.reshape(-1, 1))[:, 0]) > 0.20


def test_true_drift_is_insensitive_to_time_and_vif_is_finite():
    state = np.stack([np.linspace(10.0, 50.0, 25), np.linspace(0.0, 4.0, 25)], axis=1)
    shuffled = np.linspace(4.0, 0.0, 25)
    sensitivity = time_sensitivity(lambda rows: true_drift(rows[:, 0]), state, shuffled)
    assert sensitivity < 1e-12
    box_z = np.linspace(8.0, 70.0, 40)
    box_t = np.linspace(0.0, 4.0, 20)
    zz, tt = np.meshgrid(box_z, box_t, indexing="ij")
    vif = vif_max(np.stack([zz.ravel(), tt.ravel()], axis=1), 1)
    assert 1.0 <= vif < 20.0


def test_failure_latch_keeps_the_failed_mass():
    increments = np.zeros((3, 6))
    out = simulate_scalar(lambda z, _t: np.full(len(z), -100.0),
                          lambda z: np.ones(len(z)),
                          np.full(3, 5.0), 0.04, increments, 0.2, 250.0)
    assert out["states"].shape == (3, 7)
    assert out["failure_rate"] == 1.0
    assert out["failure_class"] == ["below_floor", "below_floor", "below_floor"]
    assert np.allclose(out["states"][:, 0], 5.0)
    assert np.isnan(out["states"][:, -1]).all()
    again = simulate_scalar(lambda z, _t: np.zeros(len(z)), lambda z: np.zeros(len(z)),
                            np.full(2, 5.0), 0.04, np.zeros((2, 4)), 0.2, 250.0)
    assert again["failure_rate"] == 0.0
    assert np.allclose(again["states"], 5.0)


def test_rollout_shapes_and_backward_are_finite():
    model = build_model(CFG)
    model.set_scalers(np.array([20.0, 2.0]), np.array([15.0, 1.0]),
                      np.array([20.0, 2.0, 30.0]), np.array([15.0, 1.0, 20.0]))
    counts = np.array([[1, 0, 2, 1], [0, 1, 0, 3]], dtype=np.int64)
    noise = torch.randn(2, 3, 4, dtype=torch.float64)
    out = rollout(model, np.array([5.0, 5.0]), counts, np.arange(4) * 0.04, 0.04, noise,
                  project=True, drift_source="network", corr_source="network",
                  kappa=0.3, theta=80.0, sigma_scale=1.0)
    assert out["elbo"].shape == (2, 3)
    assert out["left"].shape == (2, 3, 4)
    loss = -out["elbo"].mean()
    loss.backward()
    gradient = model.drift_net.layers[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    state = np.array([[5.0, 0.0], [12.0, 1.0], [30.0, 3.0]])
    numpy_drift = drift_numpy(model, state)
    torch_drift = model.drift(torch.tensor(state[:, 0], dtype=torch.float64),
                              torch.tensor(state[:, 1], dtype=torch.float64))
    assert numpy_drift.shape == (3,)
    assert np.allclose(numpy_drift, torch_drift.detach().numpy())
    sigma = paper_sigma_fn(CFG)(np.array([4.0, 9.0]))
    assert np.allclose(sigma, [2.0, 3.0])


def test_girsanov_kernel_enters_as_sigma_times_u():
    model = build_model(CFG)
    model.set_scalers(np.zeros(2), np.ones(2), np.zeros(3), np.ones(3))
    with torch.no_grad():
        for parameter in model.corr_net.parameters():
            parameter.zero_()
        model.corr_net.layers[-1].bias.fill_(0.1)
    counts = np.zeros((1, 2), dtype=np.int64)
    noise = torch.zeros(1, 1, 2, dtype=torch.float64)
    out = rollout(model, np.array([5.0]), counts, np.array([0.0, 0.04]), 0.04, noise,
                  project=True, drift_source="truth", corr_source="network",
                  kappa=0.3, theta=80.0, sigma_scale=1.0)
    # u = sqrt(z) * 0.1 and the state update adds sigma * u = z * 0.1.
    expected = 5.0 + (22.5 + 0.5) * 0.04
    assert abs(float(out["left"][0, 0, 1].detach()) - expected) < 1e-8


def test_biased_drift_has_a_worse_one_step_nll():
    rng = np.random.default_rng(1)
    z = rng.uniform(10.0, 60.0, size=4000)
    dt = 0.04
    drift = true_drift(z)
    sigma = true_sigma(z)
    increment = drift * dt + sigma * math.sqrt(dt) * rng.normal(0.0, 1.0, size=len(z))
    truth = float(np.mean(gaussian_nll(increment, drift, sigma, dt)))
    biased = float(np.mean(gaussian_nll(increment, drift + 8.0, sigma, dt)))
    assert biased > truth + 0.01


def test_decide_passes_a_clean_case_and_blocks_failure_mass_and_a_weak_posterior():
    clean = decide(CFG, _passing_metrics())
    assert clean["H1"] == "pass"
    assert clean["H2"] == "pass"
    assert clean["H2_is_square_root"] is False
    assert clean["H2_local_shadow"] is True
    assert clean["H3"] == "pass"
    assert clean["H4"] == "pass"
    assert clean["H5"] == "pass"
    assert clean["H6"] == "pass"
    assert clean["H8"] == "pass"
    assert clean["coefficients_are_closed_form"] is False
    assert clean["nll_hides_drift"] is False
    failed = _passing_metrics()
    failed["fail_teacher"] = 0.2
    blocked = decide(CFG, failed)
    assert blocked["H4"] == "indeterminate"
    assert blocked["H4b"] == "indeterminate"
    weak = _passing_metrics()
    weak["post_rmse"] = 5.5
    weak["prior_rmse"] = 6.0
    assert decide(CFG, weak)["H8"] == "fail"
    collinear = _passing_metrics()
    collinear["vif_selected"] = 40.0
    assert decide(CFG, collinear)["H3"] == "indeterminate"
    hidden = _passing_metrics()
    hidden["drift_test"] = {"sym_teacher_nrmse": 0.05, "teacher_truth_nrmse": 0.50,
                            "sym_truth_nrmse": 0.50}
    hidden["nll_teacher"] = 1.0
    hidden["nll_truth"] = 0.99
    hidden["step_ratio"] = 0.55
    verdict = decide(CFG, hidden)
    assert verdict["H1"] == "fail"
    assert verdict["nll_hides_drift"] is True
