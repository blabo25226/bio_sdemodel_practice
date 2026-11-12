"""Day 20 tests: Howard et al. (ICML 2026) bridge utilities.

These tests do not train the full teacher. They check the reference drift,
the gradient constraint, the terminal jump, failure accounting, and the
frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import nrmse
from src.daily_neural_sde.day20_howard_ccdb import (
    PotentialNet, consistency_loss, decide, finite_difference_curl, load_config,
    query_eta_field, reference_drift, simulate_bridge, teacher_eta_np, terminal_control,
    vif_max)

CFG = load_config()


def _pack() -> dict:
    return {"sym_teacher_nrmse": 0.1, "constant_floor": 0.4, "sym_teacher_rms": 0.05,
            "teacher_rms": 1.0}


def _metrics(**overrides) -> dict:
    base = {
        "eta_test": [_pack(), _pack()],
        "matches_ceiling": False,
        "diffusion_nrmse": 0.0,
        "diffusion_nonpositive_fraction": 0.0,
        "vif_degree3": 4.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "path_rmse": 0.05,
        "prejump_rmse": 0.08,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "floor_mean": 0.02,
        "law_mean_symbolic": 0.04,
        "grid_nrmse": 0.2,
        "time_teacher": 0.3,
        "time_symbolic": 0.28,
        "grid_residual_rms": 0.1,
        "path_residual_rms": 0.05,
        "eta_rms": 1.0,
        "baseline_path_rmse": 0.1,
        "pure_nrmse": 0.8,
        "structured_nrmse": 0.1,
        "teacher_curl_rms": 1e-6,
        "symbolic_relative_curl": 0.05,
    }
    base.update(overrides)
    return base


def test_frozen_thresholds_match_the_pre_registration():
    assert CFG["paper"]["venue"] == "ICML"
    assert CFG["paper"]["year"] == 2026
    assert CFG["data"]["horizon"] == 4.0
    assert CFG["data"]["sigma"] == 0.1
    assert CFG["data"]["x0"] == [0.1, -0.1]
    assert CFG["data"]["x_terminal"] == [2.0, -0.1]
    assert CFG["data"]["hill_eps"] == 0.0625
    assert CFG["model"]["self_consistency"] == "4.1"
    assert CFG["model"]["alpha_schedule"] == "average"
    assert CFG["model"]["n_steps"] == 40
    assert CFG["model"]["train_steps"] == 180
    assert CFG["model"]["grad_clip"] == 100.0
    assert CFG["model"]["head_initialized_at_zero"] is True
    assert CFG["hypotheses"]["H1_eta"]["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert CFG["hypotheses"]["H3_collinearity_gate"]["vif_max"] == 10.0
    assert CFG["hypotheses"]["H8_pure_polynomial"]["gap_min"] == 0.10
    assert CFG["interpretation"]["diffusion_teacher_is_neural"] is False
    assert CFG["interpretation"]["drift_teacher_is_neural"] is True
    assert CFG["interpretation"]["coefficients_are_closed_form"] is False
    assert CFG["interpretation"]["distillation_discovers_true_sde"] is False
    assert len({CFG["seeds"]["law_a"], CFG["seeds"]["law_b"], CFG["seeds"]["law_symbolic"]}) == 3


def test_reference_drift_matches_the_hill_field_at_two_points():
    values = reference_drift(np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(values[0], [1.0, 1.0])
    np.testing.assert_allclose(values[1], [0.0, 0.0], atol=1e-12)
    # The fourth-power Hill terms are even, so the drift at -1 is not the negative of the drift at 1.
    left = reference_drift(np.array([[-1.0, 0.5]]))
    assert left.shape == (1, 2)
    assert np.isfinite(left).all()


def test_terminal_jump_control_cancels_the_bridge_when_eta_matches_minus_b_over_sigma():
    x = np.array([[0.4, -0.2]])
    dt = 0.1
    sigma = 0.1
    terminal = np.array([2.0, -0.1])
    control = terminal_control(x, terminal, dt, sigma)
    drift = reference_drift(x)
    jump = (terminal - x) / dt
    np.testing.assert_allclose(control, (jump - drift) / sigma ** 2)
    # eta = -b / sigma makes the parameterized drift equal to the jump drift.
    eta = -drift / sigma
    remain = dt
    parameterized = drift + (terminal - x) / remain + sigma * eta
    np.testing.assert_allclose(parameterized, jump)


def test_potential_gradient_has_the_state_shape_and_a_negligible_curl():
    torch.manual_seed(0)
    net = PotentialNet(8, 1, 2).to(dtype=torch.float64)
    # A nonzero head so the gradient is not identically zero.
    last = net.net[-1]
    with torch.no_grad():
        last.weight.normal_(0.0, 0.3)
        last.bias.fill_(0.2)
    x = np.array([[0.2, -0.4], [1.1, 0.3], [1.8, -0.7]])
    eta = teacher_eta_np(net, 1.5, x, 4.0)
    assert eta.shape == (3, 2)
    assert np.isfinite(eta).all()
    features = np.column_stack([np.full(3, 1.5), x])

    def fn(points: np.ndarray) -> np.ndarray:
        out = np.zeros((len(points), 2))
        for t in np.unique(points[:, 0]):
            mask = np.isclose(points[:, 0], t)
            out[mask] = teacher_eta_np(net, float(t), points[mask, 1:], 4.0)
        return out

    curl = finite_difference_curl(fn, features, 1e-5)
    assert rms_ok(curl)


def rms_ok(curl: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(curl ** 2))) < 1e-4


def test_zero_head_eta_is_zero_and_one_loss_step_is_finite():
    net = PotentialNet(8, 1, 2).to(dtype=torch.float64)
    eta = teacher_eta_np(net, 0.0, np.zeros((4, 2)), 4.0)
    np.testing.assert_allclose(eta, 0.0)
    noise = torch.zeros(2, 3, 2, dtype=torch.float64)
    loss, n_alive = consistency_loss(
        net, noise, torch.tensor([0.1, -0.1], dtype=torch.float64),
        torch.tensor([2.0, -0.1], dtype=torch.float64), 4.0, 0.1, 8.0)
    assert n_alive == 2
    assert torch.isfinite(loss)
    loss.backward()
    grads = [param.grad for param in net.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_simulation_jump_is_exact_and_negative_diffusion_is_counted():
    def zero_eta(t: float, x: np.ndarray) -> np.ndarray:
        del t
        return np.zeros((len(x), 2))

    def positive(t: float, x: np.ndarray) -> np.ndarray:
        del t
        return np.full(len(x), 0.1)

    def negative(t: float, x: np.ndarray) -> np.ndarray:
        del t
        return np.full(len(x), -0.1)

    increments = np.zeros((3, 3, 2))
    out = simulate_bridge(zero_eta, positive, np.array([0.1, -0.1]), np.array([2.0, -0.1]),
                          4.0, 0.1, 4, increments, 8.0, True)
    assert out["failure_rate"] == 0.0
    np.testing.assert_allclose(out["states"][:, -1], np.tile([2.0, -0.1], (3, 1)))
    again = simulate_bridge(zero_eta, positive, np.array([0.1, -0.1]), np.array([2.0, -0.1]),
                            4.0, 0.1, 4, increments, 8.0, True)
    np.testing.assert_allclose(out["states"], again["states"])
    bad = simulate_bridge(zero_eta, negative, np.array([0.1, -0.1]), np.array([2.0, -0.1]),
                          4.0, 0.1, 4, increments, 8.0, True)
    assert bad["failure_rate"] == 1.0
    assert bad["failure_class"].count("nonpositive_diffusion") == 3
    assert np.isnan(bad["states"][:, 1]).all()


def test_vif_ignores_the_constant_column():
    rng = np.random.default_rng(4)
    features = np.column_stack([
        np.linspace(0.0, 3.9, 40),
        rng.normal(size=40),
        rng.normal(size=40),
    ])
    value = vif_max(features, 1)
    assert math.isfinite(value)
    assert value < 10.0


def test_decide_pass_fail_and_indeterminate_failure_mass():
    assert decide(CFG, _metrics())["H1"] == "pass"
    assert decide(CFG, _metrics())["H3_closed_form"] is False
    assert decide(CFG, _metrics())["H8"] == "pass"
    missed = _metrics()
    missed["eta_test"] = [_pack(), dict(_pack(), sym_teacher_nrmse=0.3)]
    assert decide(CFG, missed)["H1"] == "fail"
    constant = _metrics()
    constant["eta_test"] = [dict(_pack(), sym_teacher_nrmse=0.2, constant_floor=0.15), _pack()]
    assert decide(CFG, constant)["H1"] == "fail"
    assert decide(CFG, _metrics(vif_degree3=12.0))["H3"] == "indeterminate"
    assert decide(CFG, _metrics(fail_teacher=0.2, fail_symbolic=0.2))["H4"] == "indeterminate"
    assert decide(CFG, _metrics(floor_mean=0.4))["H5"] == "indeterminate"
    assert decide(CFG, _metrics(path_rmse=0.4))["H4"] == "fail"
    assert decide(CFG, _metrics(grid_nrmse=0.9))["H6"] == "fail"
    assert decide(CFG, _metrics(eta_rms=0.1))["H7"] == "fail"
    queried = query_eta_field(PotentialNet(4, 1, 1).to(dtype=torch.float64),
                              np.array([[0.0, 0.1, -0.1], [1.0, 0.2, 0.0]]), 4.0)
    assert queried.shape == (2, 2)
    assert nrmse(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
