"""Day 12 tests: Zhang et al. (ICLR 2025) FDM utilities.

These tests do not train the teacher. They check the truth fields, the
kernel loss direction, failure accounting, the normalized-coordinate
conversion, and the frozen hypothesis thresholds.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.daily_neural_sde.day12_zhang_fdm import (
    FieldNet, NeuralSDE, coarse_increments, export_layers, fdm_loss_numpy,
    init_sde, load_config, make_dataset, mlp_numpy, simulate_timed,
    stable_softplus, teacher_field, to_original, true_diffusion, true_drift,
    vif_max)
from src.daily_neural_sde.common import fit_polynomial_field, nrmse

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "daily_neural_sde_day12.json")


def test_true_fields_match_the_closed_form():
    x = np.array([[0.0, 1.0], [2.0, -1.0]])
    drift = true_drift(x, mu=1.0)
    diffusion = true_diffusion(x, g0=0.25, g2=0.05, gy=0.3)
    assert drift.shape == (2, 2)
    assert diffusion.shape == (2, 2)
    # (0, 1): dy = (1 - 0) * 1 - 0 = 1. (2, -1): dy = (1 - 4) * (-1) - 2 = 1.
    assert np.allclose(drift[0], [1.0, 1.0])
    assert np.allclose(drift[1], [-1.0, 1.0])
    assert np.allclose(diffusion[1], [0.25 + 0.05 * 4.0, 0.3])


def test_fdm_loss_is_smaller_when_paths_match():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(48, 12, 2))
    matched = fdm_loss_numpy(data, data, sigma=1.0)
    shifted = fdm_loss_numpy(data + 3.0, data, sigma=1.0)
    assert matched < shifted


def test_failure_mass_is_retained_and_classified():
    x0 = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 0.0]])
    increments = np.zeros((3, 1, 2))
    increments[2, 0, 0] = 1.0

    def drift(x, t):
        return np.zeros((len(x), 2))

    def diffusion(x, t):
        out = np.zeros((len(x), 2))
        # The initial state at norm 30 is already failed, so the active
        # batch is the two finite rows. Poison the last of those.
        out[-1] = np.nan
        return out

    sim = simulate_timed(drift, diffusion, x0, 0.1, increments, 12.0, 3.5)
    assert list(sim["failed"]) == [False, True, True]
    assert sim["failure_class"][1] == "outside_radius"
    assert sim["failure_class"][2] == "non_finite"
    assert sim["failure_rate"] == 2.0 / 3.0
    assert not np.isfinite(sim["final"][1]).any()
    assert not np.isfinite(sim["final"][2]).any()
    assert np.allclose(sim["final"][0], 0.0)


def test_zero_field_is_constant_and_reproducible():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(6, 2))
    increments = rng.normal(size=(6, 5, 2))

    def drift(x, t):
        return np.zeros((len(x), 2))

    def diffusion(x, t):
        return np.zeros((len(x), 2))

    first = simulate_timed(drift, diffusion, x0, 0.05, increments, 12.0, 3.5)
    second = simulate_timed(drift, diffusion, x0, 0.05, increments, 12.0, 3.5)
    assert np.allclose(first["states"], x0[:, None, :])
    assert np.allclose(first["states"], second["states"])
    assert first["failure_rate"] == 0.0
    assert first["excursion_rate"] == 0.0


def test_coarse_increments_match_the_wiener_sum():
    fine = np.arange(40, dtype=np.float64).reshape(2, 10, 2)
    coarse = coarse_increments(fine, 5)
    assert coarse.shape == (2, 2, 2)
    assert np.allclose(coarse[:, 0], fine[:, 0:5].sum(axis=1) / np.sqrt(5))


def test_linear_coordinate_change_has_no_ito_correction():
    mean = np.array([0.2, -0.4])
    std = np.array([2.0, 0.5])
    x = np.array([[1.0, 0.0]])
    z = (x - mean) / std
    f_z = np.array([[0.3, -0.2]])
    g_z = np.array([[0.4, 0.1]])
    dt = 0.05
    increment = np.array([0.5, -0.25])
    z1 = z + f_z * dt + g_z * np.sqrt(dt) * increment
    f_x, g_x = to_original(f_z, g_z, std)
    x1 = x + f_x * dt + g_x * np.sqrt(dt) * increment
    assert np.allclose(x1, mean + std * z1)


def test_numpy_mlp_matches_torch_softplus():
    torch.manual_seed(0)
    net = FieldNet(4, positive=True).double()
    layers = []
    for layer in (net.fc1, net.fc2, net.fc3):
        layers.append({"weight": layer.weight.detach().tolist(),
                       "bias": layer.bias.detach().tolist()})
    features = np.array([[0.2, -0.4, 0.5], [1.0, 0.0, 0.0]])
    with torch.no_grad():
        expected = net(torch.tensor(features[:, :2]), torch.tensor(features[:, 2:]))
    assert np.allclose(mlp_numpy(features, layers, True), expected.numpy(), atol=1e-12)
    assert np.allclose(stable_softplus(np.array([-40.0, 0.0, 40.0])),
                       torch.nn.functional.softplus(torch.tensor([-40.0, 0.0, 40.0])).numpy())


def test_degree3_library_recovers_the_vandepol_drift():
    rng = np.random.default_rng(2)
    xy = rng.normal(size=(400, 2))
    t = rng.random(400)
    features = np.concatenate([xy, t.reshape(-1, 1)], axis=1)
    target = true_drift(xy, mu=1.0)
    field = fit_polynomial_field(features, target, degree=3, threshold=0.0, ridge=1e-8)
    hold_xy = rng.normal(size=(200, 2))
    hold = np.concatenate([hold_xy, rng.random(200).reshape(-1, 1)], axis=1)
    assert nrmse(true_drift(hold_xy, mu=1.0), field.predict(hold)) < 1e-6


def test_splits_are_trajectory_disjoint():
    data = make_dataset(CFG)
    assert data["train"]["paths"].shape == (256, 31, 2)
    assert data["validation"]["paths"].shape == (128, 31, 2)
    assert data["test"]["paths"].shape == (256, 31, 2)
    assert data["n_steps"] == 30

    def keys(x0):
        return set(map(tuple, np.round(x0, 8)))

    assert keys(data["train"]["x0"]).isdisjoint(keys(data["test"]["x0"]))
    assert keys(data["train"]["x0"]).isdisjoint(keys(data["validation"]["x0"]))
    assert np.isfinite(data["test"]["paths"]).all()


def test_vif_on_random_states_is_finite():
    rng = np.random.default_rng(3)
    value = vif_max(rng.normal(size=(500, 3)), degree=3)
    assert np.isfinite(value)
    assert value < 1e6


def test_teacher_field_accepts_a_scalar_time():
    torch.manual_seed(4)
    model = NeuralSDE(4).double()
    init_sde(model)
    weights = {"drift": export_layers(model.drift),
               "diffusion": export_layers(model.diffusion),
               "mean": [0.0, 0.0], "std": [1.0, 1.0], "T": 1.5}
    xy = np.array([[0.2, -0.3], [1.0, 0.5]])
    scalar = teacher_field(weights, xy, 0.3, "drift")
    vector = teacher_field(weights, xy, np.array([0.3, 0.3]), "drift")
    assert scalar.shape == (2, 2)
    assert np.allclose(scalar, vector)


def test_config_thresholds_are_frozen():
    hyp = CFG["hypotheses"]
    assert hyp["H1_drift"]["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert hyp["H1_drift"]["teacher_vs_truth_test_nrmse_max"] == 0.4
    assert hyp["H2_diffusion"]["symbolic_vs_teacher_test_nrmse_max"] == 0.3
    assert hyp["H2c_range"]["fraction_symbolic_g_nonpositive_max"] == 0.02
    assert hyp["H3_collinearity_gate"]["vif_max"] == 10.0
    assert hyp["H4_simulation"]["failure_gate"] == 0.05
    assert hyp["H5_law"]["teacher_mmd2_max"] == 0.04
    assert hyp["H6_marginal_baseline"]["diffusion_nrmse_improvement_min"] == 0.05
    assert hyp["H7_discretization_gate"]["teacher_coarse_vs_fine_terminal_rmse_max"] == 0.5
    assert CFG["distillation"]["degrees"] == [1, 2, 3]
    assert CFG["teacher"]["n_opt_steps"] == 1200
    assert CFG["teacher"]["rbf_sigma"] == 1.0
    assert CFG["simulation"]["excursion_radius"] == 3.5
