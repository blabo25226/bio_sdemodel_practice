"""Day 11 tests: Kong et al. (2020) SDE-Net stepper (numpy only).

These tests do not train the teacher. They check that the diffusion
coefficient is held at the initial state, that failures and excursions
are counted separately, and that the frozen hypothesis thresholds are
the ones the run will read.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day11_kong_sdenet import (
    aggregate_increments, make_split, sample_xor, simulate_sdenet)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day11.json").read_text())


def _zero_drift(x, t):
    return np.zeros((len(x), 2))


def _zero_g(x):
    return np.zeros(len(x))


def test_diffusion_is_frozen_at_x0():
    x0 = np.array([[1.0, 0.0]])

    def drift(x, t):
        return np.ones((len(x), 2)) * np.array([1.0, 0.0])

    def g(x):
        # Equals the current x-coordinate. Re-evaluating after the first
        # step would use 1.7 instead of the initial 1.
        return np.asarray(x[:, 0], dtype=np.float64)

    increments = np.zeros((1, 2, 2))
    increments[0, :, 0] = 0.2
    sim = simulate_sdenet(drift, g, x0, 0.5, increments, 25.0, 8.0)
    # step 0: 1 + 0.5 + 1*0.2 = 1.7; step 1 with frozen g: 1.7 + 0.5 + 0.2 = 2.4
    assert np.isclose(sim["final"][0, 0], 2.4)
    assert sim["failure_rate"] == 0.0


def test_failure_and_excursion_are_separate():
    x0 = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 0.0]])
    increments = np.zeros((3, 1, 2))

    def g(x):
        out = np.zeros(len(x))
        out[2] = np.nan
        return out

    increments[2, 0, 0] = 1.0
    sim = simulate_sdenet(_zero_drift, g, x0, 0.1, increments, 25.0, 8.0)
    assert list(sim["failed"]) == [False, True, True]
    assert sim["failure_class"][1] == "outside_radius"
    assert sim["failure_class"][2] == "non_finite"
    # The finite-but-large row is an excursion; the origin is not. The
    # non-finite row is a failure and is not also forced into the excursion count.
    assert bool(sim["excursion"][1])
    assert not bool(sim["excursion"][0])
    assert not bool(sim["excursion"][2])
    assert sim["failure_rate"] == 2.0 / 3.0


def test_zero_field_is_constant():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(5, 2))
    increments = rng.normal(size=(5, 4, 2))
    sim = simulate_sdenet(_zero_drift, _zero_g, x0, 0.1, increments, 25.0, 8.0)
    assert np.allclose(sim["final"], x0)
    assert sim["failure_rate"] == 0.0
    assert sim["excursion_rate"] == 0.0


def test_aggregate_increments_sums_blocks():
    fine = np.arange(24, dtype=np.float64).reshape(2, 6, 2)
    coarse = aggregate_increments(fine, 3)
    assert coarse.shape == (2, 2, 2)
    assert np.allclose(coarse[:, 0], fine[:, 0:3].sum(axis=1))


def test_split_is_balanced_and_disjoint():
    cfg = CFG
    data = make_split(cfg)
    assert data["train"]["x0"].shape == (1200, 2)
    assert data["validation"]["x0"].shape == (400, 2)
    assert abs(data["train"]["y"].mean() - 0.5) < 1e-9
    # Trajectory identity: no shared initial condition across splits.
    def keys(x):
        return set(map(tuple, np.round(x, 8)))
    assert keys(data["train"]["x0"]).isdisjoint(keys(data["test"]["x0"]))
    assert keys(data["train"]["x0"]).isdisjoint(keys(data["validation"]["x0"]))


def test_sample_xor_modes():
    rng = np.random.default_rng(1)
    centers = np.asarray(CFG["data"]["centers"], dtype=np.float64)
    labels = np.asarray(CFG["data"]["center_labels"], dtype=int)
    x, y = sample_xor(rng, 400, 0.0, centers, labels)
    # With sigma 0 every point sits on a center, and same-sign centers are class 0.
    assert set(np.unique(y)).issubset({0, 1})
    same_sign = np.sign(x[:, 0] * x[:, 1]) > 0
    assert np.all(y[same_sign] == 0)
    assert np.all(y[~same_sign] == 1)


def test_config_thresholds_present():
    hyp = CFG["hypotheses"]
    assert hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"] == 0.3
    assert hyp["H2_diffusion_distillation"]["symbolic_g_ood_test_nrmse_vs_teacher_max"] == 0.5
    assert hyp["H2c_range"]["max"] == 0.05
    assert hyp["H4_simulation"]["teacher_accuracy_min"] == 0.85
    assert hyp["H5_paper_axis_id_ood_gap"]["failure_gate"] == 0.05
    assert CFG["distillation"]["degrees"] == [1, 2, 3]
    assert CFG["teacher"]["sigma_max"] == 1.0
