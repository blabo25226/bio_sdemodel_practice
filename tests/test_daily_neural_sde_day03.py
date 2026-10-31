"""Tests for the Day 3 (Tzen & Raginsky 2019 sigmoid) pipeline."""
from __future__ import annotations

import numpy as np

from src.daily_neural_sde.common import (expand_to_original,
                                         fit_polynomial_field, nrmse,
                                         simulate_em, tex_equations)
from src.daily_neural_sde.day03_tzen_sigmoid import (regime_of, sigmoid,
                                                     true_drift)

CFG = {"ground_truth_sde": {
    "A": [[0.1257302210933933, -0.1321048632913019],
          [0.6404226504432821, 0.10490011715303972]]}}


def test_sigmoid_and_true_drift_at_zero() -> None:
    """sigmoid(0) = 0.5, so the truth drift at the origin is (0.5, 0.5)."""
    assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-12
    out = true_drift(np.zeros((1, 2)), CFG)
    assert out.shape == (1, 2)
    assert np.allclose(out, 0.5)


def test_regime_labels_split_near_linear_and_saturated() -> None:
    """Regime labels must separate small-|A x| from large-|A x| states."""
    labels = regime_of(np.array([[0.0, 0.0], [10.0, 0.0]]), CFG)
    assert labels[0] == "near_linear"
    assert labels[1] == "saturated"


def test_degree3_polynomial_approximates_sigmoid_drift_on_grid() -> None:
    """A degree-3 polynomial must approximate the sigmoid drift well on
    the data-scale grid (the distillation premise, without any teacher)."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-2.0, 2.0, size=(4000, 2))
    y = true_drift(x, CFG)
    model = fit_polynomial_field(x, y, degree=3, threshold=0.05)
    assert nrmse(y, model.predict(x)) < 0.1
    tex = tex_equations(expand_to_original(model),
                        ("\\hat f_1", "\\hat f_2"), ("x1", "x2"))
    assert "\\hat f_1" in tex and "x1" in tex


def test_simulation_reproducible_and_failure_mass_retained() -> None:
    """Same increments -> same paths; an exploding drift fails with
    its mass kept in the failure count."""
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(16, 2))
    increments = rng.standard_normal((16, 50, 2))

    def calm_drift(x: np.ndarray) -> np.ndarray:
        return -0.1 * x

    def calm_diff(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    first = simulate_em(calm_drift, calm_diff, x0, 0.01, 50, increments, 20.0)
    second = simulate_em(calm_drift, calm_diff, x0, 0.01, 50, increments, 20.0)
    assert np.allclose(first["states"], second["states"], equal_nan=True)
    assert first["failure_rate"] == 0.0

    def exploding_drift(x: np.ndarray) -> np.ndarray:
        return 1e6 * x

    blown = simulate_em(exploding_drift, calm_diff, x0, 0.01, 50,
                        increments, 20.0)
    assert blown["failure_rate"] > 0.0
    assert blown["failure_rate"] == float(np.mean(blown["failed"]))
