"""Tests for the daily Neural SDE -> SINDy shared utilities and Day 1 fits."""
from __future__ import annotations

import numpy as np
import pytest

from src.daily_neural_sde.common import (evaluate_expanded, expand_to_original,
                                         fit_polynomial_field, monomial_powers,
                                         nrmse, simulate_em, tex_equations)
from src.daily_neural_sde.day01_lorenz import TRUE_DRIFT_SUPPORT, true_drift

CFG = {"ground_truth_sde": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
                            "alpha": [0.15, 0.15, 0.15]}}


def test_monomial_powers_counts() -> None:
    """PySINDy-compatible basis counts: 3D degree 2 -> 10, degree 3 -> 20."""
    assert len(monomial_powers(3, 2)) == 10
    assert len(monomial_powers(3, 3)) == 20
    assert len(monomial_powers(5, 2)) == 21
    assert len(monomial_powers(5, 3)) == 56


def test_stlsq_recovers_lorenz_drift() -> None:
    """Fitting the exact Lorenz drift must recover its coefficients."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-15, 15, size=(4000, 3))
    x[:, 2] = rng.uniform(0, 30, size=4000)
    # Recenter to exact means so the z-coordinate expansion of the true
    # field has no small mean-induced terms below the STLSQ threshold.
    x = x - x.mean(axis=0) + np.array([0.0, 0.0, 15.0])
    y = true_drift(x, CFG)
    model = fit_polynomial_field(x, y, degree=2, threshold=0.05)
    assert nrmse(y, model.predict(x)) < 1e-6
    expanded = expand_to_original(model)
    for i, support in enumerate(TRUE_DRIFT_SUPPORT):
        for exp, truth in support.items():
            assert expanded[i][exp] == pytest.approx(truth, rel=1e-3)


def test_expansion_matches_prediction() -> None:
    """Original-coordinate expansion must reproduce the field exactly."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 3)) * [5.0, 6.0, 7.0] + [1.0, -2.0, 3.0]
    y = true_drift(x, CFG) + rng.normal(size=(500, 3))
    model = fit_polynomial_field(x, y, degree=3, threshold=0.1)
    expanded = expand_to_original(model)
    np.testing.assert_allclose(evaluate_expanded(expanded, x), model.predict(x),
                               rtol=1e-8, atol=1e-8)


def test_tex_rendering_contains_all_components() -> None:
    """TeX output must contain one equation per field component."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 3)) * 5.0
    model = fit_polynomial_field(x, true_drift(x, CFG), degree=2, threshold=0.05)
    tex = tex_equations(expand_to_original(model), ("\\hat f_1", "\\hat f_2", "\\hat f_3"))
    for name in ("\\hat f_1", "\\hat f_2", "\\hat f_3"):
        assert name in tex
    assert "x" in tex and "z" in tex


def test_simulation_reproducible_and_failure_mass_retained() -> None:
    """Same increments -> same paths; exploding fields fail with mass kept."""
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(16, 3))
    increments = rng.standard_normal((16, 50, 3))

    def calm_drift(s: np.ndarray) -> np.ndarray:
        return -0.1 * s

    def calm_diffusion(s: np.ndarray) -> np.ndarray:
        return np.full_like(s, 0.15)

    first = simulate_em(calm_drift, calm_diffusion, x0, 0.01, 50, increments, 200.0)
    second = simulate_em(calm_drift, calm_diffusion, x0, 0.01, 50, increments, 200.0)
    np.testing.assert_array_equal(first["states"], second["states"])
    assert first["failure_rate"] == 0.0

    def exploding_drift(s: np.ndarray) -> np.ndarray:
        return 50.0 * s ** 3

    blown = simulate_em(exploding_drift, calm_diffusion, x0, 0.01, 50, increments, 200.0)
    assert blown["failure_rate"] > 0.0
    assert blown["states"].shape == (16, 51, 3)
    assert sum(c is not None for c in blown["failure_class"]) == int(blown["failed"].sum())
