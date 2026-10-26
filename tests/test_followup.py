"""Tests for the follow-up analyses (issues #1-#4).

Pure NumPy: these tests must pass without PySINDy, PyTorch, a GPU, a
teacher checkpoint, or the external query data.
"""
from pathlib import Path

import numpy as np

from src.polynomial_math import (PolynomialModel, asymmetry_ratio,
                                 evaluate_powers, monomial_powers)
from src.potential_distill import (PotentialField, fit_potential_field,
                                   potential_powers)
from src.potential_synthetic_demo import synthetic_teacher

ROOT = Path(__file__).resolve().parent.parent
DRIFT_JSON = ROOT / "outputs" / "sindy_pc5" / "drift" / "model.json"


def test_monomial_counts_match_issue1_basis_sizes():
    # Issue #1: 5-D polynomial library grows 21 (degree 2) -> 56 (degree 3).
    assert len(monomial_powers(5, 2)) == 21
    assert len(monomial_powers(5, 3)) == 56
    # A degree-3 drift needs a degree-4 potential: C(9, 4) - 1 terms.
    assert len(potential_powers(5, 3)) == 125


def test_monomial_ordering_matches_saved_model():
    model = PolynomialModel.load_json(DRIFT_JSON)
    np.testing.assert_array_equal(monomial_powers(5, 2), model.powers)


def test_analytic_jacobian_matches_finite_differences():
    model = PolynomialModel.load_json(DRIFT_JSON)
    rng = np.random.default_rng(0)
    x = model.mean + rng.normal(size=(8, 5)) * model.scale
    analytic = model.jacobian_x(x)
    eps = 1e-6
    numeric = np.zeros_like(analytic)
    for j in range(5):
        step = np.zeros(5)
        step[j] = eps
        numeric[:, :, j] = (model.predict(x + step) - model.predict(x - step)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_frozen_drift_is_far_from_a_gradient_field():
    # Regression guard for the issue #2/#3 diagnostic: in PC coordinates the
    # free SINDy drift's Jacobian asymmetry is large at in-domain states.
    model = PolynomialModel.load_json(DRIFT_JSON)
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(512, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = directions * (0.5 * float(model.metadata["domain_radius"]))
    ratio = asymmetry_ratio(model.jacobian_x(points))
    assert np.median(ratio) > 0.5
    assert np.mean(ratio < 0.05) == 0.0


def test_potential_fit_recovers_synthetic_support_and_symmetry():
    teacher, x, y = synthetic_teacher()
    fitted = fit_potential_field(x, y, drift_degree=2, threshold=0.01, alpha=1.0)
    prediction = fitted.predict_drift(x)
    nrmse = float(np.sqrt(np.mean((prediction - y) ** 2) / np.mean(y ** 2)))
    assert nrmse < 0.05
    true_support = set(map(tuple, teacher.powers[np.abs(teacher.coefficients) > 1e-8]))
    fitted_support = set(map(tuple, fitted.powers[np.abs(fitted.coefficients) > 1e-8]))
    assert fitted_support == true_support
    # PC-coordinate Jacobian is symmetric by construction (exact zeros off it).
    jac = fitted.jacobian_x(x[:16])
    np.testing.assert_allclose(jac, np.swapaxes(jac, -1, -2), atol=1e-10)
    assert np.median(asymmetry_ratio(jac)) == 0.0


def test_potential_field_json_roundtrip(tmp_path):
    teacher, x, _ = synthetic_teacher()
    path = tmp_path / "potential.json"
    teacher.save(path)
    restored = PotentialField.load(path)
    np.testing.assert_allclose(restored.predict_drift(x), teacher.predict_drift(x))


def test_evaluate_powers_constant_and_linear():
    z = np.array([[2.0, 3.0]])
    theta = evaluate_powers(z, monomial_powers(2, 1))
    np.testing.assert_allclose(theta, [[1.0, 2.0, 3.0]])
