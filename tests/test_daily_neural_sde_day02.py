"""Tests for the Day 2 (Kidger et al. 2021 OU) pipeline: exact OU transitions,
2D polynomial recovery, and the time-aware simulator."""
from __future__ import annotations

import numpy as np

from src.daily_neural_sde.common import (expand_to_original,
                                         fit_polynomial_field, monomial_powers,
                                         nrmse, tex_equations)
from src.daily_neural_sde.day02_kidger_ou import (exact_transition_mean_var,
                                                  simulate_time_aware,
                                                  true_drift)

CFG = {"ground_truth_sde": {"mu": 0.02, "theta": 0.1, "sigma": 0.4}}


def test_monomial_powers_counts_2d() -> None:
    """2D basis counts: degree 1 -> 3, degree 2 -> 6, degree 3 -> 10."""
    assert len(monomial_powers(2, 1)) == 3
    assert len(monomial_powers(2, 2)) == 6
    assert len(monomial_powers(2, 3)) == 10


def test_exact_transition_matches_fine_em_moments() -> None:
    """Exact OU transition moments must match fine Euler-Maruyama moments."""
    rng = np.random.default_rng(0)
    t0, z0, dt = 5.0, 2.0, 1.0
    mean, var = exact_transition_mean_var(np.array([t0]), np.array([z0]), dt, CFG)
    n_paths, dt_sim = 20000, 0.001
    z = np.full(n_paths, z0)
    steps = int(round(dt / dt_sim))
    for step in range(steps):
        t = t0 + step * dt_sim
        z = z + (0.02 * t - 0.1 * z) * dt_sim + 0.4 * np.sqrt(dt_sim) * rng.standard_normal(n_paths)
    assert abs(z.mean() - mean[0]) < 0.02
    assert abs(z.var() - var[0]) < 0.01


def test_stlsq_recovers_ou_drift() -> None:
    """Fitting the exact OU drift in (t, z) must recover mu and -theta."""
    rng = np.random.default_rng(1)
    tz = np.stack([rng.uniform(0, 63, size=4000),
                   rng.uniform(-2, 12, size=4000)], axis=1)
    y = true_drift(tz, CFG)
    model = fit_polynomial_field(tz, y, degree=1, threshold=0.05)
    assert nrmse(y, model.predict(tz)) < 1e-6
    expanded = expand_to_original(model)
    assert expanded[0][(1, 0)] == np.float64(0.02) or abs(expanded[0][(1, 0)] - 0.02) < 1e-6
    assert abs(expanded[0][(0, 1)] + 0.1) < 1e-6
    tex = tex_equations(expanded, ("\\hat f",), ("t", "z"))
    assert "\\hat f" in tex and "t" in tex and "z" in tex


def test_time_aware_simulation_reproducible_and_failure_mass_retained() -> None:
    """Same increments -> same paths; exploding drift fails with mass kept."""
    rng = np.random.default_rng(3)
    z0 = rng.normal(size=16)
    increments = rng.standard_normal((16, 50))

    def calm_drift(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return -0.1 * z

    def calm_diffusion(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.full_like(z, 0.4)

    first = simulate_time_aware(calm_drift, calm_diffusion, z0, 0.01, 50,
                                increments, 50.0, 10)
    second = simulate_time_aware(calm_drift, calm_diffusion, z0, 0.01, 50,
                                 increments, 50.0, 10)
    np.testing.assert_array_equal(first["states"], second["states"])
    assert first["failure_rate"] == 0.0
    assert first["states"].shape == (16, 6)

    def exploding_drift(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 50.0 * z ** 3

    blown = simulate_time_aware(exploding_drift, calm_diffusion, z0, 0.01, 50,
                                increments, 50.0, 10)
    assert blown["failure_rate"] > 0.0
    assert sum(c is not None for c in blown["failure_class"]) == int(blown["failed"].sum())
