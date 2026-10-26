"""Shared NumPy utilities for the daily Neural SDE -> SINDy series.

Contents: polynomial basis (ordering matches PySINDy's PolynomialLibrary),
an STLSQ sparse polynomial fit in the conventions of ``src/distill.py``
(RMS-normalized thresholding, transforms from the training states only),
exact expansion of a fitted field back into original coordinates, TeX
rendering of the expanded equations, and a paired Euler-Maruyama simulator
that retains failed-trajectory mass. Only NumPy is required here; teacher
training (PyTorch) lives in the per-day modules.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def monomial_powers(dimension: int, degree: int) -> np.ndarray:
    """Exponent vectors of total degree <= degree, PySINDy ordering.

    Returns an array of shape (n_terms, dimension).
    """
    powers: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        for combo in itertools.combinations_with_replacement(range(dimension), total):
            power = [0] * dimension
            for index in combo:
                power[index] += 1
            powers.append(tuple(power))
    return np.asarray(powers, dtype=int)


def evaluate_powers(z: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Monomial design matrix for states ``z`` (n, d) -> (n, n_terms)."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    theta = np.ones((len(z), len(powers)), dtype=np.float64)
    for j, power in enumerate(powers):
        active = np.flatnonzero(power)
        if len(active):
            theta[:, j] = np.prod(z[:, active] ** power[active], axis=1)
    return theta


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """sqrt(mean squared error / mean squared truth), as in src/distill.py."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2) / max(np.mean(truth ** 2), 1e-15)))


@dataclass
class PolynomialField:
    """Sparse polynomial field in standardized coordinates.

    ``z = (x - mean) / scale``; coefficients have shape (outputs, n_terms)
    and outputs are in original (unstandardized) field units.
    """

    powers: np.ndarray
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Field values for states ``x`` (n, d) -> (n, outputs)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.scale
        return evaluate_powers(z, self.powers) @ self.coefficients.T

    def save(self, path: Path) -> None:
        """Save powers / coefficients / transform as JSON (no pickle)."""
        path.write_text(json.dumps({
            "powers": self.powers.tolist(),
            "coefficients": self.coefficients.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "metadata": self.metadata,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PolynomialField":
        """Restore a field saved with :meth:`save`."""
        data = json.loads(Path(path).read_text())
        return cls(np.asarray(data["powers"], dtype=int),
                   np.asarray(data["coefficients"], dtype=np.float64),
                   np.asarray(data["mean"], dtype=np.float64),
                   np.asarray(data["scale"], dtype=np.float64),
                   data.get("metadata", {}))


def _stlsq_single(theta_n: np.ndarray, target_n: np.ndarray, threshold: float,
                  ridge: float, max_iter: int = 20) -> np.ndarray:
    """STLSQ for one output in RMS-normalized units -> normalized coefficients."""
    n_terms = theta_n.shape[1]
    active = np.ones(n_terms, dtype=bool)
    coef = np.zeros(n_terms)
    for _ in range(max_iter):
        if not active.any():
            return np.zeros(n_terms)
        design = theta_n[:, active]
        gram = design.T @ design / len(design) + ridge * np.eye(active.sum())
        coef_active = np.linalg.solve(gram, design.T @ target_n / len(design))
        small = np.abs(coef_active) < threshold
        if not small.any():
            coef = np.zeros(n_terms)
            coef[active] = coef_active
            break
        idx = np.flatnonzero(active)
        active[idx[small]] = False
        coef = np.zeros(n_terms)
        coef[active] = coef_active[~small]
    if active.any():
        # Unbias: plain least squares on the surviving support.
        design = theta_n[:, active]
        solution, *_ = np.linalg.lstsq(design, target_n, rcond=None)
        coef = np.zeros(n_terms)
        coef[active] = solution
    return coef


def fit_polynomial_field(x: np.ndarray, y: np.ndarray, degree: int, threshold: float,
                         ridge: float = 1e-4) -> PolynomialField:
    """Fit states (n, d) -> targets (n, k) with polynomial STLSQ.

    Standardization and RMS normalization use only the supplied (training)
    arrays, mirroring ``src/distill.py``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite training data")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    powers = monomial_powers(x.shape[1], degree)
    theta = evaluate_powers(z, powers)
    feature_rms = np.sqrt(np.mean(theta ** 2, axis=0))
    feature_rms[feature_rms < 1e-12] = 1.0
    target_rms = np.sqrt(np.mean(y ** 2, axis=0))
    target_rms[target_rms < 1e-12] = 1.0
    theta_n = theta / feature_rms[None, :]
    coefficients = np.zeros((y.shape[1], len(powers)))
    for j in range(y.shape[1]):
        coef_n = _stlsq_single(theta_n, y[:, j] / target_rms[j], threshold, ridge)
        coefficients[j] = coef_n * target_rms[j] / feature_rms
    metadata = {"degree": degree, "threshold": threshold, "ridge": ridge,
                "library_size": int(len(powers)),
                "active_terms": int(np.count_nonzero(coefficients)),
                "optimizer": "numpy STLSQ (RMS-normalized, unbias refit)",
                "feature_rms": feature_rms.tolist(),
                "target_rms": target_rms.tolist()}
    return PolynomialField(powers, coefficients, mean, scale, metadata)


def expand_to_original(field: PolynomialField) -> list[dict[tuple[int, ...], float]]:
    """Expand a standardized-coordinate field into original coordinates.

    Returns one dict per output mapping an exponent tuple in the original
    variables to its coefficient, so the result can be compared term by term
    with a ground-truth polynomial and rendered as TeX.
    """
    d = len(field.mean)
    expanded: list[dict[tuple[int, ...], float]] = []
    for i in range(field.coefficients.shape[0]):
        total: dict[tuple[int, ...], float] = {}
        for j, power in enumerate(field.powers):
            coef = field.coefficients[i, j]
            if coef == 0.0:
                continue
            # prod_k ((x_k - mean_k) / scale_k) ** power_k, expanded exactly.
            term: dict[tuple[int, ...], float] = {tuple([0] * d): float(coef)}
            for k in range(d):
                for _ in range(int(power[k])):
                    nxt: dict[tuple[int, ...], float] = {}
                    for exp, val in term.items():
                        shifted = list(exp)
                        shifted[k] += 1
                        key = tuple(shifted)
                        nxt[key] = nxt.get(key, 0.0) + val / field.scale[k]
                        nxt[exp] = nxt.get(exp, 0.0) - val * field.mean[k] / field.scale[k]
                    term = nxt
            for exp, val in term.items():
                total[exp] = total.get(exp, 0.0) + val
        expanded.append({exp: val for exp, val in total.items() if abs(val) > 1e-12})
    return expanded


def evaluate_expanded(expanded: list[dict[tuple[int, ...], float]],
                      x: np.ndarray) -> np.ndarray:
    """Evaluate expanded original-coordinate polynomials at states (n, d)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.zeros((len(x), len(expanded)))
    for i, poly in enumerate(expanded):
        for exp, coef in poly.items():
            term = np.full(len(x), coef)
            for k, power in enumerate(exp):
                if power:
                    term = term * x[:, k] ** power
            out[:, i] += term
    return out


def _monomial_tex(exp: tuple[int, ...], var_names: tuple[str, ...]) -> str:
    parts = []
    for name, power in zip(var_names, exp):
        if power == 1:
            parts.append(name)
        elif power > 1:
            parts.append(f"{name}^{{{power}}}")
    return " ".join(parts) if parts else "1"


def tex_equations(expanded: list[dict[tuple[int, ...], float]],
                  lhs_names: tuple[str, ...],
                  var_names: tuple[str, ...] = ("x", "y", "z"),
                  digits: int = 4) -> str:
    """Render expanded polynomials as TeX equations, one per output."""
    lines = []
    for i, poly in enumerate(expanded):
        terms = sorted(poly.items(), key=lambda kv: (sum(kv[0]), kv[0]))
        rendered = []
        for exp, coef in terms:
            mono = _monomial_tex(exp, var_names)
            magnitude = f"{abs(coef):.{digits}g}"
            body = magnitude if mono == "1" else (mono if magnitude == "1" else f"{magnitude}\\,{mono}")
            rendered.append(("-" if coef < 0 else "+", body))
        expr = ""
        for sign, body in rendered:
            if not expr:
                expr = ("- " if sign == "-" else "") + body
            else:
                expr += f" {sign} {body}"
        lines.append(f"{lhs_names[i]} &= {expr if expr else '0'} \\\\")
    return "\\begin{aligned}\n" + "\n".join(lines) + "\n\\end{aligned}"


def simulate_em(drift_fn, diffusion_fn, x0: np.ndarray, dt: float, n_steps: int,
                increments: np.ndarray, failure_radius: float,
                obs_every: int = 1) -> dict[str, Any]:
    """Paired Euler-Maruyama simulation with diagonal diffusion.

    ``increments`` has shape (n, n_steps, d) of standard normals (shared
    across models for paired comparison). Failures are latched and
    classified as ``non_finite`` or ``outside_radius``; failed trajectories
    keep their mass in the returned counts and are NaN after failure.
    Returns observed states (n, n_obs, d), failure flags and classes.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    n, d = x.shape
    if increments.shape != (n, n_steps, d):
        raise ValueError("increments shape mismatch")
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    observed = [x.copy()]
    sqrt_dt = np.sqrt(dt)
    for step in range(n_steps):
        active = ~failed
        if active.any():
            drift = drift_fn(x[active])
            diffusion = diffusion_fn(x[active])
            x[active] = x[active] + drift * dt + diffusion * sqrt_dt * increments[active, step]
        bad_nonfinite = ~np.isfinite(x).all(axis=1)
        bad_radius = np.linalg.norm(x, axis=1) > failure_radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        x[failed] = np.nan
        if (step + 1) % obs_every == 0:
            observed.append(x.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class,
            "failure_rate": float(np.mean(failed))}
