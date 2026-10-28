"""Potential-structure-preserving polynomial distillation (issue #2).

The scDiffEq teacher drift is a gradient field in PC coordinates,
``f(x) = grad_x mu(x)`` for a scalar potential ``mu`` (the physical
potential of the papers is ``U = -mu``, so ``f = -grad U``; the sign
convention is fixed by the teacher implementation and is *not* flipped
here).  The existing SINDy fit regresses each drift component separately,
so its Jacobian is generally not symmetric and the gradient structure is
lost (quantified in ``src/diagnostics.py``).

This module fits the potential instead.  Let ``z = (x - mean) / scale`` and
define the potential in standardized coordinates,
``mu_tilde(z) = mu(mean + scale * z)``.  Then

    f_i(x) = (1 / scale_i) * d mu_tilde / d z_i ,

and the PC-coordinate Jacobian ``d f_i / d x_j`` is the symmetric matrix
``H_ij / (scale_i * scale_j)`` with ``H`` the Hessian of ``mu_tilde`` --
symmetric *by construction*, for any fitted coefficients.

If ``mu_tilde`` is a polynomial of total degree ``p + 1``, its gradient is
a polynomial of degree ``p``; a degree-3 drift therefore needs a degree-4
potential (56 drift terms in 5-D correspond to 126 potential terms before
sparsification, of which the constant term is unidentifiable and dropped).

Fitting: all drift components share one coefficient vector.  Stacking the
per-component gradient designs gives one linear system, solved by ridge
regression with STLSQ-style iterative thresholding in RMS-normalized units,
mirroring the conventions of ``src/distill.py`` (thresholds are
dimensionless coefficients after training-RMS scaling; a global target RMS
is used so that the coupling between components is not rescaled away).
Only NumPy is required, so the method can be unit-tested on synthetic
gradient fields without a teacher checkpoint.  Fitting the *real* teacher
additionally needs its potential (or drift) evaluated at query states;
those query data are external artifacts and are not in this repository.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .polynomial_math import derivative_powers, evaluate_powers, monomial_powers


def potential_powers(dimension: int, drift_degree: int) -> np.ndarray:
    """Potential exponents for a drift of ``drift_degree``; constant dropped.

    Returns exponent vectors of total degree 1..drift_degree+1 in
    ``dimension`` variables, in the ``monomial_powers`` ordering.
    """
    powers = monomial_powers(dimension, drift_degree + 1)
    return powers[np.sum(powers, axis=1) > 0]


def gradient_design(z: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Gradient of every potential monomial at states ``z`` (n, d).

    Returns an array of shape (n, d, p) with
    ``out[k, i, j] = d monomial_j / d z_i (z_k)``.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    n, d = z.shape
    design = np.zeros((n, d, len(powers)))
    for i in range(d):
        new_powers, multipliers = derivative_powers(powers, i)
        design[:, i, :] = evaluate_powers(z, new_powers) * multipliers[None, :]
    return design


@dataclass
class PotentialField:
    """Sparse polynomial potential in standardized coordinates.

    ``coefficients`` are the potential coefficients (length = number of
    non-constant monomials up to degree ``drift_degree + 1``).  The
    potential itself is identifiable only up to an additive constant.
    """

    powers: np.ndarray
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]

    def standardize(self, x: np.ndarray) -> np.ndarray:
        """PC states (n, d) -> standardized states (n, d)."""
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def predict_potential(self, x: np.ndarray) -> np.ndarray:
        """Potential values mu_tilde(z(x)) for states (n, d) -> (n,)."""
        theta = evaluate_powers(self.standardize(x), self.powers)
        return theta @ self.coefficients

    def predict_drift(self, x: np.ndarray) -> np.ndarray:
        """Drift ``f_i = (1/scale_i) d mu_tilde/d z_i`` for states (n, d) -> (n, d)."""
        design = gradient_design(self.standardize(x), self.powers)
        return np.einsum("nij,j->ni", design, self.coefficients) / self.scale[None, :]

    def jacobian_x(self, x: np.ndarray) -> np.ndarray:
        """PC-coordinate drift Jacobian (n, d, d); symmetric by construction."""
        z = self.standardize(x)
        n, d = len(z), len(self.mean)
        hessian = np.zeros((n, d, d))
        for i in range(d):
            first_powers, first_mult = derivative_powers(self.powers, i)
            for j in range(d):
                second_powers, second_mult = derivative_powers(first_powers, j)
                theta = evaluate_powers(z, second_powers) * (first_mult * second_mult)[None, :]
                hessian[:, i, j] = theta @ self.coefficients
        return hessian / (self.scale[None, :, None] * self.scale[None, None, :])

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
    def load(cls, path: Path) -> "PotentialField":
        """Restore a field saved with :meth:`save`."""
        data = json.loads(Path(path).read_text())
        return cls(np.asarray(data["powers"], dtype=int),
                   np.asarray(data["coefficients"], dtype=np.float64),
                   np.asarray(data["mean"], dtype=np.float64),
                   np.asarray(data["scale"], dtype=np.float64),
                   data.get("metadata", {}))


def _ridge_solve(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge solution of ``design @ c ~= target`` via normal equations."""
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += alpha
    return np.linalg.solve(gram, design.T @ target)


def fit_potential_field(x: np.ndarray, y: np.ndarray, drift_degree: int,
                        threshold: float, alpha: float,
                        max_iter: int = 20) -> PotentialField:
    """Fit a potential whose gradient matches drift targets ``y`` at states ``x``.

    ``x``: (n, d) PC states (training only; mean/scale are estimated here).
    ``y``: (n, d) teacher drift values in PC/day at the same states.
    ``drift_degree``: degree of the resulting drift; the potential uses
    degree ``drift_degree + 1``.  ``threshold`` / ``alpha`` follow the
    RMS-normalized STLSQ conventions of ``src/distill.py``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid training states/targets")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    powers = potential_powers(x.shape[1], drift_degree)
    design = gradient_design(z, powers)  # (n, d, p)
    n, d, p = design.shape
    # Stack components: rows (k, i), target is grad_z mu = f_i * scale_i.
    stacked = design.transpose(0, 1, 2).reshape(n * d, p)
    target = (y * scale[None, :]).reshape(n * d)
    column_rms = np.sqrt(np.mean(stacked ** 2, axis=0))
    column_rms[column_rms < 1e-12] = 1.0
    target_rms = float(np.sqrt(np.mean(target ** 2))) or 1.0
    normalized = stacked / column_rms[None, :]
    scaled_target = target / target_rms

    active = np.ones(p, dtype=bool)
    coefficients = np.zeros(p)
    for _ in range(max_iter):
        solved = _ridge_solve(normalized[:, active], scaled_target, alpha)
        candidate = np.zeros(p)
        candidate[active] = solved
        keep = np.abs(candidate) >= threshold
        if keep.sum() == 0:
            keep[np.argmax(np.abs(candidate))] = True
        coefficients = candidate
        if (keep == active).all():
            break
        active = keep
    # Final refit on the selected support, ridge retained (as in distill.py).
    solved = _ridge_solve(normalized[:, active], scaled_target, alpha)
    coefficients = np.zeros(p)
    # Convert normalized-unit coefficients back: c_phys = c_norm * target_rms / column_rms.
    coefficients[active] = solved * target_rms / column_rms[active]
    metadata = {
        "method": "potential-preserving polynomial distillation",
        "drift_degree": drift_degree,
        "potential_degree": drift_degree + 1,
        "threshold": threshold,
        "alpha": alpha,
        "potential_terms": int(p),
        "active_terms": int(np.count_nonzero(coefficients)),
        "coordinates": "potential in z=(PC-mean)/scale; drift f_i=(1/scale_i) dmu/dz_i in PC/day",
        "threshold_units": "dimensionless coefficient after training RMS scaling of the stacked gradient design and targets",
        "jacobian_symmetry": "exact by construction in PC coordinates",
    }
    return PotentialField(powers, coefficients, mean, scale, metadata)
