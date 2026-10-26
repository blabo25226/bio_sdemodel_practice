"""Pure-NumPy polynomial field mathematics shared by the follow-up analyses.

The existing ``src/distill.py`` depends on PySINDy / PyTorch.  The utilities
here deliberately depend only on NumPy so that the *frozen* symbolic models
committed under ``outputs/sindy_pc5/`` (``model.json``) can be analysed in any
environment, without a teacher checkpoint, GPU, or the original query data.

Coordinates
-----------
A saved field is a polynomial in standardized coordinates
``z_i = (x_i - mean_i) / scale_i`` whose *output* is in original PC units
(drift: PC/day, diffusion factor ``G``: PC/sqrt(day)).  Two Jacobians matter
and must not be confused (see issue #2):

* ``J_z[i, j] = d f_i / d z_j`` -- Jacobian in standardized coordinates.
* ``J_x[i, j] = d f_i / d x_j = J_z[i, j] / scale_j`` -- Jacobian in PC
  coordinates.  The teacher drift is a gradient with respect to the PC
  coordinates, ``f = grad_x mu``, so *this* is the Jacobian that would be
  symmetric (a Hessian) for a structure-preserving distillation.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def monomial_powers(dimension: int, degree: int) -> np.ndarray:
    """All exponent vectors in ``dimension`` variables of total degree <= degree.

    Ordering matches ``pysindy.PolynomialLibrary`` (and therefore the saved
    ``model.json`` files): by total degree, then by combinations with
    replacement of variable indices.
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
    """Monomial design matrix for states ``z`` of shape (n, d) -> (n, p)."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    theta = np.ones((len(z), len(powers)), dtype=np.float64)
    for j, power in enumerate(powers):
        active = np.flatnonzero(power)
        if len(active):
            theta[:, j] = np.prod(z[:, active] ** power[active], axis=1)
    return theta


def derivative_powers(powers: np.ndarray, variable: int) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of each monomial w.r.t. ``z[variable]``.

    Returns ``(new_powers, multipliers)`` such that
    ``d/dz_v prod z^a = multiplier * prod z^new_power``; terms whose exponent
    in ``variable`` is zero get multiplier 0 and an unchanged (unused) power.
    """
    new_powers = np.array(powers, dtype=int, copy=True)
    multipliers = powers[:, variable].astype(np.float64)
    nonzero = multipliers > 0
    new_powers[nonzero, variable] -= 1
    return new_powers, multipliers


@dataclass
class PolynomialModel:
    """A frozen polynomial field loaded from a committed ``model.json``."""

    powers: np.ndarray
    coefficients: np.ndarray  # (outputs, terms), in terms of z, output in PC units
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict

    @classmethod
    def load_json(cls, path: str | Path) -> "PolynomialModel":
        data = json.loads(Path(path).read_text())
        return cls(
            powers=np.asarray(data["powers"], dtype=int),
            coefficients=np.asarray(data["coefficients"], dtype=np.float64),
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            metadata=data.get("metadata", {}),
        )

    @property
    def dimension(self) -> int:
        return len(self.mean)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def predict_z(self, z: np.ndarray) -> np.ndarray:
        """Field values for standardized states ``z`` (n, d) -> (n, outputs)."""
        return evaluate_powers(z, self.powers) @ self.coefficients.T

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Field values for PC states ``x`` (n, d) -> (n, outputs)."""
        return self.predict_z(self.standardize(x))

    def jacobian_z(self, z: np.ndarray) -> np.ndarray:
        """``J_z[i, j] = d f_i / d z_j`` for each state -> (n, outputs, d)."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        n, d = len(z), self.dimension
        jac = np.zeros((n, self.coefficients.shape[0], d))
        for j in range(d):
            new_powers, multipliers = derivative_powers(self.powers, j)
            if not np.any(multipliers):
                continue
            theta = evaluate_powers(z, new_powers) * multipliers[None, :]
            jac[:, :, j] = theta @ self.coefficients.T
        return jac

    def jacobian_x(self, x: np.ndarray) -> np.ndarray:
        """``J_x[i, j] = d f_i / d x_j`` in PC coordinates -> (n, outputs, d).

        ``d/d x_j = (1 / scale_j) d/d z_j``; this is the Jacobian whose
        symmetry expresses the teacher's gradient (potential) structure.
        """
        jac_z = self.jacobian_z(self.standardize(x))
        return jac_z / self.scale[None, None, :]


def asymmetry_ratio(jacobians: np.ndarray) -> np.ndarray:
    """``||J - J^T||_F / ||J||_F`` per state; 0 for an exact gradient field."""
    jac = np.asarray(jacobians, dtype=np.float64)
    numerator = np.linalg.norm(jac - np.swapaxes(jac, -1, -2), axis=(-2, -1))
    denominator = np.linalg.norm(jac, axis=(-2, -1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 0)
