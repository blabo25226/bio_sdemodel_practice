"""Synthetic sanity demo for potential-preserving distillation (issue #2).

A known sparse polynomial potential generates an exact gradient drift in
5 standardized dimensions; ``fit_potential_field`` must recover the drift
from state/drift pairs alone and return a field whose PC-coordinate
Jacobian is symmetric by construction.  This validates the *method and
code path* only -- it is not evidence about the LARRY teacher, whose query
data are external artifacts outside this repository.

    python -m src.potential_synthetic_demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .polynomial_math import asymmetry_ratio
from .potential_distill import PotentialField, fit_potential_field, potential_powers

SEED = 0


def synthetic_teacher(seed: int = SEED) -> tuple[PotentialField, np.ndarray, np.ndarray]:
    """Sparse degree-3 potential (degree-2 drift) with a non-trivial transform.

    Returns (teacher field, train states, train drifts); states are drawn in
    PC units around a nonzero mean with unequal scales, as in the real data.
    """
    rng = np.random.default_rng(seed)
    mean = np.array([3.0, -2.5, 0.5, -1.5, 1.0])
    scale = np.array([7.0, 3.0, 6.0, 2.0, 1.5])
    powers = potential_powers(5, drift_degree=2)
    coefficients = np.zeros(len(powers))
    chosen = {(2, 0, 0, 0, 0): -3.0, (0, 2, 0, 0, 0): -2.0, (0, 0, 2, 0, 0): -1.0,
              (0, 0, 0, 2, 0): -2.5, (0, 0, 0, 0, 2): -1.5, (1, 1, 0, 0, 0): 0.8,
              (0, 0, 1, 0, 1): -0.6, (3, 0, 0, 0, 0): 0.4, (1, 0, 0, 0, 0): 1.2,
              (0, 1, 0, 0, 0): -0.9}
    for power, value in chosen.items():
        coefficients[np.flatnonzero(np.all(powers == power, axis=1))[0]] = value
    teacher = PotentialField(powers, coefficients, mean, scale,
                             {"role": "synthetic teacher, not the LARRY teacher"})
    z = rng.normal(size=(6000, 5))
    x = mean + z * scale
    return teacher, x, teacher.predict_drift(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/potential_synthetic/results.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    teacher, x, y = synthetic_teacher(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    z_test = rng.normal(size=(2000, 5))
    x_test = teacher.mean + z_test * teacher.scale
    y_test = teacher.predict_drift(x_test)

    fitted = fit_potential_field(x, y, drift_degree=2, threshold=0.01, alpha=1.0)
    prediction = fitted.predict_drift(x_test)
    nrmse = float(np.sqrt(np.mean((prediction - y_test) ** 2) / np.mean(y_test ** 2)))
    result = {
        "seed": args.seed,
        "scope": "synthetic gradient field only; not the LARRY teacher",
        "train_states": int(len(x)),
        "test_states": int(len(x_test)),
        "potential_terms": int(len(fitted.powers)),
        "active_terms": int(np.count_nonzero(fitted.coefficients)),
        "true_active_terms": int(np.count_nonzero(teacher.coefficients)),
        "drift_test_nrmse": nrmse,
        "jacobian_asymmetry_median": float(np.median(asymmetry_ratio(fitted.jacobian_x(x_test)))),
        "support_recovered": bool(set(map(tuple, fitted.powers[np.abs(fitted.coefficients) > 1e-8]))
                                  == set(map(tuple, teacher.powers[np.abs(teacher.coefficients) > 1e-8]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
