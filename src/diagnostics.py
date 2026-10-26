"""Diagnostics for the frozen 5-PC symbolic SDE (issues #2 and #3).

Only the committed ``model.json`` files are used: no teacher checkpoint, no
query data, no GPU.  Everything is evaluated in batch with NumPy.

Questions addressed
-------------------
1. How far is the free (component-wise) SINDy drift from a gradient field?
   Measured as the asymmetry of the Jacobian in *PC coordinates*,
   ``J_x[i, j] = d f_i / d x_j`` (the teacher's drift is ``grad_x mu``, so a
   structure-preserving fit would have a symmetric ``J_x``).  The same
   quantity in standardized coordinates ``z`` is reported for contrast,
   because ``J_z`` mixes the per-axis scales and is *not* expected to be
   symmetric even for an exact teacher distillation.
2. How do the drift, the diffusion factor ``G`` and the Ito generator of
   ``V(x) = ||x||^2`` behave on shells ``||x|| = c * R`` around and beyond
   the training domain radius ``R`` (domain definition of
   ``src/validate.py``: Euclidean PC norm, failure at ``10 * R``)?
   For ``V = ||x||^2``, ``L V(x) = 2 x . f(x) + ||G(x)||_F^2`` (rank-1
   Brownian motion, so ``tr D = ||G||^2``).  A persistently positive
   ``L V`` outside the domain means the *generator itself* pushes the
   second moment outward there; it is a statement about the fitted
   vector field, not a proof that the continuous-time SDE explodes.
3. Probe simulations: Euler-Maruyama integrations of the symbolic SDE
   alone, started from synthetic probe points on the ``0.5 R`` and ``1.0 R``
   shells (these are *not* observed cells and are never presented as a
   replication of the validation protocol).  Failures are classified as
   non-finite, ``> 10 R`` norm exceedance, or finite-but-outside-domain,
   for two step sizes and three Brownian seeds, keeping all probe mass.

Usage
-----
    python -m src.diagnostics [--output-dir outputs/diagnostics_pc5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .polynomial_math import PolynomialModel, asymmetry_ratio

SEED = 0
SHELL_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
N_DIRECTIONS = 4096
PROBE_SEEDS = (202, 203, 204)
PROBE_DT = (0.05, 0.025)
PROBE_HORIZON_DAYS = 4.0


def sample_shell(rng: np.random.Generator, n: int, radius: float,
                 dimension: int) -> np.ndarray:
    """Uniform directions scaled to ``||x|| = radius``; states (n, d) in PC units."""
    directions = rng.normal(size=(n, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * radius


def shell_statistics(drift: PolynomialModel, diffusion: PolynomialModel,
                     radius: float, rng: np.random.Generator) -> pd.DataFrame:
    """Per-shell growth statistics of f, G and L||x||^2; batch evaluation."""
    rows = []
    for fraction in SHELL_FRACTIONS:
        x = sample_shell(rng, N_DIRECTIONS, fraction * radius, drift.dimension)
        f = drift.predict(x)
        g = diffusion.predict(x)  # (n, d), Brownian dimension 1
        norms = np.linalg.norm(x, axis=1)
        radial = np.einsum("ij,ij->i", x, f) / np.maximum(norms, 1e-12)
        generator_v = 2.0 * np.einsum("ij,ij->i", x, f) + np.sum(g ** 2, axis=1)
        rows.append({
            "shell_fraction_of_R": fraction,
            "shell_radius": fraction * radius,
            "n_points": len(x),
            "drift_norm_mean": float(np.mean(np.linalg.norm(f, axis=1))),
            "drift_norm_q95": float(np.quantile(np.linalg.norm(f, axis=1), 0.95)),
            "drift_norm_max": float(np.max(np.linalg.norm(f, axis=1))),
            "radial_drift_mean": float(np.mean(radial)),
            "radial_drift_fraction_outward": float(np.mean(radial > 0)),
            "g_norm_mean": float(np.mean(np.linalg.norm(g, axis=1))),
            "g_norm_max": float(np.max(np.linalg.norm(g, axis=1))),
            "trace_D_mean": float(np.mean(np.sum(g ** 2, axis=1))),
            "generator_norm2_mean": float(np.mean(generator_v)),
            "generator_norm2_fraction_positive": float(np.mean(generator_v > 0)),
        })
    return pd.DataFrame(rows)


def jacobian_summary(drift: PolynomialModel, radius: float,
                     rng: np.random.Generator) -> dict:
    """Jacobian asymmetry of the free drift inside the training domain."""
    points = np.concatenate([
        sample_shell(rng, 2048, fraction * radius, drift.dimension)
        for fraction in (0.25, 0.5, 0.75, 1.0)
    ])
    ratio_x = asymmetry_ratio(drift.jacobian_x(points))
    ratio_z = asymmetry_ratio(drift.jacobian_z(drift.standardize(points)))

    def stats(values: np.ndarray) -> dict:
        return {"median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
                "fraction_below_0.05": float(np.mean(values < 0.05))}

    return {
        "n_points": int(len(points)),
        "shell_fractions_of_R": [0.25, 0.5, 0.75, 1.0],
        "definition": "||J - J^T||_F / ||J||_F per state; 0 = exact gradient field",
        "jacobian_pc_coordinates": stats(ratio_x),
        "jacobian_standardized_coordinates": stats(ratio_z),
        "note": ("Symmetry is expected in PC coordinates because the teacher "
                 "drift is grad_x mu. In z coordinates the per-axis scales "
                 "break symmetry even for an exact distillation, so J_z "
                 "asymmetry alone must not be quoted as structure loss."),
    }


def simulate_probes(drift: PolynomialModel, diffusion: PolynomialModel,
                    radius: float, rng: np.random.Generator) -> pd.DataFrame:
    """Euler-Maruyama probe integrations of the symbolic SDE only.

    Initial states are synthetic shell points, not observed cells.  Every
    probe trajectory keeps its mass in exactly one outcome class.
    """
    rows = []
    initials = {fraction: sample_shell(rng, 128, fraction * radius, drift.dimension)
                for fraction in (0.5, 1.0)}
    for fraction, x0 in initials.items():
        for dt in PROBE_DT:
            steps = int(round(PROBE_HORIZON_DAYS / dt))
            for seed in PROBE_SEEDS:
                noise = np.random.default_rng(seed)
                x = x0.copy()
                nonfinite = np.zeros(len(x), dtype=bool)
                exceeded = np.zeros(len(x), dtype=bool)
                for _ in range(steps):
                    step = (drift.predict(x) * dt
                            + diffusion.predict(x) * (np.sqrt(dt) * noise.normal(size=(len(x), 1))))
                    x = x + step
                    nonfinite |= ~np.isfinite(x).all(axis=1)
                    exceeded |= np.linalg.norm(np.nan_to_num(x), axis=1) > 10 * radius
                    # Failed trajectories keep their latched classification;
                    # park them at the training mean so later steps stay finite.
                    x[nonfinite | exceeded] = drift.mean
                failed = nonfinite | exceeded
                outside = (~failed) & (np.linalg.norm(x, axis=1) > radius)
                rows.append({
                    "initial_shell_fraction_of_R": fraction,
                    "dt": dt,
                    "seed": seed,
                    "n_probes": len(x0),
                    "fraction_nonfinite": float(np.mean(nonfinite)),
                    "fraction_norm_above_10R": float(np.mean(exceeded & ~nonfinite)),
                    "fraction_any_failure": float(np.mean(failed)),
                    "fraction_finite_outside_domain": float(np.mean(outside)),
                    "fraction_finite_inside_domain": float(np.mean(~failed & ~outside)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-model", type=Path,
                        default=Path("outputs/sindy_pc5/drift/model.json"))
    parser.add_argument("--diffusion-model", type=Path,
                        default=Path("outputs/sindy_pc5/diffusion/model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics_pc5"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    drift = PolynomialModel.load_json(args.drift_model)
    diffusion = PolynomialModel.load_json(args.diffusion_model)
    radius = float(drift.metadata["domain_radius"])
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shells = shell_statistics(drift, diffusion, radius, rng)
    shells.to_csv(args.output_dir / "shell_statistics.csv", index=False)
    jacobian = jacobian_summary(drift, radius, rng)
    (args.output_dir / "jacobian_asymmetry.json").write_text(
        json.dumps(jacobian, indent=2, ensure_ascii=False))
    probes = simulate_probes(drift, diffusion, radius, rng)
    probes.to_csv(args.output_dir / "simulation_probes.csv", index=False)

    summary = {
        "seed": args.seed,
        "domain_radius_R": radius,
        "drift_model": str(args.drift_model),
        "diffusion_model": str(args.diffusion_model),
        "jacobian_asymmetry": jacobian,
        "probe_failure_fraction_by_dt": {
            str(dt): float(probes.loc[probes.dt == dt, "fraction_any_failure"].mean())
            for dt in PROBE_DT
        },
        "scope": ("Frozen committed symbolic models only. Probe initial "
                  "states are synthetic shell points, not observed cells; "
                  "results diagnose the fitted field, they do not replicate "
                  "the validation protocol of src/validate.py."),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
