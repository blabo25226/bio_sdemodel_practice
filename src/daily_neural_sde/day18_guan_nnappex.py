"""Day 18: Guan et al. (AISTATS 2026) nn-APPEX -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day18.json``):

1. Data: Section 5.1 Styblinski-Tang gradient-flow SDE. Three marginals at
   t = 0, 0.01, 0.02 from one random Gaussian mixture, Euler-Maruyama,
   diffusivity 0.2. Particles are split into disjoint train, validation,
   and test trajectories before the teacher sees any snapshot.
2. Teacher: nn-APPEX. A SiLU potential network supplies the drift
   ``b = -grad phi``. The diffusivity is a scalar MLE, not a network.
   Multi-marginal Sinkhorn couplings are the only training trajectories.
3. Distillation: direct queries of that drift. The library is powers of
   ``(x, y)`` only. A gradient-structured polynomial competes with
   componentwise STLSQ on validation. A second fit probes the constant
   diffusivity. Test is scored once.

Run from the repository root:
``python -m src.daily_neural_sde.day18_guan_nnappex``

Outputs land in ``outputs/daily_neural_sde/day18/``.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


def logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Stable log-sum-exp. Matches the reductions used by the Sinkhorn kernel."""
    array = np.asarray(values, dtype=np.float64)
    peak = np.max(array, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    total = np.log(np.sum(np.exp(array - safe), axis=axis, keepdims=True)) + safe
    if not keepdims:
        total = np.squeeze(total, axis=axis)
    return total

from .common import (PolynomialField, _stlsq_single, evaluate_expanded,
                     evaluate_powers, expand_to_original, fit_polynomial_field,
                     monomial_powers, nrmse, simulate_em, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day18.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day18"
REPORT_DIR = ROOT / "reports" / "daily_neural_sde" / "day18_guan_nnappex"
DTYPE = torch.float64

DriftFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-18 config."""
    return json.loads(path.read_text())


def true_drift(x: np.ndarray) -> np.ndarray:
    """Styblinski-Tang drift ``-grad Psi``, shape (n, d).

    ``Psi(v) = 0.5 * sum_i (v_i^4 - 16 v_i^2 + 5 v_i)``, so component
    ``i`` is ``-2 x_i^3 + 16 x_i - 2.5``.
    """
    x = np.asarray(x, dtype=np.float64)
    return -2.0 * x ** 3 + 16.0 * x - 2.5


def true_sigma(sigma2: float = 0.2) -> float:
    """Isotropic diffusion coefficient ``sqrt(sigma2)``."""
    return float(math.sqrt(sigma2))


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def rms(values: np.ndarray) -> float:
    """Root mean square of every entry."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array ** 2)))


def jsonify(obj: Any) -> Any:
    """Recursively replace NumPy values with JSON scalars."""
    if isinstance(obj, dict):
        return {str(key): jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sample_random_gmm(n: int, dimension: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Paper ``sample_random_gmm``: K uniform in 1..10, Dirichlet weights.

    Means are uniform on ``[-3, 3]^d``. Isotropic variances are uniform on
    ``[0.5, 1]``. Returns samples of shape (n, d) and the realized parameters.
    """
    n_components = int(rng.integers(1, 11))
    means = rng.uniform(-3.0, 3.0, size=(n_components, dimension))
    variances = rng.uniform(0.5, 1.0, size=n_components)
    weights = rng.dirichlet(np.ones(n_components))
    component = rng.choice(n_components, size=n, p=weights)
    samples = np.empty((n, dimension), dtype=np.float64)
    for index in range(n_components):
        mask = component == index
        count = int(mask.sum())
        if count:
            samples[mask] = means[index] + math.sqrt(float(variances[index])) * rng.standard_normal((count, dimension))
    params = {"n_components": n_components, "means": means, "variances": variances, "weights": weights}
    return samples, params


def sample_gmm_params(params: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points from an already realized mixture. Shape (n, d)."""
    means = np.asarray(params["means"], dtype=np.float64)
    variances = np.asarray(params["variances"], dtype=np.float64)
    weights = np.asarray(params["weights"], dtype=np.float64)
    component = rng.choice(len(weights), size=n, p=weights)
    samples = np.empty((n, means.shape[1]), dtype=np.float64)
    for index in range(len(weights)):
        mask = component == index
        count = int(mask.sum())
        if count:
            samples[mask] = means[index] + math.sqrt(float(variances[index])) * rng.standard_normal((count, means.shape[1]))
    return samples


def simulate_known(x0: np.ndarray, dt: float, n_steps: int, sigma: float,
                   rng: np.random.Generator, radius: float,
                   drift_fn: DriftFn = true_drift) -> dict[str, Any]:
    """Euler-Maruyama. Failed particles are resampled by the caller.

    Increments have shape (n, n_steps, d). States have shape (n, n_steps+1, d).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n, dimension = x.shape
    increments = rng.standard_normal((n, n_steps, dimension))
    states = np.empty((n, n_steps + 1, dimension), dtype=np.float64)
    states[:, 0] = x
    failed = np.zeros(n, dtype=bool)
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps):
        active = ~failed
        if active.any():
            drift = drift_fn(x[active])
            x[active] = x[active] + drift * dt + sigma * sqrt_dt * increments[active, step]
        bad = ~np.isfinite(x).all(axis=1) | (np.linalg.norm(x, axis=1) > radius)
        newly = (~failed) & bad
        failed |= newly
        x = x.copy()
        x[failed] = np.nan
        states[:, step + 1] = x
    return {"states": states, "increments": increments, "failed": failed}


def generate_population(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """One mixture, then disjoint particles. Discards are redrawn from the same mixture."""
    data = cfg["data"]
    n_total = int(data["n_train"] + data["n_val"] + data["n_test"])
    dimension = int(data["dimension"])
    x0, params = sample_random_gmm(n_total, dimension, rng)
    sigma = true_sigma(float(data["sigma2"]))
    dt = float(data["dt"])
    n_steps = int(data["n_steps"])
    radius = float(data["generation_radius"])
    out = simulate_known(x0, dt, n_steps, sigma, rng, radius)
    discarded = 0
    guard = 0
    while out["failed"].any() and guard < int(data["max_resample_rounds"]):
        bad = out["failed"]
        discarded += int(bad.sum())
        x0[bad] = sample_gmm_params(params, int(bad.sum()), rng)
        replacement = simulate_known(x0[bad], dt, n_steps, sigma, rng, radius)
        out["states"][bad] = replacement["states"]
        out["increments"][bad] = replacement["increments"]
        out["failed"][bad] = replacement["failed"]
        guard += 1
    if out["failed"].any():
        raise RuntimeError("Could not draw a finite population inside the generation radius")
    return {"states": out["states"], "increments": out["increments"], "gmm": params,
            "discarded": discarded, "sigma": sigma}


def split_population(population: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Particle-disjoint train / validation / test views. Each value is a copy."""
    data = cfg["data"]
    n_train = int(data["n_train"])
    n_val = int(data["n_val"])
    cuts = {"train": (0, n_train), "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n_train + n_val + int(data["n_test"]))}
    frames = {}
    for name, (start, stop) in cuts.items():
        states = population["states"][start:stop]
        frames[name] = {
            "states": states.copy(),
            "increments": population["increments"][start:stop].copy(),
            "x0": states[:, 0].copy(),
            "query": states.reshape(-1, states.shape[-1]).copy(),
        }
    return frames


def diffusion_prior(cfg: dict[str, Any]) -> float:
    """Section 5.1 prior ``sigma2 * 10^U[-1, 1]`` from an independent Generator."""
    rng = np.random.default_rng(int(cfg["seeds"]["diffusion_prior"]))
    sigma2 = float(cfg["data"]["sigma2"])
    return float(sigma2 * 10.0 ** rng.uniform(-1.0, 1.0))


def gaussian_cost(left: np.ndarray, right: np.ndarray, drift_step: np.ndarray,
                  precision: np.ndarray) -> np.ndarray:
    """``C[i, j] = 0.5 (y_j - x_i - b_i dt)^T P (same)``. Shape (n, n)."""
    gap = right[None, :, :] - left[:, None, :] - drift_step[:, None, :]
    return 0.5 * np.einsum("...i,ij,...j->...", gap, precision, gap)


def mmos_sinkhorn(states: np.ndarray, dt: float, drift_fn: DriftFn, sigma2: float,
                  rng: np.random.Generator, max_iter: int, tol: float,
                  epsilon: float = 1.0, eta: float = 0.5) -> dict[str, Any]:
    """Multi-marginal IPF from the released ``MMOT_trajectory_inference``.

    ``states`` has shape (n, T, d) and is shuffled within each time before
    the kernels are built, so the teacher never sees the true pairing.
    Returned paths have shape (n, T, d).
    """
    states = np.asarray(states, dtype=np.float64)
    n, n_times, dimension = states.shape
    marginals = []
    for time_index in range(n_times):
        column = states[:, time_index, :].copy()
        rng.shuffle(column)
        marginals.append(column)
    sigma2_used = max(float(sigma2), 1e-8)
    sigma_dt = (sigma2_used + 1e-8) * float(dt)
    precision = np.eye(dimension) / sigma_dt
    log_kernels = []
    for time_index in range(n_times - 1):
        drift_step = drift_fn(marginals[time_index]) * float(dt)
        cost = gaussian_cost(marginals[time_index], marginals[time_index + 1], drift_step, precision)
        log_kernels.append(-(cost - float(np.min(cost))) / float(epsilon))
    target = np.full(n, 1.0 / n)
    log_h = [np.zeros(n) for _ in range(n_times)]
    max_err = 1.0
    used = 0
    for used in range(1, max_iter + 1):
        log_minus = [np.zeros(n) for _ in range(n_times)]
        for time_index in range(n_times - 1):
            scores = log_kernels[time_index] + (log_h[time_index] + log_minus[time_index])[:, None]
            scores -= np.max(scores, axis=0, keepdims=True)
            message = logsumexp(scores, axis=0)
            message -= np.max(message)
            log_minus[time_index + 1] = message
        log_plus = [np.zeros(n) for _ in range(n_times)]
        for time_index in reversed(range(n_times - 1)):
            scores = log_kernels[time_index] + (log_h[time_index + 1] + log_plus[time_index + 1])[None, :]
            scores -= np.max(scores, axis=1, keepdims=True)
            message = logsumexp(scores, axis=1)
            message -= np.max(message)
            log_plus[time_index] = message
        max_err = 0.0
        for time_index in range(n_times):
            log_mu = log_h[time_index] + log_minus[time_index] + log_plus[time_index]
            log_mu -= logsumexp(log_mu)
            mu = np.clip(np.exp(log_mu), 1e-300, 1.0)
            mu /= mu.sum()
            log_h[time_index] = log_h[time_index] + eta * (np.log(target) - np.log(mu))
            max_err = max(max_err, float(np.max(np.abs(mu - target))))
        if max_err < tol:
            break
    else:
        used = max_iter
    # Recompute messages at the accepted scalings for the pairwise plans.
    log_minus = [np.zeros(n) for _ in range(n_times)]
    for time_index in range(n_times - 1):
        scores = log_kernels[time_index] + (log_h[time_index] + log_minus[time_index])[:, None]
        scores -= np.max(scores, axis=0, keepdims=True)
        message = logsumexp(scores, axis=0)
        message -= np.max(message)
        log_minus[time_index + 1] = message
    log_plus = [np.zeros(n) for _ in range(n_times)]
    for time_index in reversed(range(n_times - 1)):
        scores = log_kernels[time_index] + (log_h[time_index + 1] + log_plus[time_index + 1])[None, :]
        scores -= np.max(scores, axis=1, keepdims=True)
        message = logsumexp(scores, axis=1)
        message -= np.max(message)
        log_plus[time_index] = message
    plans = []
    for time_index in range(n_times - 1):
        log_plan = (log_kernels[time_index]
                    + (log_h[time_index] + log_minus[time_index])[:, None]
                    + (log_h[time_index + 1] + log_plus[time_index + 1])[None, :])
        for _ in range(3):
            log_plan += np.log(target)[:, None] - logsumexp(log_plan, axis=1, keepdims=True)
            log_plan += np.log(target)[None, :] - logsumexp(log_plan, axis=0, keepdims=True)
        log_plan -= np.max(log_plan)
        plan = np.exp(log_plan)
        plan /= np.maximum(plan.sum(axis=1, keepdims=True), 1e-300)
        plan *= target[:, None]
        plan /= np.maximum(plan.sum(axis=0, keepdims=True), 1e-300)
        plan *= target[None, :]
        plan = np.clip(plan, 0.0, None)
        plan /= np.maximum(plan.sum(axis=1, keepdims=True), 1e-300)
        plan *= target[:, None]
        plans.append(plan)
    paths = np.zeros((n, n_times, dimension), dtype=np.float64)
    cursor = rng.choice(n, size=n, p=target)
    paths[:, 0] = marginals[0][cursor]
    for time_index, plan in enumerate(plans):
        nxt = np.empty(n, dtype=int)
        for row_index, row in enumerate(cursor):
            probs = plan[int(row)]
            total = float(probs.sum())
            if not np.isfinite(total) or total <= 0.0:
                nxt[row_index] = int(rng.integers(n))
            else:
                nxt[row_index] = int(rng.choice(n, p=probs / total))
        cursor = nxt
        paths[:, time_index + 1] = marginals[time_index + 1][cursor]
    return {"paths": paths, "iterations": used, "marginal_error": max_err}


class PotentialMLP(nn.Module):
    """Scalar SiLU potential. ``depth`` is the number of hidden layers.

    Input shape (n, d), potential shape (n,), drift shape (n, d).
    """

    def __init__(self, dimension: int, width: int, depth: int) -> None:
        super().__init__()
        sizes = [dimension] + [width] * int(depth) + [1]
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Map (n, d) to (n,)."""
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < len(self.layers) - 1:
                hidden = torch.nn.functional.silu(hidden)
        return hidden.squeeze(-1)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """``-grad phi``. ``x`` has shape (n, d) and must allow gradients."""
        phi = self.potential(x).sum()
        gradient, = torch.autograd.grad(phi, x, create_graph=True)
        return -gradient


def drift_numpy(model: PotentialMLP, x: np.ndarray) -> np.ndarray:
    """Evaluate ``-grad phi`` with autograd. Shape (n, d) in, (n, d) out."""
    model.eval()
    values = torch.tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE, requires_grad=True)
    phi = model.potential(values).sum()
    gradient, = torch.autograd.grad(phi, values, create_graph=False)
    return (-gradient).detach().cpu().numpy()


def fit_potential(paths: np.ndarray, dt: float, cfg: dict[str, Any], seed: int) -> tuple[PotentialMLP, float]:
    """Adam on ``mean ||b_theta(x) - dx/dt||^2``. A new network each call.

    ``paths`` has shape (n, T, d). Returns the network and the final batch-mean loss.
    """
    model_cfg = cfg["model"]
    left = paths[:, :-1, :].reshape(-1, paths.shape[-1])
    target = ((paths[:, 1:, :] - paths[:, :-1, :]) / float(dt)).reshape(-1, paths.shape[-1])
    if not (np.isfinite(left).all() and np.isfinite(target).all()):
        raise ValueError("Non-finite Schrödinger-bridge paths")
    torch.manual_seed(int(seed))
    model = PotentialMLP(left.shape[1], int(model_cfg["width"]), int(model_cfg["depth"])).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["lr"]))
    rng = np.random.default_rng(int(seed))
    batch = int(model_cfg["batch_size"])
    last = float("nan")
    model.train()
    n = len(left)
    for _ in range(int(model_cfg["epochs"])):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            index = order[start:start + batch]
            xb = torch.tensor(left[index], dtype=DTYPE, requires_grad=True)
            yb = torch.tensor(target[index], dtype=DTYPE)
            prediction = model.drift(xb)
            loss = torch.mean((prediction - yb) ** 2)
            if not torch.isfinite(loss):
                return model, float("nan")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
    return model, last


def sigma2_mle(paths: np.ndarray, dt: float, drift_fn: DriftFn) -> float:
    """Isotropic MLE ``mean(||dx - b dt||^2) / dt``."""
    increment = paths[:, 1:, :] - paths[:, :-1, :]
    flat = paths[:, :-1, :].reshape(-1, paths.shape[-1])
    drift = drift_fn(flat).reshape(increment.shape)
    residual = increment - float(dt) * drift
    return float(np.mean(residual ** 2) / float(dt))


def zero_drift(x: np.ndarray) -> np.ndarray:
    """Reference drift of the first outer iteration. Shape of ``x``."""
    return np.zeros_like(np.asarray(x, dtype=np.float64))


def run_nn_appex(states: np.ndarray, cfg: dict[str, Any], fix_diffusion: bool,
                 label: str) -> dict[str, Any]:
    """Algorithm 1 with the released IPF and potential network.

    The last finite drift is kept. A non-finite step stops the loop.
    """
    dt = float(cfg["data"]["dt"])
    sigma2 = diffusion_prior(cfg)
    drift_fn: DriftFn = zero_drift
    model: PotentialMLP | None = None
    trace = []
    stopped = False
    for outer in range(int(cfg["model"]["outer_iterations"])):
        started = time.perf_counter()
        rng = np.random.default_rng(int(cfg["seeds"]["sinkhorn"]) + 1009 * outer + (17 if fix_diffusion else 0))
        bridge = mmos_sinkhorn(states, dt, drift_fn, sigma2, rng,
                               int(cfg["model"]["sinkhorn_iterations"]),
                               float(cfg["model"]["sinkhorn_tol"]))
        seed = int(cfg["seeds"]["network"]) + 1000 * (outer + 1)
        candidate, loss = fit_potential(bridge["paths"], dt, cfg, seed)
        if not math.isfinite(loss):
            stopped = True
            trace.append({"outer": outer + 1, "sigma2": sigma2, "loss": loss, "stopped": True,
                          "seconds": time.perf_counter() - started})
            break
        candidate_drift = lambda x, net=candidate: drift_numpy(net, x)  # noqa: E731
        sigma_new = sigma2 if fix_diffusion else sigma2_mle(bridge["paths"], dt, candidate_drift)
        if not math.isfinite(sigma_new) or sigma_new <= 0.0:
            stopped = True
            trace.append({"outer": outer + 1, "sigma2": sigma2, "loss": loss, "stopped": True,
                          "seconds": time.perf_counter() - started})
            break
        model = candidate
        drift_fn = candidate_drift
        sigma2 = max(float(sigma_new), 1e-8)
        trace.append({
            "outer": outer + 1,
            "sigma2": sigma2,
            "loss": loss,
            "sinkhorn_iterations": bridge["iterations"],
            "sinkhorn_marginal_error": bridge["marginal_error"],
            "stopped": False,
            "seconds": time.perf_counter() - started,
        })
        print(f"[{label}] outer {outer + 1}: sigma2={sigma2:.6g} loss={loss:.6g} "
              f"sinkhorn={bridge['iterations']} err={bridge['marginal_error']:.3g} "
              f"({trace[-1]['seconds']:.1f}s)", flush=True)
    if model is None:
        raise RuntimeError(f"{label} produced no finite drift")
    return {"model": model, "drift": drift_fn, "sigma2": float(sigma2),
            "sigma": true_sigma(float(sigma2)), "trace": trace, "stopped_early": stopped,
            "fix_diffusion": fix_diffusion}


def query_drift(drift_fn: DriftFn, frames: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Teacher drift on each split's observed states. Shape (n_particles * T, d)."""
    return {name: drift_fn(frame["query"]) for name, frame in frames.items()}


def constant_diffusion_target(query: np.ndarray, sigma: float) -> np.ndarray:
    """Constant isotropic scale at every query. Shape (n, d)."""
    return np.full((len(query), query.shape[1]), float(sigma), dtype=np.float64)


class GradientPolynomial:
    """Drift ``b = -grad phi`` for a polynomial potential in standardized coordinates."""

    def __init__(self, powers: np.ndarray, coefficients: np.ndarray,
                 mean: np.ndarray, scale: np.ndarray, metadata: dict[str, Any]) -> None:
        self.powers = powers
        self.coefficients = coefficients
        self.mean = mean
        self.scale = scale
        self.metadata = metadata

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Drift at states (n, d). Returns (n, d)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.scale
        design = _gradient_design(z, self.powers, self.scale)
        flat = design @ self.coefficients
        return flat.reshape(len(x), x.shape[1])

    def as_polynomial_field(self) -> PolynomialField:
        """Scalar potential, constant term included with coefficient 0, for expansion."""
        full_powers = monomial_powers(self.mean.shape[0], int(self.metadata["potential_degree"]))
        table = {tuple(power): 0.0 for power in full_powers}
        for power, coef in zip(self.powers, self.coefficients):
            table[tuple(int(v) for v in power)] = float(coef)
        coefficients = np.array([[table[tuple(power)] for power in full_powers]])
        return PolynomialField(full_powers, coefficients, self.mean, self.scale, self.metadata)


def _gradient_design(z: np.ndarray, powers: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Rows are sample-major components. Column ``alpha`` is ``-d(z^alpha)/dx``."""
    n, dimension = z.shape
    design = np.zeros((n * dimension, len(powers)), dtype=np.float64)
    for column, power in enumerate(powers):
        for axis in range(dimension):
            if int(power[axis]) == 0:
                continue
            reduced = power.copy()
            reduced[axis] -= 1
            monomial = np.ones(n, dtype=np.float64)
            for coord, exponent in enumerate(reduced):
                if int(exponent):
                    monomial *= z[:, coord] ** int(exponent)
            design[axis::dimension, column] = -int(power[axis]) * monomial / scale[axis]
    return design


def fit_gradient_field(x: np.ndarray, y: np.ndarray, degree: int, threshold: float,
                       ridge: float) -> GradientPolynomial:
    """STLSQ for a polynomial potential whose drift has total degree ``degree``.

    States (n, d), targets (n, d). Standardization uses ``x`` only.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    powers = monomial_powers(x.shape[1], degree + 1)
    powers = powers[powers.sum(axis=1) > 0]
    design = _gradient_design(z, powers, scale)
    target = y.reshape(-1)
    feature_rms = np.sqrt(np.mean(design ** 2, axis=0))
    feature_rms = np.where(feature_rms < 1e-12, 1.0, feature_rms)
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    if target_rms < 1e-12:
        target_rms = 1.0
    coef_n = _stlsq_single(design / feature_rms, target / target_rms, threshold, ridge)
    coefficients = coef_n * target_rms / feature_rms
    metadata = {"degree": degree, "potential_degree": degree + 1, "threshold": threshold,
                "ridge": ridge, "structure": "gradient",
                "library_size": int(len(powers)),
                "active_terms": int(np.count_nonzero(coefficients)),
                "optimizer": "numpy STLSQ on -grad of a polynomial potential"}
    return GradientPolynomial(powers, coefficients, mean, scale, metadata)


def differentiate_poly(poly: dict[tuple[int, ...], float], axis: int) -> dict[tuple[int, ...], float]:
    """Exact derivative of one expanded polynomial along ``axis``."""
    out: dict[tuple[int, ...], float] = {}
    for exponents, coef in poly.items():
        power = int(exponents[axis])
        if power == 0:
            continue
        reduced = list(exponents)
        reduced[axis] = power - 1
        key = tuple(reduced)
        out[key] = out.get(key, 0.0) + coef * power
    return {key: value for key, value in out.items() if abs(value) > 1e-12}


def expand_gradient(field: GradientPolynomial) -> list[dict[tuple[int, ...], float]]:
    """Original-coordinate drift of ``b = -grad phi``."""
    potential = expand_to_original(field.as_polynomial_field())[0]
    expanded = []
    for axis in range(len(field.mean)):
        derivative = differentiate_poly(potential, axis)
        expanded.append({key: -value for key, value in derivative.items() if abs(value) > 1e-12})
    return expanded


def select_drift(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Validation NRMSE. Ties: lower degree, lower threshold, then gradient structure."""
    ridge = float(cfg["distill"]["ridge"])
    best = None
    best_key = None
    best_meta: dict[str, Any] = {}
    candidates = []
    for degree in cfg["distill"]["degrees"]:
        for threshold in cfg["distill"]["thresholds"]:
            for structure in ("gradient", "free"):
                if structure == "gradient":
                    field = fit_gradient_field(x_train, y_train, int(degree), float(threshold), ridge)
                    rank = 0
                else:
                    field = fit_polynomial_field(x_train, y_train, int(degree), float(threshold), ridge)
                    rank = 1
                score = nrmse(y_val, field.predict(x_val))
                record = {"degree": int(degree), "threshold": float(threshold), "structure": structure,
                          "score": score, "active_terms": int(field.metadata["active_terms"])}
                candidates.append(record)
                key = (score, int(degree), float(threshold), rank)
                if best_key is None or key < best_key:
                    best_key = key
                    best = field
                    best_meta = record
    if best is None:
        raise RuntimeError("no drift candidate")
    best_meta = dict(best_meta)
    best_meta["candidates"] = candidates
    return best, best_meta


def select_diffusion(x_train: np.ndarray, sigma_train: float, x_val: np.ndarray,
                     sigma_val: float, cfg: dict[str, Any]) -> tuple[PolynomialField, dict[str, Any]]:
    """STLSQ of the constant diffusion teacher. Validation NRMSE, then smaller degree."""
    y_train = constant_diffusion_target(x_train, sigma_train)
    y_val = constant_diffusion_target(x_val, sigma_val)
    ridge = float(cfg["distill"]["ridge"])
    best = None
    best_key = None
    best_meta: dict[str, Any] = {}
    candidates = []
    for degree in cfg["distill"]["degrees"]:
        for threshold in cfg["distill"]["thresholds"]:
            field = fit_polynomial_field(x_train, y_train, int(degree), float(threshold), ridge)
            score = nrmse(y_val, field.predict(x_val))
            record = {"degree": int(degree), "threshold": float(threshold), "score": score,
                      "active_terms": int(np.count_nonzero(field.coefficients))}
            candidates.append(record)
            key = (score, int(degree), float(threshold))
            if best_key is None or key < best_key:
                best_key = key
                best = field
                best_meta = record
    if best is None:
        raise RuntimeError("no diffusion candidate")
    best_meta = dict(best_meta)
    best_meta["candidates"] = candidates
    return best, best_meta


def vif_max(x: np.ndarray, degree: int) -> float:
    """Max VIF of a polynomial library. The constant column is excluded."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    powers = monomial_powers(x.shape[1], degree)
    theta = evaluate_powers((x - mean) / scale, powers)
    keep = [j for j, power in enumerate(powers) if int(power.sum()) > 0]
    theta = theta[:, keep]
    worst = 1.0
    for j in range(theta.shape[1]):
        others = np.delete(theta, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, theta[:, j], rcond=None)
        residual = theta[:, j] - others @ coef
        total = float(np.sum((theta[:, j] - theta[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual ** 2)) / max(total, 1e-15)
        worst = max(worst, 1.0 / max(1.0 - r2, 1e-9))
    return float(worst)


def drift_pack(prediction: np.ndarray, teacher: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """NRMSE and residual RMS for one state cloud."""
    return {"sym_teacher_nrmse": nrmse(teacher, prediction),
            "teacher_truth_nrmse": nrmse(truth, teacher),
            "sym_truth_nrmse": nrmse(truth, prediction),
            "sym_teacher_rms": rms(prediction - teacher),
            "teacher_truth_rms": rms(teacher - truth),
            "sym_truth_rms": rms(prediction - truth),
            "teacher_rms": rms(teacher),
            "truth_rms": rms(truth)}


def cosine_similarity(prediction: np.ndarray, truth: np.ndarray, floor: float = 1e-8) -> dict[str, float]:
    """Mean cosine on points where both norms exceed ``floor``."""
    pred_norm = np.linalg.norm(prediction, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    mask = (pred_norm > floor) & (truth_norm > floor)
    if not mask.any():
        return {"cosine": float("nan"), "fraction_used": 0.0}
    dots = np.sum(prediction[mask] * truth[mask], axis=1) / (pred_norm[mask] * truth_norm[mask])
    return {"cosine": float(np.mean(dots)), "fraction_used": float(mask.mean())}


def normalized_absolute(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Paper's pointwise ``||error|| / ||truth||``. Median is the stable summary."""
    denom = np.linalg.norm(truth, axis=1)
    numer = np.linalg.norm(prediction - truth, axis=1)
    ratio = numer / np.maximum(denom, 1e-8)
    return {"median": float(np.median(ratio)), "mean": float(np.mean(ratio))}


def shuffle_sensitivity(drift_fn: DriftFn, x: np.ndarray, rng: np.random.Generator) -> float:
    """Relative RMS when the second coordinate is permuted independently."""
    base = drift_fn(x)
    shuffled = x.copy()
    shuffled[:, 1] = rng.permutation(shuffled[:, 1])
    changed = drift_fn(shuffled)
    return float(rms(changed - base) / max(rms(base), 1e-12))


def partial_rms(drift_fn: DriftFn, x: np.ndarray, step: float = 1e-4) -> dict[str, float]:
    """Finite-difference RMS of ``df1/dy`` and ``df2/dx``."""
    x = np.asarray(x, dtype=np.float64)
    base = drift_fn(x)
    y_up = x.copy()
    y_up[:, 1] += step
    x_up = x.copy()
    x_up[:, 0] += step
    df1_dy = (drift_fn(y_up)[:, 0] - base[:, 0]) / step
    df2_dx = (drift_fn(x_up)[:, 1] - base[:, 1]) / step
    return {"df1_dy_rms": float(np.sqrt(np.mean(df1_dy ** 2))),
            "df2_dx_rms": float(np.sqrt(np.mean(df2_dx ** 2))),
            "drift_rms": rms(base)}


def gaussian_nll(increment: np.ndarray, drift: np.ndarray, sigma: float, dt: float) -> float:
    """Mean Euler-Gauss NLL of one increment. ``increment`` and ``drift`` are (n, d)."""
    residual = increment - drift * dt
    variance = (float(sigma) ** 2) * float(dt)
    dimension = increment.shape[1]
    quad = np.sum(residual ** 2, axis=1) / variance
    const = dimension * math.log(2.0 * math.pi * variance)
    return float(np.mean(0.5 * (quad + const)))


def failure_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Counts from ``simulate_em``. Failed trajectories stay in the rate."""
    classes = result["failure_class"]
    return {"failure_rate": float(result["failure_rate"]),
            "non_finite": int(sum(item == "non_finite" for item in classes)),
            "outside_radius": int(sum(item == "outside_radius" for item in classes)),
            "n": int(len(classes))}


def path_rmse(left: np.ndarray, right: np.ndarray, failed: np.ndarray) -> float:
    """RMSE on trajectories that both models kept. NaN if none remain."""
    ok = ~failed
    if not ok.any():
        return float("nan")
    return rms(left[ok] - right[ok])


def terminal_cov_relative(left: np.ndarray, right: np.ndarray, failed: np.ndarray) -> float:
    """Frobenius relative error of the terminal covariance. ``right`` is the reference."""
    ok = ~failed
    if int(ok.sum()) < 3:
        return float("nan")
    cov_left = np.cov(left[ok, -1], rowvar=False)
    cov_right = np.cov(right[ok, -1], rowvar=False)
    return float(np.linalg.norm(cov_left - cov_right) / max(np.linalg.norm(cov_right), 1e-12))


def mean_path_rmse(left: np.ndarray, right: np.ndarray, failed: np.ndarray) -> float:
    """RMSE of the survivor-averaged path. NaN if every trajectory failed."""
    ok = ~failed
    if not ok.any():
        return float("nan")
    return rms(left[ok].mean(axis=0) - right[ok].mean(axis=0))


def run_simulation(drift_fn: DriftFn, sigma: float | DriftFn, x0: np.ndarray,
                   increments: np.ndarray, dt: float, radius: float) -> dict[str, Any]:
    """Paired Euler-Maruyama. ``sigma`` is a scalar or a state-dependent scale."""
    def drift(x: np.ndarray) -> np.ndarray:
        return drift_fn(x)

    def diffusion(x: np.ndarray) -> np.ndarray:
        if callable(sigma):
            return np.asarray(sigma(x), dtype=np.float64)
        return np.full_like(x, float(sigma))

    return simulate_em(drift, diffusion, x0, dt, increments.shape[1], increments, radius)


def grid_points(low: float, high: float, size: int, dimension: int = 2) -> np.ndarray:
    """Axis-aligned product grid. Shape (size**d, d)."""
    axis = np.linspace(low, high, size)
    mesh = np.stack(np.meshgrid(*([axis] * dimension), indexing="ij"), axis=-1)
    return mesh.reshape(-1, dimension)


def degree_ratio(expanded: list[dict[tuple[int, ...], float]], x: np.ndarray, degree: int) -> float:
    """RMS of the exact degree-``degree`` part over RMS of the full expanded field."""
    part = [{exp: coef for exp, coef in poly.items() if sum(exp) == degree} for poly in expanded]
    return float(rms(evaluate_expanded(part, x)) / max(rms(evaluate_expanded(expanded, x)), 1e-12))


def cross_ratio(expanded: list[dict[tuple[int, ...], float]], x: np.ndarray) -> float:
    """RMS of terms that use both coordinates, over RMS of the full field."""
    part = []
    for poly in expanded:
        kept = {exp: coef for exp, coef in poly.items() if sum(item > 0 for item in exp) >= 2}
        part.append(kept)
    return float(rms(evaluate_expanded(part, x)) / max(rms(evaluate_expanded(expanded, x)), 1e-12))


def cubic_ratio(x: np.ndarray) -> float:
    """RMS of the cubic part of the true drift over RMS of the full true drift."""
    cubic = -2.0 * np.asarray(x, dtype=np.float64) ** 3
    full = true_drift(x)
    return float(rms(cubic) / max(rms(full), 1e-12))


def step_ratio(x: np.ndarray, dt: float, sigma: float) -> float:
    """Mean ``||b|| dt / (sigma sqrt(dt))`` of the true drift."""
    norms = np.linalg.norm(true_drift(x), axis=1)
    return float(np.mean(norms) * dt / (sigma * math.sqrt(dt)))


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Thresholds are read from ``cfg`` and not modified."""
    hyp = cfg["hypotheses"]
    h1, h2 = hyp["H1_drift"], hyp["H2_diffusion"]
    drift = metrics["drift_test"]
    verdicts: dict[str, Any] = {
        "H1": _pass(drift["sym_teacher_nrmse"] <= h1["symbolic_vs_teacher_test_nrmse_max"]
                    and drift["teacher_truth_nrmse"] <= h1["teacher_vs_truth_test_nrmse_max"]
                    and drift["sym_truth_nrmse"] <= h1["symbolic_vs_truth_test_nrmse_max"]),
        "H2_distill": _pass(metrics["sigma_test_nrmse"] <= h2["symbolic_vs_teacher_test_nrmse_max"]
                            and metrics["sigma_nonpositive_fraction"] <= h2["fraction_nonpositive_max"]),
        "H2_truth": _pass(metrics["sigma2_relative_error"] <= h2["sigma2_relative_error_max"]),
        "H2_local_shadow": bool(metrics["sigma_grid_nrmse"]
                                > h2["grid_over_ontraj_factor"] * max(metrics["sigma_test_nrmse"], 1e-8)),
        "H3": ("indeterminate" if metrics["vif_degree3"] > hyp["H3_collinearity_gate"]["vif_max"]
               else "pass"),
        "H3_closed_form": False,
    }
    verdicts["H4"] = _simulation_verdict(
        hyp["H4_simulation"], metrics["fail_truth"], metrics["fail_teacher"], metrics["fail_symbolic"],
        metrics["paired_sym_teacher"], metrics["paired_teacher_truth"],
        metrics["var_sym_teacher"], metrics["var_teacher_truth"], drift["teacher_truth_nrmse"])
    verdicts["H5"] = _law_verdict(hyp["H5_law"], metrics)
    h6 = hyp["H6_shuffle_and_grid"]
    verdicts["H6"] = _pass(
        abs(metrics["shuffle_teacher"] - metrics["shuffle_truth"]) <= h6["shuffle_abs_gap_max"]
        and abs(metrics["shuffle_symbolic"] - metrics["shuffle_truth"]) <= h6["shuffle_abs_gap_max"]
        and metrics["grid_teacher_truth_nrmse"] <= h6["teacher_vs_truth_grid_nrmse_max"]
        and metrics["grid_sym_teacher_nrmse"] <= h6["symbolic_vs_teacher_grid_nrmse_max"])
    verdicts["H6_amplified"] = bool(
        metrics["grid_teacher_truth_rms"] > h6["amplification_factor"] * max(drift["teacher_truth_rms"], 1e-12))
    h7 = hyp["H7_discretization"]
    verdicts["H7"] = _pass(math.isfinite(metrics["disc_rmse"])
                           and metrics["disc_rmse"] <= h7["terminal_rmse_max"]
                           and metrics["disc_retained"] >= h7["min_retained_fraction"])
    h8 = hyp["H8_fixed_diffusion"]
    verdicts["H8"] = _pass(metrics["fixed_teacher_truth_nrmse"] - drift["teacher_truth_nrmse"]
                           >= h8["adaptive_improvement_min"]
                           and drift["teacher_truth_nrmse"] <= h8["adaptive_nrmse_max"])
    return verdicts


def _simulation_verdict(gate: dict[str, Any], fail_truth: float, fail_teacher: float,
                        fail_model: float, paired_model: float, paired_truth: float,
                        var_model: float, var_truth: float, teacher_truth: float) -> str:
    worst = max(fail_truth, fail_teacher, fail_model)
    if worst > gate["failure_rate_max"] or abs(fail_model - fail_teacher) > gate["failure_gap_max"]:
        return "indeterminate"
    if not all(math.isfinite(value) for value in (paired_model, paired_truth, var_model, var_truth)):
        return "indeterminate"
    ok = (paired_model <= gate["path_rmse_symbolic_vs_teacher_max"]
          and paired_truth <= gate["path_rmse_teacher_vs_truth_max"]
          and var_model <= gate["terminal_cov_symbolic_vs_teacher_max"]
          and var_truth <= gate["terminal_cov_teacher_vs_truth_max"]
          and teacher_truth <= gate["drift_teacher_vs_truth_max"])
    return _pass(ok)


def _law_verdict(gate: dict[str, Any], metrics: dict[str, Any]) -> str:
    worst = max(metrics["law_fail_truth"], metrics["law_fail_teacher"], metrics["law_fail_symbolic"])
    if worst > gate["failure_rate_max"]:
        return "indeterminate"
    if (not math.isfinite(metrics["floor_mean"]) or metrics["floor_mean"] > gate["mean_path_rmse_teacher_max"]
            or not math.isfinite(metrics["floor_var"]) or metrics["floor_var"] > gate["terminal_cov_teacher_max"]):
        return "indeterminate"
    ok = (metrics["law_mean_teacher"] <= gate["mean_path_rmse_teacher_max"]
          and metrics["law_mean_symbolic"] <= gate["mean_path_rmse_symbolic_max"]
          and metrics["law_var_teacher"] <= gate["terminal_cov_teacher_max"]
          and metrics["law_var_symbolic"] <= gate["terminal_cov_symbolic_max"])
    return _pass(ok)


def pre_registration_scales(seed: int = 18099) -> dict[str, Any]:
    """Truth-only scales on a probe seed. Does not touch the evaluation seed."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    population = generate_population(cfg, rng)
    frames = split_population(population, cfg)
    query = frames["train"]["query"]
    sigma = population["sigma"]
    dt = float(cfg["data"]["dt"])
    oracle = fit_polynomial_field(frames["train"]["query"], true_drift(frames["train"]["query"]),
                                  3, 0.0, float(cfg["distill"]["ridge"]))
    pure = fit_polynomial_field(frames["train"]["query"], true_drift(frames["train"]["query"]), 3, 0.0, 0.0)
    hold = frames["val"]["query"]
    x0 = frames["val"]["x0"]
    increments = frames["val"]["increments"]
    radius = float(cfg["data"]["failure_radius"])
    truth = run_simulation(true_drift, sigma, x0, increments, dt, radius)
    bias = lambda x: true_drift(x) + 2.0  # noqa: E731
    biased = run_simulation(bias, sigma, x0, increments, dt, radius)
    failed = truth["failed"] | biased["failed"]
    law_a = _fresh_truth(cfg, population["gmm"], np.random.default_rng(seed + 1))
    law_b = _fresh_truth(cfg, population["gmm"], np.random.default_rng(seed + 2))
    n_law = min(len(law_a["states"]), len(law_b["states"]))
    failed_law = np.zeros(n_law, dtype=bool)
    coarse_noise = (increments[:, 0, :] + increments[:, 1, :]) / math.sqrt(2.0)
    coarse_inc = coarse_noise[:, None, :]
    fine = truth
    coarse = run_simulation(true_drift, sigma, x0, coarse_inc, 2.0 * dt, radius)
    both = ~(fine["failed"] | coarse["failed"])
    sigma2_hat = sigma2_mle(
        np.concatenate([frames["train"]["states"], frames["val"]["states"]], axis=0), dt, true_drift)
    scales = {
        "probe_seed": seed,
        "discarded": population["discarded"],
        "gmm_components": population["gmm"]["n_components"],
        "state_min": frames["train"]["states"].min(axis=(0, 1)).tolist(),
        "state_max": frames["train"]["states"].max(axis=(0, 1)).tolist(),
        "drift_rms": rms(true_drift(query)),
        "step_ratio": step_ratio(query, dt, sigma),
        "cubic_ratio": cubic_ratio(query),
        "vif_degree3": vif_max(query, 3),
        "oracle_ridge_val_nrmse": nrmse(true_drift(hold), oracle.predict(hold)),
        "oracle_pure_val_nrmse": nrmse(true_drift(hold), pure.predict(hold)),
        "paired_bias2_rmse": path_rmse(biased["states"], truth["states"], failed),
        "sigma2_mle_true_paths": sigma2_hat,
        "floor_mean": mean_path_rmse(law_a["states"][:n_law], law_b["states"][:n_law], failed_law),
        "floor_var": terminal_cov_relative(law_a["states"][:n_law], law_b["states"][:n_law], failed_law),
        "disc_rmse": rms(fine["states"][both, -1] - coarse["states"][both, -1]) if both.any() else float("nan"),
        "diffusion_prior_eval_seed": diffusion_prior(cfg),
        "prior_relative_error": abs(diffusion_prior(cfg) - float(cfg["data"]["sigma2"])) / float(cfg["data"]["sigma2"]),
        "shuffle_truth": shuffle_sensitivity(true_drift, hold, np.random.default_rng(seed + 7)),
        "n_law": int(cfg["data"]["n_law"]),
    }
    print(json.dumps(scales, indent=2))
    return scales


def _fresh_truth(cfg: dict[str, Any], params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    n = int(cfg["data"]["n_law"])
    x0 = sample_gmm_params(params, n, rng)
    sigma = true_sigma(float(cfg["data"]["sigma2"]))
    return simulate_known(x0, float(cfg["data"]["dt"]), int(cfg["data"]["n_steps"]), sigma, rng,
                          float(cfg["data"]["generation_radius"]))


def _save_weights(model: PotentialMLP, path: Path) -> None:
    payload = {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
    path.write_text(json.dumps(payload))


def _field_expand(field: Any) -> list[dict[tuple[int, ...], float]]:
    if isinstance(field, GradientPolynomial):
        return expand_gradient(field)
    return expand_to_original(field)


def make_figure(grid: np.ndarray, truth: np.ndarray, teacher: np.ndarray, path: Path) -> None:
    """Grid error of the teacher drift. Not used for model selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = int(math.sqrt(len(grid)))
    error = np.linalg.norm(teacher - truth, axis=1).reshape(size, size)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    image = axes[0].imshow(error.T, origin="lower", extent=(grid[:, 0].min(), grid[:, 0].max(),
                                                            grid[:, 1].min(), grid[:, 1].max()))
    axes[0].set_title("||teacher drift - truth||")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].scatter(truth[:, 0], teacher[:, 0], s=12, alpha=0.8)
    axes[1].set_xlabel("true drift, component x")
    axes[1].set_ylabel("teacher drift, component x")
    axes[1].set_title("grid, not used for selection")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(cfg: dict[str, Any], population: dict[str, Any], frames: dict[str, dict[str, np.ndarray]],
             adaptive: dict[str, Any], fixed: dict[str, Any],
             drift_field: Any, drift_meta: dict[str, Any],
             sigma_field: PolynomialField, sigma_meta: dict[str, Any]) -> dict[str, Any]:
    """Score the frozen teacher once. This is the only test-set pass."""
    dt = float(cfg["data"]["dt"])
    sigma_true = float(population["sigma"])
    sigma_hat = float(adaptive["sigma"])
    teacher = query_drift(adaptive["drift"], frames)
    fixed_drift = query_drift(fixed["drift"], frames)
    truth = {name: true_drift(frame["query"]) for name, frame in frames.items()}
    symbolic = {name: drift_field.predict(frame["query"]) for name, frame in frames.items()}
    test_query = frames["test"]["query"]
    sigma_pred = sigma_field.predict(test_query)
    grid = grid_points(float(cfg["distill"]["grid_low"]), float(cfg["distill"]["grid_high"]),
                       int(cfg["distill"]["grid_size"]))
    grid_truth = true_drift(grid)
    grid_teacher = adaptive["drift"](grid)
    grid_symbolic = drift_field.predict(grid)
    grid_sigma = sigma_field.predict(grid)
    expanded = _field_expand(drift_field)
    expanded_sigma = expand_to_original(sigma_field)
    check = evaluate_expanded(expanded, test_query)
    if rms(check - symbolic["test"]) > 1e-6:
        raise RuntimeError("Expanded drift does not match the fitted field")
    oracle_teacher = fit_polynomial_field(frames["train"]["query"], teacher["train"], 3, 0.0,
                                          float(cfg["distill"]["ridge"]))
    oracle_truth = fit_polynomial_field(frames["train"]["query"], truth["train"], 3, 0.0, 0.0)
    rng_shuffle = np.random.default_rng(int(cfg["seeds"]["shuffle"]))
    partial_teacher = partial_rms(adaptive["drift"], test_query)
    partial_symbolic = partial_rms(drift_field.predict, test_query)
    partial_truth = partial_rms(true_drift, test_query)
    increments = frames["test"]["increments"]
    x0 = frames["test"]["x0"]
    radius = float(cfg["data"]["failure_radius"])
    sim_truth = run_simulation(true_drift, sigma_true, x0, increments, dt, radius)
    sim_teacher = run_simulation(adaptive["drift"], sigma_hat, x0, increments, dt, radius)
    sim_symbolic = run_simulation(drift_field.predict, sigma_field.predict, x0, increments, dt, radius)
    reproduce = rms(sim_truth["states"] - frames["test"]["states"])
    if reproduce > 1e-8:
        raise RuntimeError(f"Truth resimulation drifted from the stored paths ({reproduce})")
    failed_pair = sim_truth["failed"] | sim_teacher["failed"] | sim_symbolic["failed"]
    coarse_noise = (increments[:, 0, :] + increments[:, 1, :]) / math.sqrt(2.0)
    coarse = run_simulation(adaptive["drift"], sigma_hat, x0, coarse_noise[:, None, :], 2.0 * dt, radius)
    fine_terminal_ok = ~(sim_teacher["failed"] | coarse["failed"])
    disc = (rms(sim_teacher["states"][fine_terminal_ok, -1] - coarse["states"][fine_terminal_ok, -1])
            if fine_terminal_ok.any() else float("nan"))
    law_rng = {
        "truth": np.random.default_rng(int(cfg["seeds"]["law_truth"])),
        "floor": np.random.default_rng(int(cfg["seeds"]["law_floor"])),
        "teacher": np.random.default_rng(int(cfg["seeds"]["law_teacher"])),
        "symbolic": np.random.default_rng(int(cfg["seeds"]["law_symbolic"])),
    }
    law = {}
    for name, drift_fn, sigma in (
        ("truth", true_drift, sigma_true),
        ("floor", true_drift, sigma_true),
        ("teacher", adaptive["drift"], sigma_hat),
        ("symbolic", drift_field.predict, sigma_field.predict),
    ):
        drawn = sample_gmm_params(population["gmm"], int(cfg["data"]["n_law"]), law_rng[name])
        noise = law_rng[name].standard_normal((len(drawn), int(cfg["data"]["n_steps"]), drawn.shape[1]))
        law[name] = run_simulation(drift_fn, sigma, drawn, noise, dt, radius)
    nll_teacher = _path_nll(frames["test"], adaptive["drift"], sigma_hat, dt)
    nll_truth = _path_nll(frames["test"], true_drift, sigma_true, dt)
    metrics: dict[str, Any] = {
        "drift_test": drift_pack(symbolic["test"], teacher["test"], truth["test"]),
        "drift_val": drift_pack(symbolic["val"], teacher["val"], truth["val"]),
        "sigma_test_nrmse": nrmse(constant_diffusion_target(test_query, sigma_hat), sigma_pred),
        "sigma_test_rms": rms(sigma_pred - sigma_hat),
        "sigma_nonpositive_fraction": float(np.mean(sigma_pred <= 0.0)),
        "sigma_grid_nrmse": nrmse(np.full_like(grid_sigma, sigma_hat), grid_sigma),
        "sigma2_hat": adaptive["sigma2"],
        "sigma2_fixed": fixed["sigma2"],
        "sigma2_true": float(cfg["data"]["sigma2"]),
        "sigma2_relative_error": abs(adaptive["sigma2"] - float(cfg["data"]["sigma2"])) / float(cfg["data"]["sigma2"]),
        "vif_degree3": vif_max(frames["train"]["query"], 3),
        "oracle_teacher_nrmse": nrmse(teacher["test"], oracle_teacher.predict(test_query)),
        "oracle_truth_nrmse": nrmse(truth["test"], oracle_truth.predict(test_query)),
        "fixed_teacher_truth_nrmse": nrmse(truth["test"], fixed_drift["test"]),
        "fixed_teacher_truth_rms": rms(fixed_drift["test"] - truth["test"]),
        "shuffle_teacher": shuffle_sensitivity(adaptive["drift"], test_query, rng_shuffle),
        "shuffle_symbolic": shuffle_sensitivity(drift_field.predict, test_query,
                                                np.random.default_rng(int(cfg["seeds"]["shuffle"]))),
        "shuffle_truth": shuffle_sensitivity(true_drift, test_query,
                                             np.random.default_rng(int(cfg["seeds"]["shuffle"]))),
        "partial_teacher": partial_teacher,
        "partial_symbolic": partial_symbolic,
        "partial_truth": partial_truth,
        "grid_teacher_truth_nrmse": nrmse(grid_truth, grid_teacher),
        "grid_sym_teacher_nrmse": nrmse(grid_teacher, grid_symbolic),
        "grid_teacher_truth_rms": rms(grid_teacher - grid_truth),
        "grid_sym_teacher_rms": rms(grid_symbolic - grid_teacher),
        "grid_cosine_teacher": cosine_similarity(grid_teacher, grid_truth),
        "test_cosine_teacher": cosine_similarity(teacher["test"], truth["test"]),
        "test_normalized_teacher": normalized_absolute(teacher["test"], truth["test"]),
        "fail_truth": float(sim_truth["failure_rate"]),
        "fail_teacher": float(sim_teacher["failure_rate"]),
        "fail_symbolic": float(sim_symbolic["failure_rate"]),
        "failure_classes": {"truth": failure_summary(sim_truth), "teacher": failure_summary(sim_teacher),
                            "symbolic": failure_summary(sim_symbolic)},
        "paired_sym_teacher": path_rmse(sim_symbolic["states"], sim_teacher["states"], failed_pair),
        "paired_teacher_truth": path_rmse(sim_teacher["states"], sim_truth["states"], failed_pair),
        "var_sym_teacher": terminal_cov_relative(sim_symbolic["states"], sim_teacher["states"], failed_pair),
        "var_teacher_truth": terminal_cov_relative(sim_teacher["states"], sim_truth["states"], failed_pair),
        "disc_rmse": disc,
        "disc_retained": float(fine_terminal_ok.mean()),
        "law_fail_truth": float(law["truth"]["failure_rate"]),
        "law_fail_teacher": float(law["teacher"]["failure_rate"]),
        "law_fail_symbolic": float(law["symbolic"]["failure_rate"]),
        "floor_mean": mean_path_rmse(law["truth"]["states"], law["floor"]["states"],
                                     law["truth"]["failed"] | law["floor"]["failed"]),
        "floor_var": terminal_cov_relative(law["truth"]["states"], law["floor"]["states"],
                                           law["truth"]["failed"] | law["floor"]["failed"]),
        "law_mean_teacher": mean_path_rmse(law["teacher"]["states"], law["truth"]["states"],
                                           law["teacher"]["failed"] | law["truth"]["failed"]),
        "law_mean_symbolic": mean_path_rmse(law["symbolic"]["states"], law["truth"]["states"],
                                            law["symbolic"]["failed"] | law["truth"]["failed"]),
        "law_var_teacher": terminal_cov_relative(law["teacher"]["states"], law["truth"]["states"],
                                                 law["teacher"]["failed"] | law["truth"]["failed"]),
        "law_var_symbolic": terminal_cov_relative(law["symbolic"]["states"], law["truth"]["states"],
                                                  law["symbolic"]["failed"] | law["truth"]["failed"]),
        "step_ratio": step_ratio(frames["train"]["query"], dt, sigma_true),
        "nll_teacher": nll_teacher,
        "nll_truth": nll_truth,
        "nll_gap": abs(nll_teacher - nll_truth),
        "cubic_ratio_truth": cubic_ratio(test_query),
        "symbolic_cubic_ratio": degree_ratio(expanded, test_query, 3),
        "symbolic_cross_ratio": cross_ratio(expanded, test_query),
        "drift_selection": drift_meta,
        "sigma_selection": sigma_meta,
    }
    metrics["drift_step_ratio_blind"] = bool(metrics["step_ratio"] < float(cfg["distill"]["step_ratio_blind"]))
    metrics["nll_hides_drift"] = bool(
        metrics["nll_gap"] < float(cfg["distill"]["nll_floor_gap"])
        and metrics["drift_test"]["teacher_truth_nrmse"] > float(cfg["distill"]["nll_blind_nrmse"]))
    metrics["verdicts"] = decide(cfg, metrics)
    return {"metrics": metrics, "expanded": expanded, "expanded_sigma": expanded_sigma,
            "grid": grid, "grid_truth": grid_truth, "grid_teacher": grid_teacher}


def _path_nll(frame: dict[str, np.ndarray], drift_fn: DriftFn, sigma: float, dt: float) -> float:
    states = frame["states"]
    scores = []
    for step in range(states.shape[1] - 1):
        increment = states[:, step + 1] - states[:, step]
        scores.append(gaussian_nll(increment, drift_fn(states[:, step]), sigma, dt))
    return float(np.mean(scores))


def main() -> None:
    """Train the teacher, distill it, and write the one-shot test metrics."""
    started = time.perf_counter()
    cfg = load_config()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rng = np.random.default_rng(int(cfg["seeds"]["data"]))
    population = generate_population(cfg, rng)
    frames = split_population(population, cfg)
    print(f"population discarded={population['discarded']} "
          f"components={population['gmm']['n_components']} prior={diffusion_prior(cfg):.6g}", flush=True)
    adaptive = run_nn_appex(frames["train"]["states"], cfg, fix_diffusion=False, label="nn-APPEX")
    fixed = run_nn_appex(frames["train"]["states"], cfg, fix_diffusion=True, label="fixed-sigma")
    teacher = query_drift(adaptive["drift"], frames)
    drift_field, drift_meta = select_drift(
        frames["train"]["query"], teacher["train"], frames["val"]["query"], teacher["val"], cfg)
    sigma_field, sigma_meta = select_diffusion(
        frames["train"]["query"], adaptive["sigma"], frames["val"]["query"], adaptive["sigma"], cfg)
    print(f"selected drift degree={drift_meta['degree']} threshold={drift_meta['threshold']} "
          f"structure={drift_meta['structure']} val={drift_meta['score']:.4g}", flush=True)
    outcome = evaluate(cfg, population, frames, adaptive, fixed, drift_field, drift_meta, sigma_field, sigma_meta)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _save_weights(adaptive["model"], OUT_DIR / "teacher_nnappex_weights.json")
    _save_weights(fixed["model"], OUT_DIR / "teacher_fixed_sigma_weights.json")
    if isinstance(drift_field, GradientPolynomial):
        drift_field.as_polynomial_field().save(OUT_DIR / "symbolic_drift_potential.json")
        (OUT_DIR / "symbolic_drift.json").write_text(json.dumps({
            "structure": "gradient",
            "powers": drift_field.powers.tolist(),
            "coefficients": drift_field.coefficients.tolist(),
            "mean": drift_field.mean.tolist(),
            "scale": drift_field.scale.tolist(),
            "metadata": drift_field.metadata,
        }, indent=2))
    else:
        drift_field.save(OUT_DIR / "symbolic_drift.json")
    sigma_field.save(OUT_DIR / "symbolic_sigma.json")
    tex_drift = tex_equations(outcome["expanded"], (r"b_x(x,y)", r"b_y(x,y)"), ("x", "y"))
    tex_sigma = tex_equations(outcome["expanded_sigma"], (r"\sigma_x(x,y)", r"\sigma_y(x,y)"), ("x", "y"))
    tex = "\n".join([
        "% Day 18 symbolic SDE. Coefficients are the teacher approximation.",
        tex_drift,
        tex_sigma,
        f"% sigma2_hat = {adaptive['sigma2']:.8g}, sigma_hat = {adaptive['sigma']:.8g}",
    ])
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (REPORT_DIR / "symbolic_sde.tex").write_text(tex)
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps(jsonify({
        "drift": [{str(key): value for key, value in poly.items()} for poly in outcome["expanded"]],
        "sigma": [{str(key): value for key, value in poly.items()} for poly in outcome["expanded_sigma"]],
    }), indent=2))
    make_figure(outcome["grid"], outcome["grid_truth"], outcome["grid_teacher"],
                REPORT_DIR / "drift_grid.png")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "git_head": head,
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "diffusion_prior": diffusion_prior(cfg),
        "discarded": population["discarded"],
        "gmm": jsonify(population["gmm"]),
        "state_min": frames["train"]["states"].min(axis=(0, 1)).tolist(),
        "state_max": frames["train"]["states"].max(axis=(0, 1)).tolist(),
        "adaptive_trace": adaptive["trace"],
        "fixed_trace": fixed["trace"],
        "adaptive_stopped_early": adaptive["stopped_early"],
        "fixed_stopped_early": fixed["stopped_early"],
        "metrics": jsonify(outcome["metrics"]),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    print(json.dumps(jsonify(outcome["metrics"]["verdicts"]), indent=2))
    print(f"wrote {OUT_DIR / 'results.json'} in {payload['seconds']:.1f}s")


if __name__ == "__main__":
    main()
