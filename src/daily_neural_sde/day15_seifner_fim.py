"""Day 15: Seifner et al. (NeurIPS 2025) FIM-SDE -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day15.json``):

1. Data: the paper's 2D synthetic SDE (Appendix G.1, eqs. 38-39). The closed
   form is the data law. It is not a gene-regulatory equation.
2. Teacher: a miniature in-context recognition model. Pretraining supervises
   drift and diffusion on random polynomial SDEs (eqs. 9-10). Finetuning
   maximizes the short-time Gaussian likelihood (eq. 34).
3. Distillation: direct teacher queries, trajectory-disjoint splits, power
   library in (x1, x2), validation-only selection, test once.
4. Simulation keeps failed-trajectory mass and classifies non-finite states
   and escapes from the pre-registered ball.

Run from the repository root:
``python -m src.daily_neural_sde.day15_seifner_fim``

Outputs land in ``outputs/daily_neural_sde/day15/``.
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

from .common import (PolynomialField, evaluate_powers, expand_to_original,
                     fit_polynomial_field, monomial_powers, nrmse, simulate_em,
                     tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day15.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day15"

DriftFn = Callable[[np.ndarray], np.ndarray]
SigmaFn = Callable[[np.ndarray], np.ndarray]
DTYPE = torch.float64


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-15 config."""
    return json.loads(path.read_text())


def n_steps(cfg: dict[str, Any], dt: float | None = None) -> int:
    """Number of Euler intervals. ``T`` must be a multiple of ``dt``."""
    t = float(cfg["data"]["T"])
    step = float(cfg["data"]["dt"] if dt is None else dt)
    steps = int(round(t / step))
    if abs(steps * step - t) > 1e-9:
        raise ValueError("T is not a multiple of dt")
    return steps


def target_drift_coefficients() -> np.ndarray:
    """Monomial coefficients of eqs. 38-39, shape (2, n_terms)."""
    powers = monomial_powers(2, 3)
    index = {tuple(int(p) for p in row): i for i, row in enumerate(powers)}
    coef = np.zeros((2, len(powers)))
    coef[0, index[(1, 0)]] = 1.0
    coef[0, index[(0, 1)]] = -1.0
    coef[0, index[(3, 0)]] = -1.0
    coef[0, index[(1, 2)]] = -1.0
    coef[1, index[(1, 0)]] = 1.0
    coef[1, index[(0, 1)]] = 1.0
    coef[1, index[(2, 1)]] = -1.0
    coef[1, index[(0, 3)]] = -1.0
    return coef


def target_variance_coefficients() -> np.ndarray:
    """Coefficients of sigma^2 = (1 + x2^2, 1 + x1^2), shape (2, n_terms)."""
    powers = monomial_powers(2, 2)
    index = {tuple(int(p) for p in row): i for i, row in enumerate(powers)}
    coef = np.zeros((2, len(powers)))
    coef[0, index[(0, 0)]] = 1.0
    coef[0, index[(0, 2)]] = 1.0
    coef[1, index[(0, 0)]] = 1.0
    coef[1, index[(2, 0)]] = 1.0
    return coef


def evaluate_coefficient_field(x: np.ndarray, coef: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate a coefficient matrix, shape (n, outputs)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    powers = monomial_powers(x.shape[1], degree)
    if coef.shape != (coef.shape[0], len(powers)):
        raise ValueError("coefficient width does not match the library")
    return evaluate_powers(x, powers) @ np.asarray(coef, dtype=np.float64).T


def true_drift(x: np.ndarray) -> np.ndarray:
    """Stuart-Landau drift of eqs. 38-39, shape (n, 2).

    f1 = x1 (1 - r^2) - x2, f2 = x2 (1 - r^2) + x1, with r^2 = x1^2 + x2^2.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x1 = x[:, 0]
    x2 = x[:, 1]
    radius2 = x1 ** 2 + x2 ** 2
    return np.stack([x1 * (1.0 - radius2) - x2, x2 * (1.0 - radius2) + x1], axis=1)


def true_diffusion(x: np.ndarray) -> np.ndarray:
    """State-dependent diffusion coefficients, shape (n, 2).

    sigma1 = sqrt(1 + x2^2), sigma2 = sqrt(1 + x1^2).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.stack([np.sqrt(1.0 + x[:, 1] ** 2), np.sqrt(1.0 + x[:, 0] ** 2)], axis=1)


def drift_groups(x: np.ndarray) -> dict[str, np.ndarray]:
    """Linear, cubic, radial, and rotational parts of the true drift.

    Each array has shape (n, 2). Linear plus cubic equals the full drift.
    Radial plus rotational equals the full drift as well.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x1 = x[:, 0]
    x2 = x[:, 1]
    radius2 = x1 ** 2 + x2 ** 2
    linear = np.stack([x1 - x2, x1 + x2], axis=1)
    cubic = np.stack([-x1 * radius2, -x2 * radius2], axis=1)
    radial = np.stack([x1 * (1.0 - radius2), x2 * (1.0 - radius2)], axis=1)
    rotational = np.stack([-x2, x1], axis=1)
    return {"linear": linear, "cubic": cubic, "radial": radial,
            "rotational": rotational, "full": linear + cubic}


def field_rms(values: np.ndarray) -> float:
    """Root mean square of every entry."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def gaussian_nll(dy: np.ndarray, drift: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
    """Euler-Gauss NLL per transition, shape (n,).

    ``dy``, ``drift``, and ``sigma`` are (n, d). Variance of each increment
    component is sigma_i^2 * dt.
    """
    dy = np.asarray(dy, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    variance = np.maximum(sigma, 1e-8) ** 2 * dt
    residual = dy - drift * dt
    return 0.5 * np.sum(np.log(2.0 * math.pi * variance) + residual ** 2 / variance, axis=1)


def paper_eq34(dy: np.ndarray, drift: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
    """Eq. 34 per transition, with g_hat = sigma^2.

    This drops the additive 0.5 * d * log(2 pi dt) from :func:`gaussian_nll`.
    The argmin over a field is the same when dt is constant.
    """
    dy = np.asarray(dy, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    g_hat = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8) ** 2
    residual = dy - drift * dt
    return np.sum(residual ** 2 / (2.0 * g_hat * dt) + 0.5 * np.log(g_hat), axis=1)


def nll_constant_gap(dt: float, dimension: int = 2) -> float:
    """gaussian_nll - eq. 34, per transition, when dt is constant."""
    return 0.5 * dimension * math.log(2.0 * math.pi * dt)


class FIM(nn.Module):
    """Miniature functional-attention recognition model.

    Context rows are shape (m, 7): standardized state, standardized increment,
    its square, and dt. Queries are standardized states (n, 2). Outputs are
    the drift and diffusion in those standardized coordinates, plus a scalar
    uncertainty U.
    """

    def __init__(self, hidden: int, sigma_floor: float) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.sigma_floor = float(sigma_floor)
        self.ctx = nn.Sequential(
            nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.qry = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.wq = nn.Linear(hidden, hidden)
        self.wk = nn.Linear(hidden, hidden)
        self.wv = nn.Linear(hidden, hidden)
        self.f_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.s_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.u_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.u_head[-1].weight)
        nn.init.zeros_(self.u_head[-1].bias)

    def forward(self, z: torch.Tensor, context: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (f_z, sigma_z, U) with shapes (n, 2), (n, 2), (n,)."""
        if z.ndim != 2 or z.shape[1] != 2 or context.ndim != 2 or context.shape[1] != 7:
            raise ValueError("expected z (n, 2) and context (m, 7)")
        encoded = self.ctx(context)
        query = self.qry(z)
        scale = self.hidden ** -0.5
        logits = (self.wq(query) @ self.wk(encoded).T) * scale
        weights = torch.softmax(logits, dim=-1)
        pooled = weights @ self.wv(encoded)
        hidden = pooled + query
        drift = self.f_head(hidden)
        sigma = torch.nn.functional.softplus(self.s_head(hidden)) + self.sigma_floor
        uncertainty = self.u_head(hidden.detach()).squeeze(-1)
        return drift, sigma, uncertainty


def _linear_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """NumPy replay of nn.Linear. ``weight`` is (out, in)."""
    return x @ weight.T + bias


def _sequential_np(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray],
                   activations: list[str]) -> np.ndarray:
    """Apply linear layers with ``activations`` after each except a final linear."""
    hidden = x
    for weight, bias, activation in zip(weights, biases, activations):
        hidden = _linear_np(hidden, weight, bias)
        if activation == "tanh":
            hidden = np.tanh(hidden)
        elif activation != "linear":
            raise ValueError(activation)
    return hidden


def _softplus_np(x: np.ndarray) -> np.ndarray:
    """Match torch.nn.functional.softplus default threshold of 20."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -40.0, 20.0))))


def numpy_forward(z: np.ndarray, context: np.ndarray, state: dict[str, np.ndarray],
                  sigma_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy replay of :meth:`FIM.forward`. Shapes match the torch outputs."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    context = np.atleast_2d(np.asarray(context, dtype=np.float64))
    encoded = _sequential_np(
        context,
        [state["ctx.0.weight"], state["ctx.2.weight"]],
        [state["ctx.0.bias"], state["ctx.2.bias"]],
        ["tanh", "tanh"])
    query = _sequential_np(
        z,
        [state["qry.0.weight"], state["qry.2.weight"]],
        [state["qry.0.bias"], state["qry.2.bias"]],
        ["tanh", "linear"])
    hidden_dim = state["wq.weight"].shape[0]
    query_proj = _linear_np(query, state["wq.weight"], state["wq.bias"])
    key = _linear_np(encoded, state["wk.weight"], state["wk.bias"])
    value = _linear_np(encoded, state["wv.weight"], state["wv.bias"])
    logits = (query_proj @ key.T) / math.sqrt(hidden_dim)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)
    pooled = weights @ value
    hidden = pooled + query
    drift = _sequential_np(
        hidden,
        [state["f_head.0.weight"], state["f_head.2.weight"]],
        [state["f_head.0.bias"], state["f_head.2.bias"]],
        ["tanh", "linear"])
    raw_sigma = _sequential_np(
        hidden,
        [state["s_head.0.weight"], state["s_head.2.weight"]],
        [state["s_head.0.bias"], state["s_head.2.bias"]],
        ["tanh", "linear"])
    sigma = _softplus_np(raw_sigma) + sigma_floor
    raw_u = _sequential_np(
        hidden,
        [state["u_head.0.weight"], state["u_head.2.weight"]],
        [state["u_head.0.bias"], state["u_head.2.bias"]],
        ["tanh", "linear"])
    return drift, sigma, raw_u.reshape(-1)


def state_dict_numpy(model: FIM) -> dict[str, np.ndarray]:
    """Detach a float64 state dict into NumPy arrays."""
    return {key: value.detach().cpu().numpy().astype(np.float64)
            for key, value in model.state_dict().items()}


class FrozenTeacher:
    """Neural field in original coordinates, conditioned on a fixed context bank.

    ``mean`` and ``scale`` have shape (2,). ``context`` is (m, 7) and is
    already standardized. ``predict`` returns drift (n, 2), sigma (n, 2),
    and U (n,).
    """

    def __init__(self, model: FIM, mean: np.ndarray, scale: np.ndarray,
                 context: np.ndarray) -> None:
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.context = np.asarray(context, dtype=np.float64)
        self._context_tensor = torch.as_tensor(self.context, dtype=DTYPE)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Original-coordinate fields at states ``x`` of shape (n, 2)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.scale
        with torch.no_grad():
            drift_z, sigma_z, uncertainty = self.model(
                torch.as_tensor(z, dtype=DTYPE), self._context_tensor)
        drift = drift_z.cpu().numpy() * self.scale
        sigma = sigma_z.cpu().numpy() * self.scale
        return drift, sigma, uncertainty.cpu().numpy()

    def drift(self, x: np.ndarray) -> np.ndarray:
        """Drift only, shape (n, 2)."""
        return self.predict(x)[0]

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """Diffusion coefficients only, shape (n, 2)."""
        return self.predict(x)[1]


def heteroscedastic_parts(drift_hat: torch.Tensor, drift_true: torch.Tensor,
                          sigma_hat: torch.Tensor, sigma_true: torch.Tensor,
                          uncertainty: torch.Tensor) -> torch.Tensor:
    """Eq. 10 with U detached from the trunk and from the field weight.

    The field term uses a stopped U as a weight. The U head is trained on a
    stopped copy of the pointwise loss, so its stationary point is
    U = log(L1) and it does not increase L1 to reduce the weight.
    """
    pointwise = ((drift_hat - drift_true) ** 2).mean(dim=-1)
    pointwise = pointwise + ((sigma_hat - sigma_true) ** 2).mean(dim=-1)
    field_term = torch.exp(-uncertainty.detach()) * pointwise
    uncertainty_term = torch.exp(-uncertainty) * pointwise.detach() + uncertainty
    return (field_term + uncertainty_term).mean()


def transition_features(state: np.ndarray, increment: np.ndarray, dt: float,
                        mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Context rows (n, 7) from states and increments in original coordinates."""
    state = np.asarray(state, dtype=np.float64)
    increment = np.asarray(increment, dtype=np.float64)
    standardized = (state - mean) / scale
    increment_z = increment / scale
    time = np.full((len(state), 1), float(dt))
    return np.concatenate([standardized, increment_z, increment_z ** 2, time], axis=1)


def fit_standardizer(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and scale of states (n, 2) or (n, t, 2). Train rows only."""
    flat = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    flat = flat[np.isfinite(flat).all(axis=1)]
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def _simulate_law(drift_fn: DriftFn, sigma_fn: SigmaFn, x0: np.ndarray, dt: float,
                  increments: np.ndarray, failure_radius: float) -> dict[str, Any]:
    """Euler-Maruyama. Increments are standard normals of shape (n, steps, 2)."""
    return simulate_em(drift_fn, sigma_fn, x0, dt, increments.shape[1], increments,
                       failure_radius)


def generate_target_paths(n: int, cfg: dict[str, Any], rng: np.random.Generator
                          ) -> dict[str, np.ndarray]:
    """Ring of initial conditions at the paper's radius, plus jitter.

    Returns states (n, steps+1, 2), increments (n, steps, 2), and x0 (n, 2).
    """
    data = cfg["data"]
    steps = n_steps(cfg)
    radius = float(math.hypot(data["x0_xy"][0], data["x0_xy"][1]))
    angle = rng.uniform(0.0, 2.0 * math.pi, size=n)
    x0 = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    x0 = x0 + rng.normal(0.0, float(data["x0_jitter"]), size=x0.shape)
    increments = rng.normal(size=(n, steps, 2))
    simulated = _simulate_law(true_drift, true_diffusion, x0, float(data["dt"]),
                              increments, float(data["failure_radius"]))
    return {"states": simulated["states"], "increments": increments, "x0": x0,
            "failed": simulated["failed"]}


def make_dataset(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Trajectory-disjoint train, validation, and test, in that order."""
    data = cfg["data"]
    counts = {"train": int(data["n_train"]), "val": int(data["n_val"]),
              "test": int(data["n_test"])}
    splits = {}
    for name, count in counts.items():
        splits[name] = generate_target_paths(count, cfg, rng)
    return splits


def transitions_from_paths(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nonterminal states and increments, both (n_traj * steps, 2)."""
    current = states[:, :-1, :].reshape(-1, 2)
    nxt = states[:, 1:, :].reshape(-1, 2)
    return current, nxt - current


def sample_context(states: np.ndarray, dt: float, mean: np.ndarray, scale: np.ndarray,
                   size: int, rng: np.random.Generator) -> np.ndarray:
    """A fixed bank of standardized transitions, shape (size, 7)."""
    current, increment = transitions_from_paths(states)
    finite = np.isfinite(current).all(axis=1) & np.isfinite(increment).all(axis=1)
    current = current[finite]
    increment = increment[finite]
    if len(current) == 0:
        raise RuntimeError("no finite transitions")
    choice = rng.choice(len(current), size=size, replace=len(current) < size)
    return transition_features(current[choice], increment[choice], dt, mean, scale)


def _probe_sde(drift_coef: np.ndarray, variance_coef: np.ndarray,
               rng: np.random.Generator) -> bool:
    """Reject polynomial SDEs that leave a bounded box in a short probe."""
    x = rng.normal(0.0, 0.4, size=(4, 2))
    dt = 0.05
    for _ in range(30):
        drift = evaluate_coefficient_field(x, drift_coef, 3)
        variance = evaluate_coefficient_field(x, variance_coef, 2)
        sigma = np.sqrt(np.maximum(variance, 0.0) + 1e-8)
        shock = rng.normal(size=x.shape)
        x = x + drift * dt + sigma * math.sqrt(dt) * shock
        if not np.isfinite(x).all() or float(np.max(np.abs(x))) > 4.0:
            return False
    return True


def sample_pretrain_task(rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, Any]:
    """One random degree-3 drift and degree-2 variance polynomial, then a path.

    Tasks within L2 distance 0.5 of both target coefficient blocks are rejected
    so the held-out system is not copied into the pretraining set.
    """
    model_cfg = cfg["model"]
    target_drift = target_drift_coefficients()
    target_variance = target_variance_coefficients()
    for _ in range(500):
        drift_coef = np.zeros_like(target_drift)
        variance_coef = np.zeros_like(target_variance)
        for row in range(2):
            drift_mask = rng.random(drift_coef.shape[1]) >= 0.5
            variance_mask = rng.random(variance_coef.shape[1]) >= 0.5
            drift_coef[row, drift_mask] = rng.normal(size=int(drift_mask.sum()))
            variance_coef[row, variance_mask] = rng.normal(size=int(variance_mask.sum()))
        drift_gap = float(np.linalg.norm(drift_coef - target_drift))
        variance_gap = float(np.linalg.norm(variance_coef - target_variance))
        if drift_gap < 0.5 and variance_gap < 0.5:
            continue
        if not _probe_sde(drift_coef, variance_coef, rng):
            continue

        def drift_fn(x: np.ndarray, coef: np.ndarray = drift_coef) -> np.ndarray:
            return evaluate_coefficient_field(x, coef, 3)

        def sigma_fn(x: np.ndarray, coef: np.ndarray = variance_coef) -> np.ndarray:
            variance = evaluate_coefficient_field(x, coef, 2)
            return np.sqrt(np.maximum(variance, 0.0) + 1e-8)

        steps = int(model_cfg["pretrain_steps"])
        dt = float(model_cfg["pretrain_dt"])
        x0 = rng.normal(0.0, 0.4, size=(1, 2))
        increments = rng.normal(size=(1, steps, 2))
        simulated = _simulate_law(drift_fn, sigma_fn, x0, dt, increments, 4.0)
        states = simulated["states"]
        if simulated["failed"][0] or not np.isfinite(states).all():
            continue
        mean, scale = fit_standardizer(states)
        if float(np.min(scale)) < 0.05:
            continue
        current, increment = transitions_from_paths(states)
        context = transition_features(current, increment, dt, mean, scale)
        path_queries = current
        box_queries = rng.uniform(-1.5, 1.5, size=(int(model_cfg["pretrain_queries"]) // 2, 2))
        take = rng.choice(len(path_queries), size=int(model_cfg["pretrain_queries"]) // 2,
                          replace=True)
        queries = np.concatenate([box_queries, path_queries[take]], axis=0)
        truth_drift = drift_fn(queries)
        truth_sigma = sigma_fn(queries)
        if not (np.isfinite(truth_drift).all() and np.isfinite(truth_sigma).all()):
            continue
        return {"context": context, "queries": queries, "drift": truth_drift,
                "sigma": truth_sigma, "mean": mean, "scale": scale}
    raise RuntimeError("failed to sample a stable pretraining SDE")


def build_model(cfg: dict[str, Any]) -> FIM:
    """Float64 recognition model with the frozen width and diffusion floor."""
    model_cfg = cfg["model"]
    model = FIM(int(model_cfg["hidden"]), float(model_cfg["sigma_floor"]))
    return model.to(dtype=DTYPE)


def pretrain(model: FIM, tasks: list[dict[str, Any]], cfg: dict[str, Any]
             ) -> dict[str, float]:
    """Supervise normalized drift and diffusion. Returns the last-epoch loss split."""
    model_cfg = cfg["model"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg["lr"]),
                                  weight_decay=float(model_cfg["weight_decay"]))
    last_field = 0.0
    last_uncertainty = 0.0
    last_weighted = 0.0
    model.train()
    for _epoch in range(int(model_cfg["pretrain_epochs"])):
        field_acc = 0.0
        uncertainty_acc = 0.0
        weighted_acc = 0.0
        for task in tasks:
            z = (task["queries"] - task["mean"]) / task["scale"]
            drift_true = task["drift"] / task["scale"]
            sigma_true = task["sigma"] / task["scale"]
            context = torch.as_tensor(task["context"], dtype=DTYPE)
            z_tensor = torch.as_tensor(z, dtype=DTYPE)
            drift_hat, sigma_hat, uncertainty = model(z_tensor, context)
            loss = heteroscedastic_parts(
                drift_hat, torch.as_tensor(drift_true, dtype=DTYPE),
                sigma_hat, torch.as_tensor(sigma_true, dtype=DTYPE), uncertainty)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg["grad_clip"]))
            optimizer.step()
            with torch.no_grad():
                pointwise = ((drift_hat - torch.as_tensor(drift_true, dtype=DTYPE)) ** 2).mean(-1)
                pointwise = pointwise + (
                    (sigma_hat - torch.as_tensor(sigma_true, dtype=DTYPE)) ** 2).mean(-1)
                field_acc += float(pointwise.mean())
                uncertainty_acc += float(uncertainty.mean())
                weighted_acc += float((torch.exp(-uncertainty.detach()) * pointwise).mean())
        count = max(len(tasks), 1)
        last_field = field_acc / count
        last_uncertainty = uncertainty_acc / count
        last_weighted = weighted_acc / count
    return {"mean_l1": last_field, "mean_u": last_uncertainty,
            "mean_weighted_l1": last_weighted,
            "penalty_ratio": abs(last_uncertainty) / max(last_weighted, 1e-8)}


def _batch_nll(model: FIM, current: np.ndarray, increment: np.ndarray, dt: float,
               mean: np.ndarray, scale: np.ndarray, context: torch.Tensor) -> torch.Tensor:
    """Mean Euler-Gauss NLL of a batch, in original coordinates."""
    z = (current - mean) / scale
    drift_z, sigma_z, _uncertainty = model(torch.as_tensor(z, dtype=DTYPE), context)
    drift = drift_z * torch.as_tensor(scale, dtype=DTYPE)
    sigma = sigma_z * torch.as_tensor(scale, dtype=DTYPE)
    variance = sigma ** 2 * dt
    residual = torch.as_tensor(increment, dtype=DTYPE) - drift * dt
    nll = 0.5 * (torch.log(2.0 * math.pi * variance) + residual ** 2 / variance)
    return nll.sum(dim=-1).mean()


def finetune(model: FIM, current: np.ndarray, increment: np.ndarray,
             val_current: np.ndarray, val_increment: np.ndarray, context: np.ndarray,
             mean: np.ndarray, scale: np.ndarray, cfg: dict[str, Any], steps: int
             ) -> dict[str, Any]:
    """Eq. 34 Gaussian finetune. U is not in the loss. Best validation NLL is restored."""
    model_cfg = cfg["model"]
    dt = float(cfg["data"]["dt"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg["lr"]),
                                  weight_decay=float(model_cfg["weight_decay"]))
    context_tensor = torch.as_tensor(context, dtype=DTYPE)
    best_nll = math.inf
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_step = 0
    history = []
    n_rows = len(current)
    rng = np.random.default_rng(int(cfg["seed"]) + 17 + steps)
    for step in range(steps + 1):
        if step % int(model_cfg["eval_every"]) == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_nll = float(_batch_nll(model, val_current, val_increment, dt, mean,
                                            scale, context_tensor))
            history.append({"step": step, "val_nll": val_nll})
            if val_nll < best_nll:
                best_nll = val_nll
                best_step = step
                best_state = {key: value.detach().clone()
                              for key, value in model.state_dict().items()}
            model.train()
        if step == steps:
            break
        choice = rng.choice(n_rows, size=min(int(model_cfg["batch_size"]), n_rows),
                            replace=False)
        loss = _batch_nll(model, current[choice], increment[choice], dt, mean, scale,
                          context_tensor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg["grad_clip"]))
        optimizer.step()
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_nll": best_nll, "best_step": best_step, "history": history}


def validation_nll(model: FIM, current: np.ndarray, increment: np.ndarray,
                   context: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                   dt: float) -> float:
    """Mean validation NLL with a frozen context bank."""
    model.eval()
    with torch.no_grad():
        return float(_batch_nll(model, current, increment, dt, mean, scale,
                                torch.as_tensor(context, dtype=DTYPE)))


def choose_candidate(scores: dict[str, float], priority: list[str], atol: float) -> str:
    """Lowest validation NLL. Ties within ``atol`` follow ``priority`` order."""
    finite = {name: scores[name] for name in priority if math.isfinite(scores.get(name, math.nan))}
    if not finite:
        raise RuntimeError("no finite teacher candidate")
    best = min(finite.values())
    eligible = [name for name in priority if name in finite and finite[name] <= best + atol]
    return eligible[0]


def component_nrmse(truth: np.ndarray, prediction: np.ndarray) -> list[float]:
    """NRMSE of each column."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return [nrmse(truth[:, j], prediction[:, j]) for j in range(truth.shape[1])]


def residual_rms(truth: np.ndarray, prediction: np.ndarray) -> list[float]:
    """Per-column root mean square residual."""
    gap = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return [float(np.sqrt(np.mean(gap[:, j] ** 2))) for j in range(gap.shape[1])]


def column_rms(values: np.ndarray) -> list[float]:
    """Per-column root mean square."""
    values = np.asarray(values, dtype=np.float64)
    return [float(np.sqrt(np.mean(values[:, j] ** 2))) for j in range(values.shape[1])]


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


def fit_ols(x: np.ndarray, y: np.ndarray, degree: int) -> PolynomialField:
    """Dense least squares. Standardization uses ``x`` only. Not a selected model."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    powers = monomial_powers(x.shape[1], degree)
    theta = evaluate_powers((x - mean) / scale, powers)
    coef = np.zeros((y.shape[1], theta.shape[1]))
    for j in range(y.shape[1]):
        solution, *_ = np.linalg.lstsq(theta, y[:, j], rcond=None)
        coef[j] = solution
    return PolynomialField(powers, coef, mean, scale,
                           {"degree": degree, "fit": "ols", "library_size": int(len(powers))})


def select_field(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, degrees: list[int], thresholds: list[float],
                 ridge: float) -> tuple[PolynomialField, dict[str, Any]]:
    """Pick degree and threshold by pooled validation NRMSE. Ties keep the earlier pair."""
    best: PolynomialField | None = None
    best_val = math.inf
    best_meta: dict[str, Any] = {}
    candidates = []
    for degree in degrees:
        for threshold in thresholds:
            field = fit_polynomial_field(x_train, y_train, degree, threshold, ridge)
            score = nrmse(y_val, field.predict(x_val))
            candidates.append({"degree": degree, "threshold": threshold,
                               "validation_nrmse": score,
                               "active_terms": int(np.count_nonzero(field.coefficients))})
            if score < best_val:
                best_val = score
                best = field
                best_meta = {"degree": degree, "threshold": threshold,
                             "validation_nrmse": score,
                             "active_terms": int(np.count_nonzero(field.coefficients)),
                             "library_size": int(field.metadata["library_size"])}
    if best is None:
        raise RuntimeError("no candidate")
    best_meta["candidates"] = candidates
    return best, best_meta


def percentile_grid(x: np.ndarray, size: int, quantiles: list[float], pad: float) -> np.ndarray:
    """Axis-aligned grid on a padded central box of ``x``. Shape (size^d, d)."""
    x = np.asarray(x, dtype=np.float64)
    low = np.quantile(x, quantiles[0], axis=0)
    high = np.quantile(x, quantiles[1], axis=0)
    span = high - low
    low = low - pad * span
    high = high + pad * span
    axes = [np.linspace(low[j], high[j], size) for j in range(x.shape[1])]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return mesh.reshape(-1, x.shape[1])


def ring_points(radius: float, count: int) -> np.ndarray:
    """Equally spaced points on a circle, shape (count, 2)."""
    angle = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def failure_counts(classes: list[str | None]) -> dict[str, int]:
    """Counts of the two pre-registered failure classes."""
    return {"non_finite": sum(item == "non_finite" for item in classes),
            "outside_radius": sum(item == "outside_radius" for item in classes)}


def paired_path_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Time-averaged RMSE over trajectories finite in both ensembles."""
    errors = []
    for time in range(left.shape[1]):
        ok = np.isfinite(left[:, time]).all(axis=1) & np.isfinite(right[:, time]).all(axis=1)
        if ok.any():
            errors.append(float(np.mean((left[ok, time] - right[ok, time]) ** 2)))
    if not errors:
        return math.nan
    return float(math.sqrt(float(np.mean(errors))))


def mean_path(states: np.ndarray) -> np.ndarray:
    """Time-wise nanmean, shape (n_obs, d)."""
    return np.nanmean(states, axis=0)


def mean_path_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """RMSE of two ensemble-mean paths, skipping non-finite times."""
    ok = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    if not ok.any():
        return math.nan
    return float(np.sqrt(np.mean((left[ok] - right[ok]) ** 2)))


def terminal_cov(states: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Covariance of finite terminal states, and the retained count."""
    final = states[:, -1, :]
    ok = np.isfinite(final).all(axis=1)
    if int(ok.sum()) < 3:
        return None, int(ok.sum())
    return np.cov(final[ok], rowvar=False), int(ok.sum())


def rel_frob(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    """||left - right||_F / ||right||_F."""
    if left is None or right is None:
        return None
    return float(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1e-8))


def query_bundle(states: np.ndarray, teacher: FrozenTeacher) -> dict[str, np.ndarray]:
    """Nonterminal states plus teacher and truth fields."""
    current, increment = transitions_from_paths(states)
    finite = np.isfinite(current).all(axis=1) & np.isfinite(increment).all(axis=1)
    current = current[finite]
    increment = increment[finite]
    drift, sigma, uncertainty = teacher.predict(current)
    return {"features": current, "increment": increment, "drift": drift, "sigma": sigma,
            "uncertainty": uncertainty, "drift_truth": true_drift(current),
            "sigma_truth": true_diffusion(current)}


def monomial_table(field: PolynomialField) -> list[dict[str, float]]:
    """Original-coordinate coefficients per output, keyed by exponent text."""
    tables = []
    for poly in expand_to_original(field):
        tables.append({",".join(str(power) for power in exp): float(coef)
                       for exp, coef in sorted(poly.items())})
    return tables


def named_coefficients(field: PolynomialField, names: tuple[str, ...]) -> dict[str, list[float]]:
    """Pull a few monomials out of the original-coordinate expansion."""
    wanted = {name: [] for name in names}
    index = {name: i for i, name in enumerate(names)}
    for poly in expand_to_original(field):
        values = {name: 0.0 for name in names}
        for exp, coef in poly.items():
            key = ",".join(str(power) for power in exp)
            if key in index:
                values[key] = float(coef)
        for name in names:
            wanted[name].append(values[name])
    return wanted


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Thresholds are read from ``cfg`` and not modified."""
    hyp = cfg["hypotheses"]
    h1, h2 = hyp["H1_drift"], hyp["H2_diffusion"]
    h4, h5 = hyp["H4_simulation"], hyp["H5_law"]
    h6, h7 = hyp["H6_uncertainty"], hyp["H7_discretization_gate"]
    verdicts: dict[str, Any] = {
        "H1": _pass(
            all(value <= h1["symbolic_vs_teacher_test_nrmse_max"]
                for value in metrics["drift_sym_teacher"])
            and all(value <= h1["teacher_vs_truth_test_nrmse_max"]
                    for value in metrics["drift_teacher_truth"])
            and all(value <= h1["symbolic_vs_truth_test_nrmse_max"]
                    for value in metrics["drift_sym_truth"])),
        "H2": _pass(
            all(value <= h2["symbolic_vs_teacher_test_nrmse_max"]
                for value in metrics["sigma_sym_teacher"])
            and all(value <= h2["teacher_vs_truth_test_nrmse_max"]
                    for value in metrics["sigma_teacher_truth"])
            and all(value <= h2["symbolic_vs_truth_test_nrmse_max"]
                    for value in metrics["sigma_sym_truth"])
            and metrics["variance_sym_teacher"]
            <= h2["symbolic_vs_teacher_variance_nrmse_max"]),
        "H2c": _pass(metrics["fraction_sigma_nonpositive"]
                     <= hyp["H2c_range"]["fraction_symbolic_sigma_nonpositive_max"]),
        "H3": ("indeterminate" if metrics["vif"] > hyp["H3_collinearity_gate"]["vif_max"]
               else "pass"),
        "H3_closed_form": False,
    }
    fail_ok = (metrics["fail_truth"] <= h4["failure_rate_max"]
               and metrics["fail_teacher"] <= h4["failure_rate_max"]
               and metrics["fail_symbolic"] <= h4["failure_rate_max"]
               and abs(metrics["fail_symbolic"] - metrics["fail_teacher"])
               <= h4["failure_rate_gap_max"])
    h7_ok = (math.isfinite(metrics["disc_rmse"])
             and metrics["disc_rmse"] <= h7["teacher_coarse_vs_fine_terminal_rmse_max"]
             and metrics["disc_retained"] >= h7["min_retained_fraction"])
    path_ok = (math.isfinite(metrics["paired_sym_teacher"])
               and math.isfinite(metrics["paired_teacher_truth"])
               and metrics["paired_sym_teacher"]
               <= h4["paired_path_rmse_symbolic_vs_teacher_max"]
               and metrics["paired_teacher_truth"]
               <= h4["paired_path_rmse_teacher_vs_truth_max"])
    cov_sym = metrics["cov_sym_teacher"]
    cov_truth = metrics["cov_teacher_truth"]
    cov_ok = (cov_sym is not None and cov_truth is not None
              and cov_sym <= h4["terminal_cov_rel_frob_symbolic_vs_teacher_max"]
              and cov_truth <= h4["terminal_cov_rel_frob_teacher_vs_truth_max"])
    drift_ok = all(value <= h4["drift_teacher_vs_truth_test_nrmse_max"]
                   for value in metrics["drift_teacher_truth"])
    gate = (max(metrics["fail_truth"], metrics["fail_teacher"], metrics["fail_symbolic"])
            <= h4["failure_gate"] and h7_ok and cov_sym is not None and cov_truth is not None)
    verdicts["H4_failure"] = _pass(fail_ok)
    verdicts["H4_path_match_without_field_recovery"] = bool(path_ok and not drift_ok)
    verdicts["H4"] = (_pass(fail_ok and path_ok and cov_ok and drift_ok)
                      if gate else "indeterminate")
    verdicts["H7"] = _pass(h7_ok)
    law_fail = max(metrics["law_fail_truth"], metrics["law_fail_teacher"],
                   metrics["law_fail_symbolic"])
    floor_mean = metrics["floor_mean"]
    floor_cov = metrics["floor_cov"]
    floor_blocks = (not math.isfinite(floor_mean)
                    or floor_mean > h5["mean_path_rmse_teacher_max"]
                    or floor_cov is None
                    or not math.isfinite(floor_cov)
                    or floor_cov > h5["terminal_cov_rel_frob_teacher_max"])
    law_numbers_ok = (
        math.isfinite(metrics["law_mean_teacher"])
        and math.isfinite(metrics["law_mean_symbolic"])
        and metrics["law_cov_teacher"] is not None
        and metrics["law_cov_symbolic"] is not None
        and metrics["law_mean_teacher"] <= h5["mean_path_rmse_teacher_max"]
        and metrics["law_mean_symbolic"] <= h5["mean_path_rmse_symbolic_max"]
        and metrics["law_cov_teacher"] <= h5["terminal_cov_rel_frob_teacher_max"]
        and metrics["law_cov_symbolic"] <= h5["terminal_cov_rel_frob_symbolic_max"])
    verdicts["H5"] = "indeterminate" if law_fail > h5["failure_gate"] or floor_blocks else _pass(
        law_numbers_ok)
    if not math.isfinite(metrics["u_rise"]) or not math.isfinite(metrics["penalty_ratio"]):
        verdicts["H6"] = "indeterminate"
        verdicts["H6_reason"] = "missing_uncertainty"
    elif metrics["penalty_ratio"] < h6["penalty_ratio_min"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "uncertainty_term_negligible"
    elif metrics["u_rise"] < h6["u_interior_minus_id_min"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "no_ood_rise"
    else:
        verdicts["H6"] = "pass"
        verdicts["H6_reason"] = "pretrained_rise"
    verdicts["coefficients_are_closed_form"] = False
    return verdicts


def git_head() -> str:
    """Current commit, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def jsonify(obj: Any) -> Any:
    """Recursively replace NumPy values with JSON-native ones."""
    if isinstance(obj, dict):
        return {str(key): jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array).all(axis=1)
    return mask


def step_scales(states: np.ndarray, dt: float) -> dict[str, float]:
    """Mean |f| dt and mean |sigma| sqrt(dt) on finite nonterminal states."""
    current, _increment = transitions_from_paths(states)
    current = current[np.isfinite(current).all(axis=1)]
    drift = true_drift(current)
    sigma = true_diffusion(current)
    drift_step = float(np.mean(np.linalg.norm(drift, axis=1)) * dt)
    sigma_step = float(np.mean(np.linalg.norm(sigma, axis=1)) * math.sqrt(dt))
    return {"mean_abs_drift_dt": drift_step, "mean_abs_sigma_sqrt_dt": sigma_step,
            "drift_over_diffusion_step": drift_step / max(sigma_step, 1e-8)}


def discretization_check(teacher: FrozenTeacher, x0: np.ndarray, cfg: dict[str, Any],
                         rng: np.random.Generator) -> dict[str, float]:
    """Terminal RMSE of coarse EM versus k fine steps with the same Wiener path."""
    data = cfg["data"]
    k = int(cfg["hypotheses"]["H7_discretization_gate"]["fine_factor"])
    dt = float(data["dt"])
    steps = n_steps(cfg)
    fine_noise = rng.normal(size=(len(x0), steps, k, 2))
    coarse_noise = fine_noise.sum(axis=2) / math.sqrt(k)
    radius = float(data["failure_radius"])
    coarse = _simulate_law(teacher.drift, teacher.diffusion, x0, dt, coarse_noise, radius)
    fine = _simulate_law(teacher.drift, teacher.diffusion, x0, dt / k,
                         fine_noise.reshape(len(x0), steps * k, 2), radius)
    coarse_final = coarse["states"][:, -1, :]
    fine_final = fine["states"][:, -1, :]
    both = np.isfinite(coarse_final).all(axis=1) & np.isfinite(fine_final).all(axis=1)
    if not both.any():
        return {"disc_rmse": math.nan, "disc_retained": 0.0}
    rmse = float(np.sqrt(np.mean((coarse_final[both] - fine_final[both]) ** 2)))
    return {"disc_rmse": rmse, "disc_retained": float(np.mean(both))}


def uncertainty_probe(teacher: FrozenTeacher, train_states: np.ndarray, cfg: dict[str, Any]
                      ) -> dict[str, float]:
    """Mean U on train states, inside the cycle, and outside the train radius box."""
    h6 = cfg["hypotheses"]["H6_uncertainty"]
    current, _increment = transitions_from_paths(train_states)
    current = current[np.isfinite(current).all(axis=1)]
    rng = np.random.default_rng(int(cfg["seed"]) + 23)
    choice = rng.choice(len(current), size=min(256, len(current)), replace=False)
    identified = current[choice]
    interior = ring_points(float(h6["interior_radius"]) * 0.5, int(h6["n_probe"]))
    radii = np.linalg.norm(current, axis=1)
    low, high = np.quantile(radii, [0.05, 0.95])
    exterior_radius = float(high + (high - low))
    exterior = ring_points(exterior_radius, int(h6["n_probe"]))
    _fd, _sd, u_id = teacher.predict(identified)
    _fi, _si, u_in = teacher.predict(interior)
    _fe, _se, u_out = teacher.predict(exterior)
    return {"u_id": float(np.mean(u_id)), "u_interior": float(np.mean(u_in)),
            "u_exterior": float(np.mean(u_out)),
            "u_rise": float(np.mean(u_in) - np.mean(u_id)),
            "u_exterior_rise": float(np.mean(u_out) - np.mean(u_id)),
            "exterior_radius": exterior_radius}


def field_metrics(truth: np.ndarray, teacher_values: np.ndarray,
                  symbolic_values: np.ndarray) -> dict[str, Any]:
    """Component NRMSE and residual RMS for truth, teacher, and symbolic."""
    return {
        "sym_teacher": component_nrmse(teacher_values, symbolic_values),
        "teacher_truth": component_nrmse(truth, teacher_values),
        "sym_truth": component_nrmse(truth, symbolic_values),
        "resid_sym_teacher": residual_rms(teacher_values, symbolic_values),
        "resid_teacher_truth": residual_rms(truth, teacher_values),
        "resid_sym_truth": residual_rms(truth, symbolic_values),
        "teacher_rms": column_rms(teacher_values),
        "truth_rms": column_rms(truth),
        "pooled_sym_teacher": nrmse(teacher_values, symbolic_values),
        "pooled_teacher_truth": nrmse(truth, teacher_values),
        "pooled_sym_truth": nrmse(truth, symbolic_values),
    }


def main() -> dict[str, Any]:
    """Train the miniature FIM, distill it, and write the day-15 outputs."""
    started = time.time()
    cfg = load_config()
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed)
    pre_rng = np.random.default_rng(seed + 1000)
    context_rng = np.random.default_rng(seed + 11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(cfg, data_rng)
    dt = float(cfg["data"]["dt"])
    mean, scale = fit_standardizer(dataset["train"]["states"])
    context = sample_context(dataset["train"]["states"], dt, mean, scale,
                             int(cfg["model"]["context_size"]), context_rng)
    train_x, train_dy = transitions_from_paths(dataset["train"]["states"])
    val_x, val_dy = transitions_from_paths(dataset["val"]["states"])
    train_ok = _finite_rows(train_x, train_dy)
    val_ok = _finite_rows(val_x, val_dy)
    train_x, train_dy = train_x[train_ok], train_dy[train_ok]
    val_x, val_dy = val_x[val_ok], val_dy[val_ok]

    n_tasks = int(cfg["model"]["pretrain_tasks"])
    tasks = []
    for index in range(n_tasks):
        tasks.append(sample_pretrain_task(pre_rng, cfg))
        if (index + 1) % 16 == 0:
            print(f"pretrain tasks {index + 1}/{n_tasks}", flush=True)
    pretrained = build_model(cfg)
    pre_stats = pretrain(pretrained, tasks, cfg)
    print("pretrain penalty_ratio", pre_stats["penalty_ratio"], "mean_u", pre_stats["mean_u"],
          flush=True)
    zero_shot = validation_nll(pretrained, val_x, val_dy, context, mean, scale, dt)

    finetuned = build_model(cfg)
    finetuned.load_state_dict(pretrained.state_dict())
    fine_info = finetune(finetuned, train_x, train_dy, val_x, val_dy, context, mean, scale,
                         cfg, int(cfg["model"]["finetune_steps"]))
    print("finetune", fine_info["best_step"], fine_info["best_val_nll"], flush=True)
    scratch = build_model(cfg)
    scratch_info = finetune(scratch, train_x, train_dy, val_x, val_dy, context, mean, scale,
                            cfg, int(cfg["model"]["scratch_steps"]))
    print("scratch", scratch_info["best_step"], scratch_info["best_val_nll"], flush=True)
    scores = {"zero_shot": zero_shot, "finetuned": fine_info["best_val_nll"],
              "scratch": scratch_info["best_val_nll"]}
    chosen = choose_candidate(scores, list(cfg["model"]["candidate_priority"]),
                              float(cfg["model"]["nll_tie_atol"]))
    models = {"zero_shot": pretrained, "finetuned": finetuned, "scratch": scratch}
    teacher = FrozenTeacher(models[chosen], mean, scale, context)
    # Validation field errors are diagnostics. They do not choose the teacher.
    val_drift_hat, val_sigma_hat, _val_u = teacher.predict(val_x)
    selection_field = {
        "val_drift_nrmse_selected": component_nrmse(true_drift(val_x), val_drift_hat),
        "val_sigma_nrmse_selected": component_nrmse(true_diffusion(val_x), val_sigma_hat),
    }
    for name, candidate in models.items():
        frozen = FrozenTeacher(candidate, mean, scale, context)
        drift_hat, sigma_hat, _u = frozen.predict(val_x)
        selection_field[name] = {
            "val_nll": scores[name],
            "val_drift_nrmse": component_nrmse(true_drift(val_x), drift_hat),
            "val_sigma_nrmse": component_nrmse(true_diffusion(val_x), sigma_hat),
        }

    queries = {name: query_bundle(dataset[name]["states"], teacher)
               for name in ("train", "val", "test")}
    distill = cfg["distill"]
    drift_field, drift_meta = select_field(
        queries["train"]["features"], queries["train"]["drift"],
        queries["val"]["features"], queries["val"]["drift"],
        list(distill["degrees"]), list(distill["thresholds"]), float(distill["ridge"]))
    sigma_field, sigma_meta = select_field(
        queries["train"]["features"], queries["train"]["sigma"],
        queries["val"]["features"], queries["val"]["sigma"],
        list(distill["degrees"]), list(distill["thresholds"]), float(distill["ridge"]))

    test = queries["test"]
    drift_sym = drift_field.predict(test["features"])
    sigma_sym = sigma_field.predict(test["features"])
    drift_numbers = field_metrics(test["drift_truth"], test["drift"], drift_sym)
    sigma_numbers = field_metrics(test["sigma_truth"], test["sigma"], sigma_sym)
    variance_sym = nrmse(test["sigma"] ** 2, np.maximum(sigma_sym, 0.0) ** 2)
    fraction_nonpositive = float(np.mean(sigma_sym <= 0.0))

    oracle_drift = fit_ols(queries["train"]["features"], queries["train"]["drift_truth"],
                           int(distill["oracle_degree"]))
    oracle_sigma = fit_ols(queries["train"]["features"], queries["train"]["sigma_truth"],
                           int(distill["oracle_degree"]))
    oracle_variance = fit_ols(queries["train"]["features"], queries["train"]["sigma_truth"] ** 2,
                              2)
    ceiling_drift = fit_ols(queries["train"]["features"], queries["train"]["drift"],
                            int(distill["oracle_degree"]))
    ceiling_sigma = fit_ols(queries["train"]["features"], queries["train"]["sigma"],
                            int(distill["oracle_degree"]))
    key_names = ("1,0", "0,1", "3,0", "2,1", "1,2", "0,3")
    groups = drift_groups(test["features"])
    grid = percentile_grid(queries["train"]["features"], int(distill["grid_size"]),
                           list(distill["grid_quantile"]), float(distill["grid_pad_fraction"]))
    grid_drift_t, grid_sigma_t, _grid_u = teacher.predict(grid)
    grid_drift_s = drift_field.predict(grid)
    grid_sigma_s = sigma_field.predict(grid)
    grid_drift_truth = true_drift(grid)
    grid_sigma_truth = true_diffusion(grid)

    radius = float(cfg["data"]["failure_radius"])
    paired_rng = np.random.default_rng(seed + 30)
    test_x0 = dataset["test"]["x0"]
    paired_noise = paired_rng.normal(size=(len(test_x0), n_steps(cfg), 2))
    sim_truth = _simulate_law(true_drift, true_diffusion, test_x0, dt, paired_noise, radius)
    sim_teacher = _simulate_law(teacher.drift, teacher.diffusion, test_x0, dt, paired_noise,
                                radius)
    sim_symbolic = _simulate_law(drift_field.predict, sigma_field.predict, test_x0, dt,
                                 paired_noise, radius)
    cov_truth_pair = terminal_cov(sim_truth["states"])[0]
    cov_teacher_pair = terminal_cov(sim_teacher["states"])[0]
    cov_symbolic_pair = terminal_cov(sim_symbolic["states"])[0]

    law_rng = np.random.default_rng(seed + 40)
    law_n = int(cfg["data"]["law_paths"])
    law_x0 = np.repeat(np.asarray(cfg["data"]["x0_xy"], dtype=np.float64)[None, :], law_n, axis=0)
    law_steps = n_steps(cfg)
    noise_truth = law_rng.normal(size=(law_n, law_steps, 2))
    noise_floor = law_rng.normal(size=(law_n, law_steps, 2))
    noise_teacher = law_rng.normal(size=(law_n, law_steps, 2))
    noise_symbolic = law_rng.normal(size=(law_n, law_steps, 2))
    law_truth = _simulate_law(true_drift, true_diffusion, law_x0, dt, noise_truth, radius)
    law_floor = _simulate_law(true_drift, true_diffusion, law_x0, dt, noise_floor, radius)
    law_teacher = _simulate_law(teacher.drift, teacher.diffusion, law_x0, dt, noise_teacher,
                                radius)
    law_symbolic = _simulate_law(drift_field.predict, sigma_field.predict, law_x0, dt,
                                 noise_symbolic, radius)
    disc = discretization_check(teacher, law_x0[: int(cfg["data"]["paired_paths"])], cfg,
                                np.random.default_rng(seed + 50))
    u_pre = uncertainty_probe(FrozenTeacher(pretrained, mean, scale, context),
                              dataset["train"]["states"], cfg)
    u_selected = uncertainty_probe(teacher, dataset["train"]["states"], cfg)
    floor_nll = float(np.mean(gaussian_nll(
        val_dy, true_drift(val_x), true_diffusion(val_x), dt)))
    train_scales = step_scales(dataset["train"]["states"], dt)
    test_scales = step_scales(dataset["test"]["states"], dt)

    metrics = {
        "drift_sym_teacher": drift_numbers["sym_teacher"],
        "drift_teacher_truth": drift_numbers["teacher_truth"],
        "drift_sym_truth": drift_numbers["sym_truth"],
        "sigma_sym_teacher": sigma_numbers["sym_teacher"],
        "sigma_teacher_truth": sigma_numbers["teacher_truth"],
        "sigma_sym_truth": sigma_numbers["sym_truth"],
        "variance_sym_teacher": variance_sym,
        "fraction_sigma_nonpositive": fraction_nonpositive,
        "vif": vif_max(queries["train"]["features"], int(distill["vif_degree"])),
        "fail_truth": float(sim_truth["failure_rate"]),
        "fail_teacher": float(sim_teacher["failure_rate"]),
        "fail_symbolic": float(sim_symbolic["failure_rate"]),
        "paired_sym_teacher": paired_path_rmse(sim_symbolic["states"], sim_teacher["states"]),
        "paired_teacher_truth": paired_path_rmse(sim_teacher["states"], sim_truth["states"]),
        "cov_sym_teacher": rel_frob(cov_symbolic_pair, cov_teacher_pair),
        "cov_teacher_truth": rel_frob(cov_teacher_pair, cov_truth_pair),
        "disc_rmse": disc["disc_rmse"],
        "disc_retained": disc["disc_retained"],
        "law_fail_truth": float(law_truth["failure_rate"]),
        "law_fail_teacher": float(law_teacher["failure_rate"]),
        "law_fail_symbolic": float(law_symbolic["failure_rate"]),
        "law_mean_teacher": mean_path_rmse(mean_path(law_teacher["states"]),
                                           mean_path(law_truth["states"])),
        "law_mean_symbolic": mean_path_rmse(mean_path(law_symbolic["states"]),
                                            mean_path(law_truth["states"])),
        "law_cov_teacher": rel_frob(terminal_cov(law_teacher["states"])[0],
                                    terminal_cov(law_truth["states"])[0]),
        "law_cov_symbolic": rel_frob(terminal_cov(law_symbolic["states"])[0],
                                     terminal_cov(law_truth["states"])[0]),
        "floor_mean": mean_path_rmse(mean_path(law_floor["states"]),
                                     mean_path(law_truth["states"])),
        "floor_cov": rel_frob(terminal_cov(law_floor["states"])[0],
                              terminal_cov(law_truth["states"])[0]),
        "u_rise": u_pre["u_rise"],
        "penalty_ratio": pre_stats["penalty_ratio"],
    }
    verdicts = decide(cfg, metrics)
    interp = cfg["interpretation"]
    oracle_drift_nrmse = nrmse(test["drift_truth"], oracle_drift.predict(test["features"]))
    oracle_sigma_nrmse = nrmse(test["sigma_truth"], oracle_sigma.predict(test["features"]))
    oracle_variance_nrmse = nrmse(test["sigma_truth"] ** 2,
                                  oracle_variance.predict(test["features"]))
    ontraj_resid = float(np.sqrt(np.mean((test["drift"] - test["drift_truth"]) ** 2)))
    grid_resid = float(np.sqrt(np.mean((grid_drift_t - grid_drift_truth) ** 2)))
    flags = {
        "drift_oracle_exact": oracle_drift_nrmse <= float(interp["oracle_exact_nrmse_max"]),
        "sigma_library_not_closed_form": oracle_sigma_nrmse
        > float(interp["diffusion_sigma_library_ceiling_above"]),
        "variance_oracle_exact": oracle_variance_nrmse <= float(interp["oracle_exact_nrmse_max"]),
        "cubic_rms_ratio": field_rms(groups["cubic"]) / max(field_rms(groups["full"]), 1e-8),
        "radial_rms_ratio": field_rms(groups["radial"]) / max(field_rms(groups["full"]), 1e-8),
        "small_cubic": field_rms(groups["cubic"]) / max(field_rms(groups["full"]), 1e-8)
        < float(interp["cubic_rms_ratio_small"]),
        "nll_gap": scores[chosen] - floor_nll,
        "nll_at_floor": (scores[chosen] - floor_nll) < float(interp["nll_floor_gap_max"]),
        "grid_over_ontraj": grid_resid / max(ontraj_resid, 1e-8),
        "grid_exceeds_ontraj": grid_resid / max(ontraj_resid, 1e-8)
        >= float(interp["grid_over_ontraj_rms_min"]),
        "coefficients_are_closed_form": False,
    }
    probe = teacher.predict(np.array([[0.2, -0.4], [1.5, 1.5], [-0.7, 0.3]]))
    shape_log = {
        "query": [3, 2],
        "drift": list(probe[0].shape),
        "diffusion": list(probe[1].shape),
        "uncertainty": list(probe[2].shape),
        "context": list(context.shape),
        "torch_numpy_max_abs": float(np.max(np.abs(
            probe[0] - numpy_forward(
                (np.array([[0.2, -0.4], [1.5, 1.5], [-0.7, 0.3]]) - mean) / scale,
                context, state_dict_numpy(models[chosen]),
                float(cfg["model"]["sigma_floor"]))[0] * scale))),
    }

    symbolic_tex = tex_equations(
        expand_to_original(drift_field) + expand_to_original(sigma_field),
        ("f_{1}", "f_{2}", "\\sigma_{1}", "\\sigma_{2}"),
        ("x_{1}", "x_{2}"))
    results = {
        "paper": cfg["paper"],
        "seed": seed,
        "chosen_candidate": chosen,
        "selection_val_nll": scores,
        "selection_field_val": selection_field,
        "finetune": {"best_step": fine_info["best_step"], "history": fine_info["history"]},
        "scratch": {"best_step": scratch_info["best_step"], "history": scratch_info["history"]},
        "pretrain": pre_stats,
        "verdicts": verdicts,
        "metrics": metrics,
        "drift": drift_numbers,
        "diffusion": sigma_numbers,
        "drift_selection": drift_meta,
        "diffusion_selection": sigma_meta,
        "oracle": {
            "drift_vs_truth": oracle_drift_nrmse,
            "sigma_vs_truth": oracle_sigma_nrmse,
            "variance_degree2_vs_truth": oracle_variance_nrmse,
            "drift_ols_vs_teacher": nrmse(test["drift"], ceiling_drift.predict(test["features"])),
            "sigma_ols_vs_teacher": nrmse(test["sigma"], ceiling_sigma.predict(test["features"])),
            "truth_key_monomials": named_coefficients(oracle_drift, key_names),
            "teacher_key_monomials": named_coefficients(ceiling_drift, key_names),
            "symbolic_key_monomials": named_coefficients(drift_field, key_names),
        },
        "groups_rms": {name: field_rms(value) for name, value in groups.items()},
        "grid": {
            "drift_rms_sym_teacher": float(np.sqrt(np.mean((grid_drift_s - grid_drift_t) ** 2))),
            "drift_rms_teacher_truth": grid_resid,
            "drift_rms_sym_truth": float(np.sqrt(np.mean((grid_drift_s - grid_drift_truth) ** 2))),
            "sigma_rms_sym_teacher": float(np.sqrt(np.mean((grid_sigma_s - grid_sigma_t) ** 2))),
            "sigma_rms_teacher_truth": float(np.sqrt(np.mean((grid_sigma_t - grid_sigma_truth) ** 2))),
            "sigma_rms_sym_truth": float(np.sqrt(np.mean((grid_sigma_s - grid_sigma_truth) ** 2))),
            "ontraj_drift_resid_rms_teacher_truth": ontraj_resid,
            "box_low": grid.min(axis=0).tolist(),
            "box_high": grid.max(axis=0).tolist(),
        },
        "uncertainty_pretrained": u_pre,
        "uncertainty_selected": u_selected,
        "nll_floor_val": floor_nll,
        "step_scales_train": train_scales,
        "step_scales_test": test_scales,
        "flags": flags,
        "failure_classes": {
            "truth": failure_counts(sim_truth["failure_class"]),
            "teacher": failure_counts(sim_teacher["failure_class"]),
            "symbolic": failure_counts(sim_symbolic["failure_class"]),
        },
        "shapes": shape_log,
        "standardizer": {"mean": mean.tolist(), "scale": scale.tolist()},
        "dropped_train_rows": int((~train_ok).sum()),
        "dropped_val_rows": int((~val_ok).sum()),
        "query_counts": {name: int(len(queries[name]["features"])) for name in queries},
        "runtime_seconds": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "git_head": git_head(),
        "symbolic_tex": symbolic_tex,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    (OUT_DIR / "symbolic_sde.tex").write_text(symbolic_tex + "\n")
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": monomial_table(drift_field),
        "diffusion": monomial_table(sigma_field),
    }, indent=2))
    drift_field.save(OUT_DIR / "symbolic_f.json")
    sigma_field.save(OUT_DIR / "symbolic_g.json")
    payload = {"candidate": chosen, "sigma_floor": float(cfg["model"]["sigma_floor"]),
               "hidden": int(cfg["model"]["hidden"]), "mean": mean.tolist(),
               "scale": scale.tolist(), "context": context.tolist(),
               "state_dict": {key: value.tolist()
                              for key, value in state_dict_numpy(models[chosen]).items()}}
    (OUT_DIR / "teacher_fim_weights.json").write_text(json.dumps(payload))
    print(json.dumps(jsonify({
        "chosen": chosen, "val_nll": scores, "verdicts": verdicts,
        "drift_teacher_truth": drift_numbers["teacher_truth"],
        "sigma_teacher_truth": sigma_numbers["teacher_truth"],
        "runtime": results["runtime_seconds"],
    }), indent=2))
    return results


if __name__ == "__main__":
    main()
