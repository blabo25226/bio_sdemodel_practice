"""Day 14: Koprulu, Djeumou & Topcu (ICLR 2025) NUNO -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day14.json``):

1. Data: controlled point mass with cubic drag, observable (q, v), scalar action u.
   The closed form is the data law. It is not a gene-regulatory equation.
2. Teacher: physics-structured neural drift and distance-aware velocity diffusion
   (NUNO equations 5-8). An aleatoric-only diffusion with the same budget is the
   H6 baseline (Appendix D, NUNO-al).
3. Distillation: direct teacher queries, trajectory-disjoint splits, power library
   in (q, v, u), validation-only selection, test once.
4. Simulation keeps failed-trajectory mass and classifies non-finite states and
   escapes from the pre-registered ball.

Run from the repository root:
``python -m src.daily_neural_sde.day14_koprulu_nuno``

Outputs land in ``outputs/daily_neural_sde/day14/``.
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
                     fit_polynomial_field, monomial_powers, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day14.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day14"

DriftFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
DiffFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-14 config."""
    return json.loads(path.read_text())


def n_steps(cfg: dict[str, Any]) -> int:
    """Number of Euler intervals. ``T`` must be a multiple of ``dt``."""
    t = float(cfg["data"]["T"])
    dt = float(cfg["data"]["dt"])
    steps = int(round(t / dt))
    if abs(steps * dt - t) > 1e-9:
        raise ValueError("T is not a multiple of dt")
    return steps


def true_drift(q: np.ndarray, v: np.ndarray, u: np.ndarray, b: float, c: float,
               d: float) -> np.ndarray:
    """Closed-form drift, shape (n, 2), columns (q-dot, v-dot).

    ``q``, ``v``, ``u`` are shape (n,).
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    return np.stack([v, b * u - c * v - d * v ** 3], axis=1)


def true_diffusion(q: np.ndarray, sigma: float) -> np.ndarray:
    """Closed-form velocity diffusion, shape (n,). Position diffusion is 0."""
    return np.full(np.asarray(q).shape, float(sigma), dtype=np.float64)


def behavior_action(q: np.ndarray, v: np.ndarray, noise: np.ndarray,
                    cfg: dict[str, Any]) -> np.ndarray:
    """Clipped linear feedback plus the supplied noise. Shape matches ``q``."""
    policy = cfg["data"]["behavior"]
    raw = (-float(policy["k_p"]) * q - float(policy["k_d"]) * v
           + np.asarray(noise, dtype=np.float64))
    return np.clip(raw, float(policy["u_min"]), float(policy["u_max"]))


def sample_x0(rng: np.random.Generator, n: int, cfg: dict[str, Any]) -> np.ndarray:
    """Initial states, shape (n, 2), from the frozen uniform boxes."""
    data = cfg["data"]
    q = rng.uniform(data["x0_q"][0], data["x0_q"][1], size=n)
    v = rng.uniform(data["x0_v"][0], data["x0_v"][1], size=n)
    return np.stack([q, v], axis=1)


def _mark_failures(x: np.ndarray, failed: np.ndarray,
                   failure_class: list[str | None], radius: float) -> np.ndarray:
    """Latch new failures. Non-finite takes priority over the radius."""
    bad_nonfinite = ~np.isfinite(x).all(axis=1)
    finite = np.isfinite(x).all(axis=1)
    bad_radius = finite & (np.linalg.norm(x, axis=1) > radius)
    newly = (~failed) & (bad_nonfinite | bad_radius)
    for idx in np.flatnonzero(newly):
        failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
    return failed | newly


def simulate_controlled(drift_fn: DriftFn, diffusion_fn: DiffFn, x0: np.ndarray,
                        actions: np.ndarray, dt: float, increments: np.ndarray,
                        failure_radius: float) -> dict[str, Any]:
    """Euler-Maruyama with an exogenous action tape.

    ``x0`` is (n, 2), ``actions`` and ``increments`` are (n, n_steps).
    ``drift_fn(x, u)`` returns (n, 2). ``diffusion_fn(x, u)`` returns (n,).
    Failed rows stay NaN and keep their mass in ``failure_rate``.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    actions = np.asarray(actions, dtype=np.float64)
    increments = np.asarray(increments, dtype=np.float64)
    n, n_steps_ = actions.shape
    if increments.shape != (n, n_steps_) or x.shape != (n, 2):
        raise ValueError("shape mismatch")
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    failed = _mark_failures(x, failed, failure_class, failure_radius)
    x[failed] = np.nan
    observed = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps_):
        active = np.flatnonzero(~failed)
        if len(active):
            drift = np.asarray(drift_fn(x[active], actions[active, step]), dtype=np.float64)
            diffusion = np.asarray(diffusion_fn(x[active], actions[active, step]),
                                   dtype=np.float64).reshape(-1)
            if drift.shape != (len(active), 2) or diffusion.shape != (len(active),):
                raise ValueError("field shape mismatch")
            x[active, 0] = x[active, 0] + drift[:, 0] * dt
            x[active, 1] = (x[active, 1] + drift[:, 1] * dt
                            + diffusion * sqrt_dt * increments[active, step])
        failed = _mark_failures(x, failed, failure_class, failure_radius)
        x[failed] = np.nan
        observed.append(x.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class, "failure_rate": float(np.mean(failed))}


def generate_truth_paths(n: int, cfg: dict[str, Any], rng: np.random.Generator
                         ) -> dict[str, np.ndarray]:
    """Closed-loop Euler paths of the cubic-drag law. Shapes (n, steps+1, 2) and (n, steps)."""
    data = cfg["data"]
    dt = float(data["dt"])
    steps = n_steps(cfg)
    b, c, d = float(data["b"]), float(data["c"]), float(data["d"])
    sigma = float(data["sigma"])
    x = sample_x0(rng, n, cfg)
    states = np.empty((n, steps + 1, 2), dtype=np.float64)
    actions = np.empty((n, steps), dtype=np.float64)
    increments = rng.standard_normal((n, steps))
    noises = rng.normal(0.0, float(data["behavior"]["noise_std"]), size=(n, steps))
    states[:, 0, :] = x
    sqrt_dt = math.sqrt(dt)
    for step in range(steps):
        u = behavior_action(x[:, 0], x[:, 1], noises[:, step], cfg)
        actions[:, step] = u
        drift = true_drift(x[:, 0], x[:, 1], u, b, c, d)
        x = np.column_stack([
            x[:, 0] + drift[:, 0] * dt,
            x[:, 1] + drift[:, 1] * dt + sigma * sqrt_dt * increments[:, step],
        ])
        states[:, step + 1, :] = x
    return {"states": states, "actions": actions, "increments": increments}


def make_dataset(cfg: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Trajectory-disjoint train, validation, and test logs. One RNG stream, in that order."""
    split = cfg["split"]
    rng = np.random.default_rng(int(split["seed"]))
    out = {}
    for name in ("train", "validation", "test"):
        out[name] = generate_truth_paths(int(split[name]), cfg, rng)
    return out


def normalization_from_train(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of training (q, v, u). ``states`` is (n, steps+1, 2)."""
    qv = states[:, :-1, :].reshape(-1, 2)
    rows = np.concatenate([qv, actions.reshape(-1, 1)], axis=1)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def strong_convexity_deficit(eta_y: torch.Tensor, eta_x: torch.Tensor,
                             grad_x: torch.Tensor, delta: torch.Tensor,
                             mu: torch.Tensor) -> torch.Tensor:
    """Local strong-convexity slack, shape (n, 1).

    The printed NUNO formula dots the gradient with (x - y). The inequality
    named in the text is the opposite direction, so this uses (y - x).
    ``delta`` is y - x. Positive slack means the sample satisfies the constraint.
    """
    linear = (grad_x * delta).sum(dim=-1, keepdim=True)
    return eta_y - eta_x - linear - mu * (delta ** 2).sum(dim=-1, keepdim=True)


class _MLP(nn.Module):
    """Two hidden layers. ``activation`` is ``tanh`` or ``silu``."""

    def __init__(self, in_dim: int, width: int, out_dim: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, out_dim)
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shape (n, in_dim) -> (n, out_dim)."""
        h = self._act(self.fc1(x))
        h = self._act(self.fc2(h))
        return self.fc3(h)

    def _act(self, h: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(h)
        if self.activation_name == "silu":
            return torch.nn.functional.silu(h)
        raise ValueError(self.activation_name)


class NunoModel(nn.Module):
    """Physics-structured drift and diagonal velocity diffusion.

    Position drift equals the velocity. Position diffusion is identically 0.
    """

    def __init__(self, cfg: dict[str, Any], epistemic: bool):
        super().__init__()
        teacher = cfg["teacher"]
        width = int(teacher["width"])
        mu_width = int(teacher["mu_width"])
        self.epistemic = bool(epistemic)
        self.g_floor = float(teacher["g_floor"])
        self.wmax = float(teacher["wmax"])
        self.mu_max = float(teacher["mu_max"])
        self.G = _MLP(1, width, 1, teacher["activation_drift"])
        self.H = _MLP(2, width, 1, teacher["activation_drift"])
        self.res = _MLP(3, width, 1, teacher["activation_drift"])
        self.sigma = _MLP(3, width, 1, teacher["activation_sigma"])
        if self.epistemic:
            self.eta = _MLP(3, width, 1, teacher["activation_eta"])
            self.mu_net = _MLP(3, mu_width, 1, teacher["activation_eta"])
            self.h_w_raw = nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.h_b = nn.Parameter(torch.tensor(float(teacher["h_bias_init"]), dtype=torch.float64))
        else:
            self.eta = None
            self.mu_net = None
        self.register_buffer("mean", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("std", torch.ones(3, dtype=torch.float64))
        self._init_weights(float(teacher["sigma_softplus_at_init"]))
        self.double()

    def _init_weights(self, sigma_level: float) -> None:
        nets = [self.G, self.H, self.res, self.sigma]
        if self.eta is not None:
            nets.extend([self.eta, self.mu_net])
        for net in nets:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, -0.05, 0.05)
                    nn.init.zeros_(module.bias)
        # softplus(bias) = sigma_level, before g_floor is added.
        self.sigma.fc3.bias.data.fill_(math.log(math.expm1(sigma_level)))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Copy training-only (q, v, u) moments into the input buffers."""
        self.mean.copy_(torch.tensor(mean, dtype=torch.float64))
        self.std.copy_(torch.tensor(std, dtype=torch.float64))

    def _z(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raw = torch.stack([q, v, u], dim=-1)
        return (raw - self.mean) / self.std

    def aleatoric(self, z: torch.Tensor) -> torch.Tensor:
        """Positive aleatoric velocity scale, shape (n,)."""
        raw = self.sigma(z).squeeze(-1)
        return torch.nn.functional.softplus(raw) + self.g_floor

    def eta_value(self, z: torch.Tensor) -> torch.Tensor:
        """Distance-aware scalar, shape (n, 1). The aleatoric model returns zeros."""
        if self.eta is None:
            return torch.zeros(z.shape[0], 1, dtype=z.dtype, device=z.device)
        return self.eta(z)

    def epistemic_diffusion(self, eta: torch.Tensor) -> torch.Tensor:
        """Bounded monotone map h(eta), shape (n,). Zero when the model is aleatoric-only."""
        if not self.epistemic:
            return torch.zeros(eta.shape[0], dtype=eta.dtype, device=eta.device)
        slope = 1.0 + torch.nn.functional.softplus(self.h_w_raw)
        return self.wmax * torch.sigmoid(slope * eta.squeeze(-1) + self.h_b)

    def mu(self, z: torch.Tensor) -> torch.Tensor:
        """Capped strong-convexity coefficient, shape (n, 1)."""
        if self.mu_net is None:
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        return self.mu_max * torch.sigmoid(self.mu_net(z))

    def fields(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return drift (n, 2), velocity diffusion (n,), eta (n, 1), residual (n,)."""
        z = self._z(q, v, u)
        gain = self.G(z[:, 1:2]).squeeze(-1)
        damping = self.H(z[:, 0:2]).squeeze(-1)
        residual = self.res(z).squeeze(-1)
        drift_v = gain * u + damping * v + residual
        drift = torch.stack([v, drift_v], dim=-1)
        eta = self.eta_value(z)
        diffusion = self.aleatoric(z) + self.epistemic_diffusion(eta)
        return drift, diffusion, eta, residual


def velocity_nll(drift_v: torch.Tensor, diffusion: torch.Tensor, v: torch.Tensor,
                 v_next: torch.Tensor, dt: float) -> torch.Tensor:
    """Euler-Maruyama Gaussian NLL of one velocity increment, shape (n,)."""
    var = diffusion ** 2 * dt
    resid = v_next - v - drift_v * dt
    return 0.5 * (resid ** 2 / var + torch.log(var))


def uncertainty_losses(model: NunoModel, q: torch.Tensor, v: torch.Tensor,
                       u: torch.Tensor, radius: float, n_pairs: int
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_sc, L_grad, L_mu at the data anchors, with ``n_pairs`` local samples each."""
    z = model._z(q, v, u)
    eta_data, grad_data = _eta_and_grad(model, z)
    loss_grad = (grad_data ** 2).sum(dim=-1).mean() + (eta_data ** 2).mean()
    mu = model.mu(z)
    loss_mu = (1.0 / mu).mean()
    eps1 = torch.randn(n_pairs, *z.shape, dtype=torch.float64)
    eps2 = torch.randn(n_pairs, *z.shape, dtype=torch.float64)
    z1 = (z.detach() + radius * eps1).reshape(-1, 3)
    z2 = (z.detach() + radius * eps2).reshape(-1, 3)
    eta1, grad1 = _eta_and_grad(model, z1)
    eta2 = model.eta_value(z2)
    delta = z2 - z1
    mu_pairs = mu.reshape(1, -1, 1).expand(n_pairs, -1, 1).reshape(-1, 1)
    deficit = strong_convexity_deficit(eta2, eta1, grad1, delta, mu_pairs)
    loss_sc = torch.clamp(deficit, max=0.0).pow(2).mean()
    return loss_sc, loss_grad, loss_mu


def _eta_and_grad(model: NunoModel, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """eta and d(eta)/dz with the graph kept for the parameter step. Shapes (n, 1), (n, 3)."""
    z = z.detach().requires_grad_(True)
    eta = model.eta_value(z)
    grad, = torch.autograd.grad(eta.sum(), z, create_graph=True)
    return eta, grad


def _batch_tensors(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64) for key, value in batch.items()}


def window_loss(model: NunoModel, batch: dict[str, torch.Tensor], cfg: dict[str, Any]
                ) -> tuple[torch.Tensor, dict[str, float]]:
    """Horizon-2 teacher-forced velocity NLL plus the frozen penalty weights."""
    teacher = cfg["teacher"]
    dt = float(cfg["data"]["dt"])
    nlls = []
    residuals = []
    for step in range(int(teacher["horizon_transitions"])):
        drift, diffusion, _eta, residual = model.fields(
            batch["q"][:, step], batch["v"][:, step], batch["u"][:, step])
        nlls.append(velocity_nll(drift[:, 1], diffusion, batch["v"][:, step],
                                 batch["v"][:, step + 1], dt))
        residuals.append(residual)
    data = torch.stack(nlls, dim=0).mean()
    res_pen = torch.stack(residuals, dim=0).pow(2).mean()
    total = float(teacher["lambda_data"]) * data + float(teacher["lambda_res"]) * res_pen
    parts = {"data": float(data.detach()), "res": float(res_pen.detach()),
             "sc": 0.0, "grad": 0.0, "mu": 0.0}
    if model.epistemic:
        sc, grad, mu = uncertainty_losses(
            model, batch["q"][:, 0], batch["v"][:, 0], batch["u"][:, 0],
            float(teacher["ball_radius"]), int(teacher["n_pairs"]))
        total = (total + float(teacher["lambda_sc"]) * sc
                 + float(teacher["lambda_grad"]) * grad
                 + float(teacher["lambda_mu"]) * mu)
        parts["sc"] = float(sc.detach())
        parts["grad"] = float(grad.detach())
        parts["mu"] = float(mu.detach())
    return total, parts


def build_windows(states: np.ndarray, actions: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Teacher-forced windows. ``q`` has shape (n_windows, horizon+1)."""
    # states: (n, steps+1, 2), actions: (n, steps)
    n_traj, steps_p1, _ = states.shape
    n_start = steps_p1 - horizon
    if n_start <= 0:
        raise ValueError("horizon exceeds the trajectory")
    q = np.empty((n_traj * n_start, horizon + 1), dtype=np.float64)
    v = np.empty_like(q)
    u = np.empty((n_traj * n_start, horizon), dtype=np.float64)
    cursor = 0
    for traj in range(n_traj):
        for start in range(n_start):
            sl = slice(start, start + horizon + 1)
            q[cursor] = states[traj, sl, 0]
            v[cursor] = states[traj, sl, 1]
            u[cursor] = actions[traj, start:start + horizon]
            cursor += 1
    return {"q": q, "v": v, "u": u}


def validation_nll(model: NunoModel, states: np.ndarray, actions: np.ndarray,
                   dt: float) -> float:
    """Mean teacher-forced velocity NLL on every stored transition."""
    model.eval()
    q = torch.tensor(states[:, :-1, 0].reshape(-1), dtype=torch.float64)
    v = torch.tensor(states[:, :-1, 1].reshape(-1), dtype=torch.float64)
    u = torch.tensor(actions.reshape(-1), dtype=torch.float64)
    v_next = torch.tensor(states[:, 1:, 1].reshape(-1), dtype=torch.float64)
    with torch.no_grad():
        drift, diffusion, _eta, _res = model.fields(q, v, u)
        nll = velocity_nll(drift[:, 1], diffusion, v, v_next, dt)
    model.train()
    return float(nll.mean())


def train_model(epistemic: bool, train: dict[str, np.ndarray],
                validation: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray,
                cfg: dict[str, Any], seed: int) -> tuple[NunoModel, dict[str, Any]]:
    """Adam on horizon-2 windows. Restore the best validation velocity NLL."""
    torch.manual_seed(seed)
    model = NunoModel(cfg, epistemic).to(dtype=torch.float64)
    model.set_normalization(mean, std)
    teacher = cfg["teacher"]
    windows = build_windows(train["states"], train["actions"], int(teacher["horizon_transitions"]))
    n_windows = len(windows["q"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(teacher["lr"]),
                                 weight_decay=float(teacher["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = -1
    history = []
    dt = float(cfg["data"]["dt"])
    batch = int(teacher["batch_size"])
    model.train()
    for step in range(int(teacher["n_opt_steps"])):
        pick = rng.integers(0, n_windows, size=batch)
        batch_np = {key: value[pick] for key, value in windows.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, parts = window_loss(model, _batch_tensors(batch_np), cfg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(teacher["grad_clip"]))
        optimizer.step()
        record = (step % int(teacher["eval_every"]) == 0
                  or step == int(teacher["n_opt_steps"]) - 1)
        if record:
            score = validation_nll(model, validation["states"], validation["actions"], dt)
            history.append({"step": step, "train_loss": float(loss.detach()),
                            "val_nll": score, **parts})
            if score < best_score:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            print(f"  step {step} val_nll {score:.4f} data {parts['data']:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_step": best_step, "best_val_nll": best_score,
                   "history_tail": history[-3:], "n_windows": n_windows}


def _linear_layers(module: _MLP) -> list[dict[str, list[float]]]:
    layers = []
    for layer in (module.fc1, module.fc2, module.fc3):
        layers.append({"weight": layer.weight.detach().cpu().numpy().tolist(),
                       "bias": layer.bias.detach().cpu().numpy().tolist()})
    return layers


def pack_weights(model: NunoModel) -> dict[str, Any]:
    """JSON-ready teacher. Linear weights use PyTorch layout (out, in)."""
    payload: dict[str, Any] = {
        "epistemic": model.epistemic,
        "g_floor": model.g_floor,
        "wmax": model.wmax,
        "mu_max": model.mu_max,
        "mean": model.mean.detach().cpu().numpy().tolist(),
        "std": model.std.detach().cpu().numpy().tolist(),
        "G": _linear_layers(model.G),
        "H": _linear_layers(model.H),
        "res": _linear_layers(model.res),
        "sigma": _linear_layers(model.sigma),
        "activation_drift": model.G.activation_name,
        "activation_sigma": model.sigma.activation_name,
    }
    if model.epistemic:
        payload["eta"] = _linear_layers(model.eta)
        payload["mu"] = _linear_layers(model.mu_net)
        payload["activation_eta"] = model.eta.activation_name
        payload["h_w_raw"] = float(model.h_w_raw.detach())
        payload["h_b"] = float(model.h_b.detach())
    return payload


def _act_np(h: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(h)
    if name == "silu":
        return h / (1.0 + np.exp(-h))
    raise ValueError(name)


def _mlp_np(x: np.ndarray, layers: list[dict[str, list[float]]], activation: str) -> np.ndarray:
    h = np.asarray(x, dtype=np.float64)
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        weight = np.asarray(layer["weight"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        h = h @ weight.T + bias
        if i < last:
            h = _act_np(h, activation)
    return h


def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def teacher_fields(weights: dict[str, Any], q: np.ndarray, v: np.ndarray, u: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy replay. Returns drift (n, 2), velocity diffusion (n,), eta (n,)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    raw = np.stack([q, v, u], axis=1)
    mean = np.asarray(weights["mean"], dtype=np.float64)
    std = np.asarray(weights["std"], dtype=np.float64)
    z = (raw - mean) / std
    gain = _mlp_np(z[:, 1:2], weights["G"], weights["activation_drift"]).reshape(-1)
    damping = _mlp_np(z[:, 0:2], weights["H"], weights["activation_drift"]).reshape(-1)
    residual = _mlp_np(z, weights["res"], weights["activation_drift"]).reshape(-1)
    drift = np.stack([v, gain * u + damping * v + residual], axis=1)
    sigma = _softplus_np(_mlp_np(z, weights["sigma"], weights["activation_sigma"]).reshape(-1))
    sigma = sigma + float(weights["g_floor"])
    if weights["epistemic"]:
        eta = _mlp_np(z, weights["eta"], weights["activation_eta"]).reshape(-1)
        slope = 1.0 + _softplus_np(np.asarray(weights["h_w_raw"], dtype=np.float64))
        h = float(weights["wmax"]) / (1.0 + np.exp(-(slope * eta + float(weights["h_b"]))))
        diffusion = sigma + h
    else:
        eta = np.zeros(len(q), dtype=np.float64)
        diffusion = sigma
    return drift, diffusion, eta


def parity_error(model: NunoModel, weights: dict[str, Any], seed: int) -> float:
    """Max absolute gap between the torch fields and the NumPy replay."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=8)
    v = rng.normal(size=8)
    u = rng.normal(size=8)
    with torch.no_grad():
        drift_t, diff_t, eta_t, _res = model.fields(
            torch.tensor(q, dtype=torch.float64),
            torch.tensor(v, dtype=torch.float64),
            torch.tensor(u, dtype=torch.float64))
    drift_n, diff_n, eta_n = teacher_fields(weights, q, v, u)
    return float(max(np.max(np.abs(drift_t.numpy() - drift_n)),
                     np.max(np.abs(diff_t.numpy() - diff_n)),
                     np.max(np.abs(eta_t.numpy().reshape(-1) - eta_n))))


def query_rows(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nonterminal (q, v, u) and the feature matrix (n, 3)."""
    q = states[:, :-1, 0].reshape(-1)
    v = states[:, :-1, 1].reshape(-1)
    u = actions.reshape(-1)
    features = np.stack([q, v, u], axis=1)
    return q, v, u, features


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Row mask that is true where every array is finite."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array).all(axis=1) if array.ndim == 2 else np.isfinite(array)
    return mask


def vif_max(x: np.ndarray, degree: int) -> float:
    """Max VIF of the degree-``degree`` library. The constant column is excluded."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    powers = monomial_powers(x.shape[1], degree)
    theta = evaluate_powers((x - mean) / scale, powers)
    keep = [j for j, power in enumerate(powers) if power.sum() > 0]
    theta = theta[:, keep]
    worst = 1.0
    for j in range(theta.shape[1]):
        others = np.delete(theta, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, theta[:, j], rcond=None)
        resid = theta[:, j] - others @ coef
        ss = float(np.sum((theta[:, j] - theta[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss, 1e-15)
        worst = max(worst, 1.0 / max(1.0 - r2, 1e-9))
    return float(worst)


def select_field(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, degrees: list[int], thresholds: list[float],
                 ridge: float) -> tuple[PolynomialField, dict[str, Any]]:
    """Pick degree and threshold by pooled validation NRMSE. Test is not read."""
    best: PolynomialField | None = None
    best_val = math.inf
    best_meta: dict[str, Any] = {}
    candidates = []
    for degree in degrees:
        for threshold in thresholds:
            field = fit_polynomial_field(x_train, y_train, degree, threshold, ridge)
            val = nrmse(y_val, field.predict(x_val))
            candidates.append({"degree": degree, "threshold": threshold,
                               "validation_nrmse": val,
                               "active_terms": int(np.count_nonzero(field.coefficients))})
            if val < best_val:
                best_val = val
                best = field
                best_meta = {"degree": degree, "threshold": threshold,
                             "validation_nrmse": val,
                             "active_terms": int(np.count_nonzero(field.coefficients)),
                             "library_size": int(field.metadata["library_size"])}
    if best is None:
        raise RuntimeError("no candidate")
    best_meta["candidates"] = candidates
    return best, best_meta


def rms(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean square of the residual."""
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def mean_path(states: np.ndarray) -> np.ndarray:
    """Time-wise nanmean, shape (n_obs, d)."""
    return np.nanmean(states, axis=0)


def mean_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two mean paths, ignoring non-finite rows."""
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not ok.any():
        return math.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def terminal_cov(states: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Covariance of finite terminal states, and how many were retained."""
    final = states[:, -1, :]
    ok = np.isfinite(final).all(axis=1)
    if int(ok.sum()) < 3:
        return None, int(ok.sum())
    return np.cov(final[ok], rowvar=False), int(ok.sum())


def rel_frob(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    """||left - right||_F / ||right||_F. None if either covariance is missing."""
    if left is None or right is None:
        return None
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-8))


def closed_loop_rollout(drift_fn: DriftFn, diffusion_fn: DiffFn, x0: np.ndarray,
                        action_noise: np.ndarray, increments: np.ndarray, dt: float,
                        cfg: dict[str, Any], failure_radius: float) -> dict[str, Any]:
    """Behavior policy in the loop. ``action_noise`` and ``increments`` are (n, steps)."""
    x = np.array(x0, dtype=np.float64, copy=True)
    n, steps = increments.shape
    actions = np.full((n, steps), np.nan, dtype=np.float64)
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    failed = _mark_failures(x, failed, failure_class, failure_radius)
    x[failed] = np.nan
    observed = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    for step in range(steps):
        active = np.flatnonzero(~failed)
        if len(active):
            u = behavior_action(x[active, 0], x[active, 1], action_noise[active, step], cfg)
            actions[active, step] = u
            actions[failed, step] = np.nan
            drift = np.asarray(drift_fn(x[active], u), dtype=np.float64)
            diffusion = np.asarray(diffusion_fn(x[active], u), dtype=np.float64).reshape(-1)
            x[active, 0] = x[active, 0] + drift[:, 0] * dt
            x[active, 1] = (x[active, 1] + drift[:, 1] * dt
                            + diffusion * sqrt_dt * increments[active, step])
        failed = _mark_failures(x, failed, failure_class, failure_radius)
        x[failed] = np.nan
        observed.append(x.copy())
    return {"states": np.stack(observed, axis=1), "actions": actions, "failed": failed,
            "failure_class": failure_class, "failure_rate": float(np.mean(failed))}


def truth_callables(cfg: dict[str, Any]) -> tuple[DriftFn, DiffFn]:
    """Closed-form drift and velocity diffusion as simulator callbacks."""
    data = cfg["data"]
    b, c, d = float(data["b"]), float(data["c"]), float(data["d"])
    sigma = float(data["sigma"])

    def drift(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return true_drift(x[:, 0], x[:, 1], u, b, c, d)

    def diffusion(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return true_diffusion(x[:, 0], sigma)

    return drift, diffusion


def numpy_callables(weights: dict[str, Any]) -> tuple[DriftFn, DiffFn]:
    """Teacher or baseline fields as simulator callbacks."""

    def drift(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return teacher_fields(weights, x[:, 0], x[:, 1], u)[0]

    def diffusion(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return teacher_fields(weights, x[:, 0], x[:, 1], u)[1]

    return drift, diffusion


def polynomial_callables(drift_field: PolynomialField, diffusion_field: PolynomialField
                         ) -> tuple[DriftFn, DiffFn]:
    """Symbolic fields in (q, v, u)."""

    def drift(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        features = np.column_stack([x[:, 0], x[:, 1], u])
        return drift_field.predict(features)

    def diffusion(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        features = np.column_stack([x[:, 0], x[:, 1], u])
        return diffusion_field.predict(features).reshape(-1)

    return drift, diffusion


def percentile_grid(features: np.ndarray, n_side: int = 5) -> np.ndarray:
    """Padded 5^3 lattice on the training feature box. Shape (n_side**3, 3)."""
    axes = []
    for col in range(features.shape[1]):
        lo, hi = np.percentile(features[:, col], [5, 95])
        pad = 0.5 * (hi - lo)
        axes.append(np.linspace(lo - pad, hi + pad, n_side))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def ood_position_shift(features: np.ndarray) -> np.ndarray:
    """Training rows with q moved one training interquartile span past q_95."""
    lo, hi = np.percentile(features[:, 0], [5, 95])
    shifted = np.array(features, dtype=np.float64, copy=True)
    shifted[:, 0] = hi + 1.0 * (hi - lo)
    return shifted


def coarse_increments(fine: np.ndarray) -> np.ndarray:
    """Normalize the sum of fine standard normals so the coarse increment has variance 1.

    ``fine`` has shape (n, n_coarse, factor).
    """
    factor = fine.shape[-1]
    return fine.sum(axis=-1) / math.sqrt(factor)


def discretization_probe(drift_fn: DriftFn, diffusion_fn: DiffFn, x0: np.ndarray,
                         actions: np.ndarray, dt: float, fine: np.ndarray,
                         failure_radius: float) -> dict[str, float]:
    """Terminal RMSE of one coarse step size against ``fine_factor`` substeps.

    The action is held fixed across the substeps of each coarse interval.
    """
    coarse_eps = coarse_increments(fine)
    coarse = simulate_controlled(drift_fn, diffusion_fn, x0, actions, dt, coarse_eps,
                                 failure_radius)
    n, n_coarse, factor = fine.shape
    fine_actions = np.repeat(actions, factor, axis=1)
    fine_eps = fine.reshape(n, n_coarse * factor)
    fine_sim = simulate_controlled(drift_fn, diffusion_fn, x0, fine_actions, dt / factor,
                                   fine_eps, failure_radius)
    # Compare at coarse times. Fine states are stored every substep, so index (k+1)*factor.
    fine_states = fine_sim["states"][:, ::factor, :]
    both = np.isfinite(coarse["states"][:, -1, :]).all(axis=1) & np.isfinite(
        fine_states[:, -1, :]).all(axis=1)
    retained = float(np.mean(both))
    if both.any():
        gap = coarse["states"][both, -1, :] - fine_states[both, -1, :]
        rmse = float(np.sqrt(np.mean(gap ** 2)))
    else:
        rmse = math.nan
    return {"terminal_rmse": rmse, "retained_fraction": retained}


def component_nrmse(truth: np.ndarray, pred: np.ndarray, col: int) -> float:
    """NRMSE of one column."""
    return nrmse(truth[:, col], pred[:, col])


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Thresholds are read from ``cfg`` and not modified."""
    hyp = cfg["hypotheses"]
    h1, h2 = hyp["H1_drift"], hyp["H2_diffusion"]
    h4, h5 = hyp["H4_simulation"], hyp["H5_law"]
    h6, h7 = hyp["H6_aleatoric_baseline"], hyp["H7_discretization_gate"]
    verdicts: dict[str, Any] = {
        "H1": _pass(
            metrics["drift_v_sym_teacher"] <= h1["velocity_symbolic_vs_teacher_test_nrmse_max"]
            and metrics["drift_v_teacher_truth"] <= h1["velocity_teacher_vs_truth_test_nrmse_max"]
            and metrics["drift_v_sym_truth"] <= h1["velocity_symbolic_vs_truth_test_nrmse_max"]
            and metrics["drift_q_sym_teacher"] <= h1["position_symbolic_vs_teacher_test_nrmse_max"]),
        "H2": _pass(
            metrics["g_sym_teacher"] <= h2["symbolic_vs_teacher_test_nrmse_max"]
            and metrics["g_teacher_truth"] <= h2["teacher_vs_truth_test_nrmse_max"]
            and metrics["g_sym_truth"] <= h2["symbolic_vs_truth_test_nrmse_max"]
            and metrics["sigma_sym_teacher"] <= h2["symbolic_vs_teacher_sigma_nrmse_max"]),
        "H2c": _pass(metrics["fraction_g_nonpositive"]
                     <= hyp["H2c_range"]["fraction_symbolic_g_nonpositive_max"]),
        "H3": ("indeterminate" if metrics["vif"] > hyp["H3_collinearity_gate"]["vif_max"]
               else "pass"),
    }
    fail_ok = (metrics["fail_truth"] <= h4["failure_rate_max"]
               and metrics["fail_teacher"] <= h4["failure_rate_max"]
               and metrics["fail_symbolic"] <= h4["failure_rate_max"]
               and abs(metrics["fail_symbolic"] - metrics["fail_teacher"])
               <= h4["failure_rate_gap_max"])
    h7_ok = (math.isfinite(metrics["disc_rmse"])
             and metrics["disc_rmse"] <= h7["teacher_coarse_vs_fine_terminal_rmse_max"]
             and metrics["disc_retained"] >= h7["min_retained_fraction"])
    path_ok = (math.isfinite(metrics["mean_sym_teacher"])
               and math.isfinite(metrics["mean_teacher_truth"])
               and metrics["mean_sym_teacher"] <= h4["paired_mean_rmse_symbolic_vs_teacher_max"]
               and metrics["mean_teacher_truth"] <= h4["paired_mean_rmse_teacher_vs_truth_max"])
    cov_sym = metrics["cov_sym_teacher"]
    cov_truth = metrics["cov_teacher_truth"]
    cov_ok = (cov_sym is not None and cov_truth is not None
              and cov_sym <= h4["terminal_cov_rel_frob_symbolic_vs_teacher_max"]
              and cov_truth <= h4["terminal_cov_rel_frob_teacher_vs_truth_max"])
    drift_ok = (metrics["drift_v_teacher_truth"]
                <= h4["velocity_teacher_vs_truth_test_nrmse_max"])
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
                    or floor_mean > h5["mean_path_rmse_teacher_vs_observed_max"]
                    or floor_cov is None
                    or not math.isfinite(floor_cov)
                    or floor_cov > h5["terminal_cov_rel_frob_teacher_vs_observed_max"])
    law_numbers_ok = (
        math.isfinite(metrics["law_mean_teacher"])
        and math.isfinite(metrics["law_mean_symbolic"])
        and metrics["law_cov_teacher"] is not None
        and metrics["law_cov_symbolic"] is not None
        and metrics["law_mean_teacher"] <= h5["mean_path_rmse_teacher_vs_observed_max"]
        and metrics["law_mean_symbolic"] <= h5["mean_path_rmse_symbolic_vs_observed_max"]
        and metrics["law_cov_teacher"] <= h5["terminal_cov_rel_frob_teacher_vs_observed_max"]
        and metrics["law_cov_symbolic"] <= h5["terminal_cov_rel_frob_symbolic_vs_observed_max"])
    if law_fail > h5["failure_gate"] or floor_blocks:
        verdicts["H5"] = "indeterminate"
    else:
        verdicts["H5"] = _pass(law_numbers_ok)
    shared = (metrics["g_full_truth"] >= h6["shared_floor_nrmse"]
              and metrics["g_aleatoric_truth"] >= h6["shared_floor_nrmse"])
    lift_gap = metrics["g_lift_full"] - metrics["g_lift_aleatoric"]
    verdicts["H6"] = _pass(
        (not shared)
        and metrics["g_full_truth"] <= h6["full_id_nrmse_max"]
        and metrics["h_ood_minus_id"] >= h6["h_ood_minus_id_min"]
        and lift_gap >= h6["g_lift_gap_min"])
    verdicts["H6_shared_floor"] = bool(shared)
    return verdicts


def to_jsonable(obj: Any) -> Any:
    """Recursively replace NumPy scalars and arrays."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def git_head() -> str:
    """Current commit, or an empty string if git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def loss_snapshot(model: NunoModel, states: np.ndarray, actions: np.ndarray,
                  cfg: dict[str, Any], n_windows: int = 128) -> dict[str, float]:
    """Training-loss parts on a fixed prefix of the training windows, after selection."""
    windows = build_windows(states, actions, int(cfg["teacher"]["horizon_transitions"]))
    take = min(n_windows, len(windows["q"]))
    batch = {key: value[:take] for key, value in windows.items()}
    model.train()
    total, parts = window_loss(model, _batch_tensors(batch), cfg)
    model.eval()
    parts["total"] = float(total.detach())
    parts["n_windows"] = float(take)
    return parts


def main() -> dict[str, Any]:
    """Train, distill, validate once, and write the day-14 outputs."""
    started = time.time()
    cfg = load_config()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    print("generating cubic-drag logs", flush=True)
    dataset = make_dataset(cfg)
    mean, std = normalization_from_train(dataset["train"]["states"], dataset["train"]["actions"])
    print("training full NUNO teacher", flush=True)
    full_model, full_log = train_model(True, dataset["train"], dataset["validation"],
                                       mean, std, cfg, int(cfg["seed"]))
    print("training aleatoric-only baseline", flush=True)
    aleatoric_model, aleatoric_log = train_model(
        False, dataset["train"], dataset["validation"], mean, std, cfg, int(cfg["seed"]))
    full_weights = pack_weights(full_model)
    aleatoric_weights = pack_weights(aleatoric_model)
    parity = parity_error(full_model, full_weights, int(cfg["seed"]) + 7)
    if parity > 1e-8:
        raise RuntimeError(f"numpy/torch field mismatch {parity}")
    with torch.no_grad():
        probe_q = torch.zeros(8, dtype=torch.float64)
        probe_v = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)
        probe_u = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)
        probe_f, probe_g, probe_eta, _res = full_model.fields(probe_q, probe_v, probe_u)
    probe = {"q": list(probe_q.shape), "v": list(probe_v.shape), "u": list(probe_u.shape),
             "drift": list(probe_f.shape), "diffusion": list(probe_g.shape),
             "eta": list(probe_eta.shape), "numpy_torch_max_abs": parity}

    data = cfg["data"]
    b, c, d = float(data["b"]), float(data["c"]), float(data["d"])
    sigma = float(data["sigma"])
    dt = float(data["dt"])

    def pack_split(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        q, v, u, features = query_rows(split["states"], split["actions"])
        drift_t, g_t, _eta = teacher_fields(full_weights, q, v, u)
        drift_a, g_a, _ = teacher_fields(aleatoric_weights, q, v, u)
        drift_true = true_drift(q, v, u, b, c, d)
        g_true = true_diffusion(q, sigma)
        mask = finite_mask(features, drift_t, g_t.reshape(-1, 1), drift_true, g_true.reshape(-1, 1))
        return {"features": features[mask], "drift_teacher": drift_t[mask],
                "g_teacher": g_t[mask].reshape(-1, 1),
                "drift_aleatoric": drift_a[mask], "g_aleatoric": g_a[mask].reshape(-1, 1),
                "drift_truth": drift_true[mask], "g_truth": g_true[mask].reshape(-1, 1),
                "eta": _eta[mask], "dropped": int((~mask).sum())}

    print("distilling", flush=True)
    train_q = pack_split(dataset["train"])
    val_q = pack_split(dataset["validation"])
    test_q = pack_split(dataset["test"])
    distill = cfg["distillation"]
    drift_field, drift_meta = select_field(
        train_q["features"], train_q["drift_teacher"], val_q["features"], val_q["drift_teacher"],
        distill["degrees"], distill["thresholds"], float(distill["ridge"]))
    g_field, g_meta = select_field(
        train_q["features"], train_q["g_teacher"], val_q["features"], val_q["g_teacher"],
        distill["degrees"], distill["thresholds"], float(distill["ridge"]))
    # Test is read here, once, after selection.
    drift_hat = drift_field.predict(test_q["features"])
    g_hat = g_field.predict(test_q["features"])
    ceiling = fit_polynomial_field(train_q["features"], train_q["drift_teacher"], 3, 0.0,
                                   float(distill["ridge"]))
    ceiling_g = fit_polynomial_field(train_q["features"], train_q["g_teacher"], 3, 0.0,
                                     float(distill["ridge"]))
    oracle = fit_polynomial_field(train_q["features"], train_q["drift_truth"], 3, 0.0,
                                  float(distill["ridge"]))
    oracle_g = fit_polynomial_field(train_q["features"], train_q["g_truth"], 3, 0.0,
                                    float(distill["ridge"]))
    names = ("\\dot q", "\\dot v")
    drift_tex = tex_equations(expand_to_original(drift_field), names, ("q", "v", "u"))
    g_tex = tex_equations(expand_to_original(g_field), ("g_v",), ("q", "v", "u"))
    symbolic_tex = ("\\begin{aligned}\n"
                    + drift_tex.split("\n", 1)[1].rsplit("\n", 1)[0] + "\n"
                    + g_tex.split("\n", 1)[1].rsplit("\n", 1)[0] + "\n"
                    + "g_q &= 0\n"
                    + "\\end{aligned}")

    grid = percentile_grid(train_q["features"])
    grid_drift_t, grid_g_t, _ = teacher_fields(full_weights, grid[:, 0], grid[:, 1], grid[:, 2])
    grid_drift_true = true_drift(grid[:, 0], grid[:, 1], grid[:, 2], b, c, d)
    grid_g_true = true_diffusion(grid[:, 0], sigma).reshape(-1, 1)
    grid_drift_hat = drift_field.predict(grid)
    grid_g_hat = g_field.predict(grid)

    ood = ood_position_shift(train_q["features"])
    _d_id, g_id, eta_id = teacher_fields(full_weights, test_q["features"][:, 0],
                                         test_q["features"][:, 1], test_q["features"][:, 2])
    _d_ood, g_ood, eta_ood = teacher_fields(full_weights, ood[:, 0], ood[:, 1], ood[:, 2])
    # h is the gap between the full diffusion and the same sigma network with h removed.
    sigma_only = dict(full_weights)
    sigma_only["epistemic"] = False
    h_id = g_id - teacher_fields(sigma_only, test_q["features"][:, 0],
                                 test_q["features"][:, 1], test_q["features"][:, 2])[1]
    h_ood = g_ood - teacher_fields(sigma_only, ood[:, 0], ood[:, 1], ood[:, 2])[1]
    _da, g_al_id, _ = teacher_fields(aleatoric_weights, test_q["features"][:, 0],
                                     test_q["features"][:, 1], test_q["features"][:, 2])
    _db, g_al_ood, _ = teacher_fields(aleatoric_weights, ood[:, 0], ood[:, 1], ood[:, 2])

    drift_inc = float(np.mean(np.abs(test_q["drift_teacher"][:, 1]) * dt))
    diff_inc = float(np.mean(np.abs(test_q["g_teacher"].reshape(-1)) * math.sqrt(dt)))
    ratio = diff_inc / max(drift_inc, 1e-8)

    print("simulating", flush=True)
    radius = float(cfg["simulation"]["failure_radius"])
    test_states = dataset["test"]["states"]
    test_actions = dataset["test"]["actions"]
    x0 = test_states[:, 0, :]
    steps = n_steps(cfg)
    rng_disc = np.random.default_rng(int(cfg["simulation"]["discretization_seed"]))
    increments = rng_disc.standard_normal((len(x0), steps))
    truth_d, truth_g = truth_callables(cfg)
    teacher_d, teacher_g = numpy_callables(full_weights)
    symbolic_d, symbolic_g = polynomial_callables(drift_field, g_field)
    sim_truth = simulate_controlled(truth_d, truth_g, x0, test_actions, dt, increments, radius)
    sim_teacher = simulate_controlled(teacher_d, teacher_g, x0, test_actions, dt, increments, radius)
    sim_symbolic = simulate_controlled(symbolic_d, symbolic_g, x0, test_actions, dt, increments, radius)
    factor = int(cfg["simulation"]["fine_factor"])
    fine = rng_disc.standard_normal((len(x0), steps, factor))
    disc = discretization_probe(teacher_d, teacher_g, x0, test_actions, dt, fine, radius)

    law_n = int(cfg["split"]["law"])
    noise_std = float(data["behavior"]["noise_std"])

    def independent_ensemble(drift_fn: DriftFn, diffusion_fn: DiffFn, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        x0_draw = sample_x0(rng, law_n, cfg)
        noise = rng.normal(0.0, noise_std, size=(law_n, steps))
        inc = rng.standard_normal((law_n, steps))
        return closed_loop_rollout(drift_fn, diffusion_fn, x0_draw, noise, inc, dt, cfg, radius)

    law_seed = int(cfg["simulation"]["law_seed"])
    law_truth = independent_ensemble(truth_d, truth_g, law_seed)
    law_teacher = independent_ensemble(teacher_d, teacher_g, law_seed + 1)
    law_symbolic = independent_ensemble(symbolic_d, symbolic_g, law_seed + 2)
    law_floor = independent_ensemble(truth_d, truth_g, int(cfg["simulation"]["floor_seed"]))

    loss_parts = loss_snapshot(full_model, dataset["train"]["states"],
                               dataset["train"]["actions"], cfg)
    vif = vif_max(train_q["features"], 3)
    g_hat_flat = g_hat.reshape(-1)
    metrics = {
        "drift_v_sym_teacher": component_nrmse(test_q["drift_teacher"], drift_hat, 1),
        "drift_v_teacher_truth": component_nrmse(test_q["drift_truth"], test_q["drift_teacher"], 1),
        "drift_v_sym_truth": component_nrmse(test_q["drift_truth"], drift_hat, 1),
        "drift_q_sym_teacher": component_nrmse(test_q["drift_teacher"], drift_hat, 0),
        "drift_q_teacher_truth": component_nrmse(test_q["drift_truth"], test_q["drift_teacher"], 0),
        "drift_q_sym_truth": component_nrmse(test_q["drift_truth"], drift_hat, 0),
        "g_sym_teacher": nrmse(test_q["g_teacher"], g_hat),
        "g_teacher_truth": nrmse(test_q["g_truth"], test_q["g_teacher"]),
        "g_sym_truth": nrmse(test_q["g_truth"], g_hat),
        "sigma_sym_teacher": nrmse(test_q["g_teacher"] ** 2, g_hat ** 2),
        "fraction_g_nonpositive": float(np.mean(g_hat_flat <= 0.0)),
        "vif": vif,
        "fail_truth": sim_truth["failure_rate"],
        "fail_teacher": sim_teacher["failure_rate"],
        "fail_symbolic": sim_symbolic["failure_rate"],
        "mean_sym_teacher": mean_rmse(mean_path(sim_symbolic["states"]), mean_path(sim_teacher["states"])),
        "mean_teacher_truth": mean_rmse(mean_path(sim_teacher["states"]), mean_path(sim_truth["states"])),
        "cov_sym_teacher": rel_frob(*[terminal_cov(s["states"])[0]
                                      for s in (sim_symbolic, sim_teacher)]),
        "cov_teacher_truth": rel_frob(*[terminal_cov(s["states"])[0]
                                        for s in (sim_teacher, sim_truth)]),
        "disc_rmse": disc["terminal_rmse"],
        "disc_retained": disc["retained_fraction"],
        "law_fail_truth": law_truth["failure_rate"],
        "law_fail_teacher": law_teacher["failure_rate"],
        "law_fail_symbolic": law_symbolic["failure_rate"],
        "law_mean_teacher": mean_rmse(mean_path(law_teacher["states"]), mean_path(law_truth["states"])),
        "law_mean_symbolic": mean_rmse(mean_path(law_symbolic["states"]), mean_path(law_truth["states"])),
        "law_cov_teacher": rel_frob(terminal_cov(law_teacher["states"])[0],
                                    terminal_cov(law_truth["states"])[0]),
        "law_cov_symbolic": rel_frob(terminal_cov(law_symbolic["states"])[0],
                                     terminal_cov(law_truth["states"])[0]),
        "floor_mean": mean_rmse(mean_path(law_floor["states"]), mean_path(law_truth["states"])),
        "floor_cov": rel_frob(terminal_cov(law_floor["states"])[0],
                              terminal_cov(law_truth["states"])[0]),
        "g_full_truth": nrmse(test_q["g_truth"], test_q["g_teacher"]),
        "g_aleatoric_truth": nrmse(test_q["g_truth"], test_q["g_aleatoric"]),
        "h_ood_minus_id": float(np.mean(h_ood) - np.mean(h_id)),
        "g_lift_full": float(np.mean(g_ood) - np.mean(g_id)),
        "g_lift_aleatoric": float(np.mean(g_al_ood) - np.mean(g_al_id)),
    }
    # rel_frob on the paired covs above used a starred list that may include None.
    # Recompute explicitly so a missing covariance stays None rather than raising.
    metrics["cov_sym_teacher"] = rel_frob(terminal_cov(sim_symbolic["states"])[0],
                                          terminal_cov(sim_teacher["states"])[0])
    metrics["cov_teacher_truth"] = rel_frob(terminal_cov(sim_teacher["states"])[0],
                                            terminal_cov(sim_truth["states"])[0])
    verdicts = decide(cfg, metrics)
    oracle_drift = component_nrmse(test_q["drift_truth"], oracle.predict(test_q["features"]), 1)
    oracle_g = nrmse(test_q["g_truth"], oracle_g.predict(test_q["features"]))
    attribute = (oracle_drift <= float(distill["oracle_exact_nrmse_max"])
                 and oracle_g <= float(distill["oracle_exact_nrmse_max"]))

    def class_counts(sim: dict[str, Any]) -> dict[str, int]:
        counts = {"non_finite": 0, "outside_radius": 0}
        for name in sim["failure_class"]:
            if name is not None:
                counts[name] += 1
        return counts

    results = {
        "day": 14,
        "paper": cfg["paper"]["title"],
        "venue": cfg["paper"]["journal"],
        "seed": cfg["seed"],
        "git_head": git_head(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "elapsed_sec": time.time() - started,
        "probe": probe,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "train_log": full_log,
        "aleatoric_log": aleatoric_log,
        "loss_snapshot": loss_parts,
        "drift_selection": drift_meta,
        "diffusion_selection": g_meta,
        "test": {
            "drift_velocity_nrmse": {
                "symbolic_vs_teacher": metrics["drift_v_sym_teacher"],
                "teacher_vs_truth": metrics["drift_v_teacher_truth"],
                "symbolic_vs_truth": metrics["drift_v_sym_truth"],
                "residual_rms_symbolic_vs_teacher": rms(test_q["drift_teacher"][:, 1], drift_hat[:, 1]),
                "residual_rms_teacher_vs_truth": rms(test_q["drift_truth"][:, 1],
                                                     test_q["drift_teacher"][:, 1]),
                "residual_rms_symbolic_vs_truth": rms(test_q["drift_truth"][:, 1], drift_hat[:, 1]),
                "teacher_rms": float(np.sqrt(np.mean(test_q["drift_teacher"][:, 1] ** 2))),
                "truth_rms": float(np.sqrt(np.mean(test_q["drift_truth"][:, 1] ** 2))),
            },
            "drift_position_nrmse": {
                "symbolic_vs_teacher": metrics["drift_q_sym_teacher"],
                "teacher_vs_truth": metrics["drift_q_teacher_truth"],
                "symbolic_vs_truth": metrics["drift_q_sym_truth"],
            },
            "diffusion_nrmse": {
                "symbolic_vs_teacher": metrics["g_sym_teacher"],
                "teacher_vs_truth": metrics["g_teacher_truth"],
                "symbolic_vs_truth": metrics["g_sym_truth"],
                "sigma_symbolic_vs_teacher": metrics["sigma_sym_teacher"],
                "residual_rms_symbolic_vs_teacher": rms(test_q["g_teacher"], g_hat),
                "residual_rms_teacher_vs_truth": rms(test_q["g_truth"], test_q["g_teacher"]),
                "residual_rms_symbolic_vs_truth": rms(test_q["g_truth"], g_hat),
                "teacher_rms": float(np.sqrt(np.mean(test_q["g_teacher"] ** 2))),
                "truth_rms": float(np.sqrt(np.mean(test_q["g_truth"] ** 2))),
                "fraction_symbolic_nonpositive": metrics["fraction_g_nonpositive"],
            },
            "one_step_scale": {
                "mean_abs_drift_increment": drift_inc,
                "mean_abs_diffusion_increment": diff_inc,
                "diffusion_over_drift": ratio,
            },
            "dropped_rows": {"train": train_q["dropped"], "validation": val_q["dropped"],
                             "test": test_q["dropped"]},
        },
        "oracle": {
            "velocity_drift_test_nrmse": oracle_drift,
            "diffusion_test_nrmse": oracle_g,
            "attribute_truth_failure_to_teacher": attribute,
            "ceiling_velocity_drift_test_nrmse": component_nrmse(
                test_q["drift_teacher"], ceiling.predict(test_q["features"]), 1),
            "ceiling_diffusion_test_nrmse": nrmse(test_q["g_teacher"],
                                                  ceiling_g.predict(test_q["features"])),
        },
        "grid": {
            "drift_velocity_rms_symbolic_vs_teacher": rms(grid_drift_t[:, 1], grid_drift_hat[:, 1]),
            "drift_velocity_rms_teacher_vs_truth": rms(grid_drift_true[:, 1], grid_drift_t[:, 1]),
            "drift_velocity_rms_symbolic_vs_truth": rms(grid_drift_true[:, 1], grid_drift_hat[:, 1]),
            "drift_velocity_nrmse_symbolic_vs_teacher": component_nrmse(grid_drift_t, grid_drift_hat, 1),
            "diffusion_rms_symbolic_vs_teacher": rms(grid_g_t, grid_g_hat),
            "diffusion_rms_teacher_vs_truth": rms(grid_g_true, grid_g_t.reshape(-1, 1)),
            "diffusion_rms_symbolic_vs_truth": rms(grid_g_true, grid_g_hat),
        },
        "uncertainty": {
            "mean_h_id": float(np.mean(h_id)),
            "mean_h_ood": float(np.mean(h_ood)),
            "mean_eta_id": float(np.mean(eta_id)),
            "mean_eta_ood": float(np.mean(eta_ood)),
            "mean_g_id": float(np.mean(g_id)),
            "mean_g_ood": float(np.mean(g_ood)),
            "mean_g_aleatoric_id": float(np.mean(g_al_id)),
            "mean_g_aleatoric_ood": float(np.mean(g_al_ood)),
            "h_ood_minus_id": metrics["h_ood_minus_id"],
            "g_lift_full": metrics["g_lift_full"],
            "g_lift_aleatoric": metrics["g_lift_aleatoric"],
        },
        "vif_degree3": vif,
        "simulation": {
            "failure_rate": {"truth": metrics["fail_truth"], "teacher": metrics["fail_teacher"],
                             "symbolic": metrics["fail_symbolic"]},
            "failure_class": {"truth": class_counts(sim_truth),
                              "teacher": class_counts(sim_teacher),
                              "symbolic": class_counts(sim_symbolic)},
            "paired_mean_rmse": {"symbolic_vs_teacher": metrics["mean_sym_teacher"],
                                 "teacher_vs_truth": metrics["mean_teacher_truth"]},
            "terminal_cov_rel_frob": {"symbolic_vs_teacher": metrics["cov_sym_teacher"],
                                      "teacher_vs_truth": metrics["cov_teacher_truth"]},
            "discretization": disc,
        },
        "law": {
            "failure_rate": {"truth": metrics["law_fail_truth"],
                             "teacher": metrics["law_fail_teacher"],
                             "symbolic": metrics["law_fail_symbolic"]},
            "mean_rmse": {"teacher_vs_observed": metrics["law_mean_teacher"],
                          "symbolic_vs_observed": metrics["law_mean_symbolic"],
                          "floor": metrics["floor_mean"]},
            "terminal_cov_rel_frob": {"teacher_vs_observed": metrics["law_cov_teacher"],
                                      "symbolic_vs_observed": metrics["law_cov_symbolic"],
                                      "floor": metrics["floor_cov"]},
        },
        "verdicts": verdicts,
        "symbolic_tex": symbolic_tex,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(to_jsonable(results), indent=2))
    (OUT_DIR / "teacher_nuno_weights.json").write_text(json.dumps(full_weights))
    (OUT_DIR / "teacher_aleatoric_weights.json").write_text(json.dumps(aleatoric_weights))
    drift_field.save(OUT_DIR / "symbolic_f.json")
    g_field.save(OUT_DIR / "symbolic_g.json")
    (OUT_DIR / "symbolic_sde.tex").write_text(symbolic_tex + "\n")
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": [{str(k): v for k, v in poly.items()} for poly in expand_to_original(drift_field)],
        "diffusion": [{str(k): v for k, v in poly.items()}
                      for poly in expand_to_original(g_field)],
        "position_diffusion": "0 (structural, not fit)",
    }, indent=2))
    print(json.dumps({"verdicts": verdicts,
                      "drift_v": metrics["drift_v_sym_teacher"],
                      "g": metrics["g_sym_teacher"],
                      "elapsed": results["elapsed_sec"]}, indent=2), flush=True)
    return results


if __name__ == "__main__":
    main()
