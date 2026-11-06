"""Day 12: Zhang et al. (ICLR 2025) finite-dimensional matching -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day12.json``):

1. Data: a noisy van der Pol diffusion in observable coordinates (x, y).
   The closed form is the data law. It is not a gene-regulatory equation.
2. Teacher: neural drift and neural diffusion trained by the released
   FDDiscriminator loss (consecutive two-time joints, Gaussian RBF).
   A marginal-only network with the same budget is the H6 baseline.
3. Distillation: direct teacher queries, trajectory-disjoint splits,
   power library in (x, y, t), validation-only selection, test once.
4. Simulation keeps failed-trajectory mass and separates an excursion
   radius from the failure radius.

Run from the repository root:
``python -m src.daily_neural_sde.day12_zhang_fdm``

Outputs land in ``outputs/daily_neural_sde/day12/``.
"""
from __future__ import annotations

import json
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day12.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day12"

FieldFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-12 config."""
    return json.loads(path.read_text())


# ---------------------------------------------------------------- truth

def true_drift(x: np.ndarray, mu: float) -> np.ndarray:
    """Van der Pol drift. ``x`` has shape (n, 2) and the result has shape (n, 2)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x)
    out[:, 0] = x[:, 1]
    out[:, 1] = mu * (1.0 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
    return out


def true_diffusion(x: np.ndarray, g0: float, g2: float, gy: float) -> np.ndarray:
    """Diagonal diffusion. ``x`` has shape (n, 2) and the result has shape (n, 2)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.empty((len(x), 2), dtype=np.float64)
    out[:, 0] = g0 + g2 * x[:, 0] ** 2
    out[:, 1] = gy
    return out


def to_original(f_z: np.ndarray, g_z: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized-coordinate fields back to x. Shapes (n, 2).

    The map x = mean + std * z is linear, so Ito's formula adds no
    second-order correction: dx = std * dz elementwise.
    """
    std = np.asarray(std, dtype=np.float64)
    return f_z * std, g_z * std


# ---------------------------------------------------------------- simulation

def _latch(x: np.ndarray, failed: np.ndarray, failure_class: list[str | None],
           excursion: np.ndarray, failure_radius: float,
           excursion_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mark new failures and excursions. Failed rows of ``x`` become NaN."""
    finite = np.isfinite(x).all(axis=1)
    norm = np.linalg.norm(np.where(finite[:, None], x, 0.0), axis=1)
    excursion = excursion | (~failed & finite & (norm > excursion_radius))
    bad_nonfinite = ~finite
    bad_radius = finite & (norm > failure_radius)
    newly = (~failed) & (bad_nonfinite | bad_radius)
    for idx in np.flatnonzero(newly):
        failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
    failed = failed | newly
    x = np.array(x, dtype=np.float64, copy=True)
    x[failed] = np.nan
    return x, failed, excursion


def simulate_timed(drift_fn: Callable[[np.ndarray, float], np.ndarray],
                   diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
                   x0: np.ndarray, dt: float, increments: np.ndarray,
                   failure_radius: float, excursion_radius: float) -> dict[str, Any]:
    """Euler-Maruyama for a time-dependent diagonal SDE.

    ``drift_fn(x, t)`` and ``diffusion_fn(x, t)`` map ``x`` of shape
    (n_active, 2) and a scalar time to arrays of shape (n_active, 2).
    ``increments`` has shape (n, n_steps, 2) and contains standard normals.
    The update adds ``diffusion * sqrt(dt) * increment``.

    Failures latch and are classified ``non_finite`` or ``outside_radius``.
    Failed trajectories stay in the counts and are NaN from the failure
    step onward, including a failure already present at the initial state.
    An excursion is ``||X|| > excursion_radius`` on a finite row that has
    not yet failed.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x0 must have shape (n, 2)")
    n = len(x)
    if increments.ndim != 3 or increments.shape[0] != n or increments.shape[2] != 2:
        raise ValueError("increments must have shape (n, n_steps, 2)")
    failed = np.zeros(n, dtype=bool)
    excursion = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    x, failed, excursion = _latch(x, failed, failure_class, excursion,
                                  failure_radius, excursion_radius)
    states = [x.copy()]
    sqrt_dt = float(np.sqrt(dt))
    n_steps = increments.shape[1]
    for step in range(n_steps):
        active = ~failed
        if active.any():
            drift = np.asarray(drift_fn(x[active], step * dt), dtype=np.float64)
            diffusion = np.array(np.asarray(diffusion_fn(x[active], step * dt),
                                            dtype=np.float64), copy=True)
            if drift.shape != (int(active.sum()), 2):
                raise ValueError("drift_fn must return shape (n_active, 2)")
            if diffusion.shape != (int(active.sum()), 2):
                raise ValueError("diffusion_fn must return shape (n_active, 2)")
            x[active] = x[active] + drift * dt + diffusion * sqrt_dt * increments[active, step]
        x, failed, excursion = _latch(x, failed, failure_class, excursion,
                                      failure_radius, excursion_radius)
        states.append(x.copy())
    return {"states": np.stack(states, axis=1), "final": x, "failed": failed,
            "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion": excursion,
            "excursion_rate": float(np.mean(excursion))}


def failure_counts(sim: dict[str, Any]) -> dict[str, Any]:
    """Failure and excursion counts. Failed paths remain in ``n``."""
    classes = sim["failure_class"]
    return {"failure_rate": sim["failure_rate"],
            "excursion_rate": sim["excursion_rate"],
            "n": len(classes),
            "n_non_finite": int(sum(c == "non_finite" for c in classes)),
            "n_outside_radius": int(sum(c == "outside_radius" for c in classes))}


def coarse_increments(fine: np.ndarray, block: int) -> np.ndarray:
    """Standard normals for a block sum of Wiener increments.

    ``fine`` has shape (n, n_fine, d) of N(0, 1) draws at the fine step.
    The coarse draw is the sum over each block divided by sqrt(block),
    which is N(0, 1) at the coarse step.
    """
    if fine.shape[1] % block != 0:
        raise ValueError("fine step count is not divisible by block")
    n, n_fine, d = fine.shape
    grouped = fine.reshape(n, n_fine // block, block, d)
    return grouped.sum(axis=2) / np.sqrt(block)


# ---------------------------------------------------------------- data

def make_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Trajectory-disjoint truth paths. One x0 and one Wiener path per row."""
    data = cfg["data"]
    dt = float(data["dt"])
    n_steps = int(round(float(data["T"]) / dt))
    if abs(n_steps * dt - float(data["T"])) > 1e-12:
        raise ValueError("T is not an integer multiple of dt")
    mu, g0, g2, gy = (float(data["mu"]), float(data["g0"]),
                      float(data["g2"]), float(data["gy"]))
    radius = float(data["data_failure_radius"])
    rng = np.random.default_rng(int(cfg["split"]["seed"]))

    def drift(x: np.ndarray, t: float) -> np.ndarray:
        return true_drift(x, mu)

    def diffusion(x: np.ndarray, t: float) -> np.ndarray:
        return true_diffusion(x, g0, g2, gy)

    out: dict[str, Any] = {"dt": dt, "n_steps": n_steps,
                           "times": dt * np.arange(n_steps + 1)}
    for name in ("train", "validation", "test"):
        n = int(cfg["split"][name])
        x0 = rng.normal(size=(n, 2))
        increments = rng.standard_normal((n, n_steps, 2))
        sim = simulate_timed(drift, diffusion, x0, dt, increments, radius, radius)
        if sim["failure_rate"] != 0.0:
            raise RuntimeError(f"{name} truth paths failed at radius {radius}")
        out[name] = {"x0": x0, "increments": increments, "paths": sim["states"]}
    return out


def normalization_from_train(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate mean and std of the training paths. Shape (2,)."""
    flat = np.asarray(paths, dtype=np.float64).reshape(-1, 2)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


# ---------------------------------------------------------------- kernel losses

def rbf_matrix(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF Gram matrix. ``a`` is (n, d), ``b`` is (m, d)."""
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True)
    dist = aa + bb.T - 2.0 * a @ b.T
    return np.exp(-dist / (2.0 * sigma ** 2))


def pair_loss_numpy(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    """Off-diagonal E k(a,a) - 2 E k(a,b). Smaller when a matches b."""
    kxx = rbf_matrix(a, a, sigma)
    kxy = rbf_matrix(a, b, sigma)
    n = len(a)
    off = (float(kxx.sum()) - float(np.trace(kxx))) / (n * (n - 1))
    return off - 2.0 * float(kxy.mean())


def fdm_loss_numpy(gen: np.ndarray, data: np.ndarray, sigma: float) -> float:
    """Consecutive-joint FDDiscriminator loss. Paths have shape (n, L, d)."""
    left_g = gen[:, :-1, :]
    right_g = gen[:, 1:, :]
    left_d = data[:, :-1, :]
    right_d = data[:, 1:, :]
    losses = []
    for t in range(left_g.shape[1]):
        a = np.concatenate([left_g[:, t], right_g[:, t]], axis=1)
        b = np.concatenate([left_d[:, t], right_d[:, t]], axis=1)
        losses.append(pair_loss_numpy(a, b, sigma))
    return float(np.mean(losses))


def mmd2(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    """Unbiased-diagonal MMD^2 of the Gaussian RBF. Rows are samples."""
    kxx = rbf_matrix(a, a, sigma)
    kyy = rbf_matrix(b, b, sigma)
    kxy = rbf_matrix(a, b, sigma)
    np.fill_diagonal(kxx, np.nan)
    np.fill_diagonal(kyy, np.nan)
    return float(np.nanmean(kxx) + np.nanmean(kyy) - 2.0 * np.mean(kxy))


def _kernel_mean(x: torch.Tensor, y: torch.Tensor, sigma: float,
                 exclude_diag: bool) -> torch.Tensor:
    """RBF mean over the sample axis. ``x`` and ``y`` have shape (T, B, D)."""
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True)
    dist = xx + yy.transpose(1, 2) - 2.0 * torch.matmul(x, y.transpose(1, 2))
    kernel = torch.exp(-dist / (2.0 * sigma ** 2))
    if exclude_diag:
        batch = x.shape[1]
        eye = torch.eye(batch, dtype=torch.bool, device=x.device)
        kernel = kernel.masked_fill(eye.unsqueeze(0), 0.0)
        return kernel.sum(dim=(1, 2)) / (batch * (batch - 1))
    return kernel.mean(dim=(1, 2))


def fdm_loss_torch(gen: torch.Tensor, data: torch.Tensor, sigma: float) -> torch.Tensor:
    """FDDiscriminator loss. ``gen`` and ``data`` have shape (B, L, 2)."""
    x = torch.cat((gen[:, :-1, :], gen[:, 1:, :]), dim=-1).transpose(0, 1)
    y = torch.cat((data[:, :-1, :], data[:, 1:, :]), dim=-1).transpose(0, 1)
    return (_kernel_mean(x, x, sigma, True) - 2.0 * _kernel_mean(x, y, sigma, False)).mean()


def marginal_loss_torch(gen: torch.Tensor, data: torch.Tensor, sigma: float) -> torch.Tensor:
    """Single-time version of the same kernel loss, averaged over the grid."""
    x = gen.transpose(0, 1)
    y = data.transpose(0, 1)
    return (_kernel_mean(x, x, sigma, True) - 2.0 * _kernel_mean(x, y, sigma, False)).mean()


# ---------------------------------------------------------------- teacher net

class FieldNet(nn.Module):
    """MLP field. Input (z, t/T) has shape (n, 3). Output has shape (n, 2)."""

    def __init__(self, width: int, positive: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 2)
        self.positive = positive

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat((x, t_norm), dim=1)
        hidden = torch.tanh(self.fc1(hidden))
        hidden = torch.tanh(self.fc2(hidden))
        value = self.fc3(hidden)
        if self.positive:
            value = torch.nn.functional.softplus(value)
        return value


class NeuralSDE(nn.Module):
    """Separate drift and diffusion networks. Both see (z, t/T)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.drift = FieldNet(width, positive=False)
        self.diffusion = FieldNet(width, positive=True)

    def forward(self, x: torch.Tensor, t_norm: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.drift(x, t_norm), self.diffusion(x, t_norm)


def init_sde(model: NeuralSDE) -> None:
    """Small output weights. Diffusion bias sets softplus(-0.5) near the data scale."""
    for net in (model.drift, model.diffusion):
        for layer in (net.fc1, net.fc2):
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(net.fc3.weight, gain=0.1)
        nn.init.zeros_(net.fc3.bias)
    nn.init.constant_(model.diffusion.fc3.bias, -0.5)


def rollout(model: NeuralSDE, x0: torch.Tensor, dW: torch.Tensor,
            dt: float, horizon: float) -> torch.Tensor:
    """Euler-Maruyama in the normalized coordinate.

    ``x0`` has shape (B, 2). ``dW`` has shape (B, n_steps, 2) of standard
    normals. The returned path has shape (B, n_steps + 1, 2).
    """
    x = x0
    states = [x]
    n_steps = dW.shape[1]
    sqrt_dt = dt ** 0.5
    for k in range(n_steps):
        t_norm = torch.full((x.shape[0], 1), (k * dt) / horizon,
                            dtype=x.dtype, device=x.device)
        drift, diffusion = model(x, t_norm)
        x = x + drift * dt + diffusion * sqrt_dt * dW[:, k]
        states.append(x)
    return torch.stack(states, dim=1)


def train_model(kind: str, paths_z: np.ndarray, cfg: dict[str, Any]
                ) -> tuple[NeuralSDE, dict[str, Any]]:
    """Train one neural SDE. ``kind`` is ``fdm`` or ``marginal``.

    The torch and numpy generators are reseeded from the config seed so the
    two kinds share their initialization and their Brownian draws.
    """
    torch.manual_seed(int(cfg["seed"]))
    model = NeuralSDE(int(cfg["teacher"]["width"])).double()
    init_sde(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["teacher"]["lr"]),
                                 weight_decay=float(cfg["teacher"]["weight_decay"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    batch = int(cfg["teacher"]["batch_size"])
    dt = float(cfg["data"]["dt"])
    horizon = float(cfg["data"]["T"])
    sigma = float(cfg["teacher"]["rbf_sigma"])
    n_opt = int(cfg["teacher"]["n_opt_steps"])
    clip = float(cfg["teacher"]["grad_clip"])
    paths = torch.tensor(paths_z, dtype=torch.float64)
    n_paths = paths.shape[0]
    trace: list[dict[str, float]] = []
    skipped = 0
    for step in range(n_opt):
        index = rng.integers(0, n_paths, size=batch)
        data = paths[index]
        dW = torch.randn(batch, data.shape[1] - 1, 2, dtype=torch.float64)
        optimizer.zero_grad()
        generated = rollout(model, data[:, 0], dW, dt, horizon)
        if kind == "fdm":
            loss = fdm_loss_torch(generated, data, sigma)
        elif kind == "marginal":
            loss = marginal_loss_torch(generated, data, sigma)
        else:
            raise ValueError(kind)
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if step % 200 == 0 or step == n_opt - 1:
            value = float(loss.detach())
            trace.append({"step": step, "loss": value})
            print(f"{kind} step {step} loss {value:.6f}", flush=True)
    model.eval()
    return model, {"loss_trace": trace, "skipped_steps": skipped, "kind": kind}


def export_layers(net: FieldNet) -> list[dict[str, list[float]]]:
    """Linear layers as JSON lists. Weight shape is (out, in), matching PyTorch."""
    layers = []
    for layer in (net.fc1, net.fc2, net.fc3):
        layers.append({"weight": layer.weight.detach().cpu().tolist(),
                       "bias": layer.bias.detach().cpu().tolist()})
    return layers


def stable_softplus(value: np.ndarray) -> np.ndarray:
    """log(1 + exp(value)), stable for large |value|. Matches torch.softplus."""
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def mlp_numpy(features: np.ndarray, layers: list[dict[str, Any]], positive: bool) -> np.ndarray:
    """Evaluate an exported MLP. ``features`` has shape (n, 3)."""
    hidden = np.asarray(features, dtype=np.float64)
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        weight = np.asarray(layer["weight"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        hidden = hidden @ weight.T + bias
        if i < last:
            hidden = np.tanh(hidden)
        elif positive:
            hidden = stable_softplus(hidden)
    return hidden


def teacher_field(weights: dict[str, Any], x: np.ndarray, t: np.ndarray,
                  which: str) -> np.ndarray:
    """Original-coordinate field. ``x`` is (n, 2), ``t`` is (n,), result is (n, 2)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size == 1:
        t = np.full(len(x), t.item())
    if t.shape != (len(x),):
        raise ValueError("t must be a scalar or shape (n,)")
    mean = np.asarray(weights["mean"], dtype=np.float64)
    std = np.asarray(weights["std"], dtype=np.float64)
    z = (x - mean) / std
    features = np.concatenate([z, (t / float(weights["T"])).reshape(-1, 1)], axis=1)
    positive = which == "diffusion"
    value_z = mlp_numpy(features, weights[which], positive)
    return value_z * std


def parity_error(model: NeuralSDE, weights: dict[str, Any], seed: int) -> float:
    """Max absolute gap between the torch net and the exported numpy net."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(32, 2))
    t_norm = rng.random(32)
    with torch.no_grad():
        drift_t, diffusion_t = model(torch.tensor(z), torch.tensor(t_norm).reshape(-1, 1))
    features = np.concatenate([z, t_norm.reshape(-1, 1)], axis=1)
    drift_n = mlp_numpy(features, weights["drift"], False)
    diffusion_n = mlp_numpy(features, weights["diffusion"], True)
    gap = max(float(np.max(np.abs(drift_t.numpy() - drift_n))),
              float(np.max(np.abs(diffusion_t.numpy() - diffusion_n))))
    return gap


# ---------------------------------------------------------------- SINDy

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
    best_val = np.inf
    best_meta: dict[str, Any] = {}
    for degree in degrees:
        for threshold in thresholds:
            field = fit_polynomial_field(x_train, y_train, degree, threshold, ridge)
            val = nrmse(y_val, field.predict(x_val))
            if val < best_val:
                best_val = val
                best = field
                best_meta = {"degree": degree, "threshold": threshold,
                             "validation_nrmse": val,
                             "active_terms": int(np.count_nonzero(field.coefficients)),
                             "library_size": int(field.metadata["library_size"])}
    if best is None:
        raise RuntimeError("no candidate")
    return best, best_meta


def query_bundle(paths: np.ndarray, times: np.ndarray,
                 fields: dict[str, FieldFn]) -> dict[str, Any]:
    """Stack (x, y, t) and several direct field queries on one shared mask.

    A row is kept only when every field is finite, so the targets stay aligned.
    """
    n_traj, _, _ = paths.shape
    xy = paths.reshape(-1, 2)
    t = np.tile(np.asarray(times, dtype=np.float64), n_traj)
    features = np.concatenate([xy, t.reshape(-1, 1)], axis=1)
    targets = {name: fn(xy, t) for name, fn in fields.items()}
    ok = np.isfinite(features).all(axis=1)
    for target in targets.values():
        ok = ok & np.isfinite(target).all(axis=1)
    out: dict[str, Any] = {"features": features[ok], "dropped": int((~ok).sum())}
    out.update({name: target[ok] for name, target in targets.items()})
    return out


def rms(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean square of the residual."""
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def field_block(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Pooled NRMSE, residual RMS, truth RMS, and per-component NRMSE."""
    return {"nrmse": nrmse(truth, prediction),
            "residual_rms": rms(truth, prediction),
            "truth_rms": float(np.sqrt(np.mean(truth ** 2))),
            "nrmse_components": [nrmse(truth[:, j], prediction[:, j])
                                 for j in range(truth.shape[1])]}


def mean_path(states: np.ndarray) -> np.ndarray:
    """Time series of the cross-trajectory mean. Shape (L, 2). NaNs stay out."""
    return np.nanmean(states, axis=0)


def mean_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square of the Euclidean gap between two mean paths."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def terminal_cov(states: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Covariance of finite terminal states, and the number retained."""
    final = states[:, -1, :]
    ok = np.isfinite(final).all(axis=1)
    if int(ok.sum()) < 3:
        return None, int(ok.sum())
    return np.cov(final[ok], rowvar=False), int(ok.sum())


def rel_frob(left: np.ndarray, right: np.ndarray) -> float:
    """||left - right||_F / ||right||_F."""
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-8))


def fresh_paths(drift_fn: Callable[[np.ndarray, float], np.ndarray],
                diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
                n: int, dt: float, n_steps: int, seed: int,
                failure_radius: float, excursion_radius: float) -> dict[str, Any]:
    """Unconditional paths from N(0, I) initial conditions."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, 2))
    increments = rng.standard_normal((n, n_steps, 2))
    return simulate_timed(drift_fn, diffusion_fn, x0, dt, increments,
                          failure_radius, excursion_radius)


def two_time_mmd(paths_a: np.ndarray, paths_b: np.ndarray, pairs: list[list[int]],
                 mean: np.ndarray, std: np.ndarray, sigma: float) -> float:
    """Average MMD^2 of normalized two-time joints over the listed index pairs."""
    values = []
    for i, j in pairs:
        a = np.concatenate([(paths_a[:, i] - mean) / std, (paths_a[:, j] - mean) / std], axis=1)
        b = np.concatenate([(paths_b[:, i] - mean) / std, (paths_b[:, j] - mean) / std], axis=1)
        ok_a = np.isfinite(a).all(axis=1)
        ok_b = np.isfinite(b).all(axis=1)
        if ok_a.sum() < 3 or ok_b.sum() < 3:
            return float("nan")
        values.append(mmd2(a[ok_a], b[ok_b], sigma))
    return float(np.mean(values))


def diagnostic_grid(train_xy: np.ndarray, times: list[float],
                    teacher_f: FieldFn, teacher_g: FieldFn,
                    sym_f: PolynomialField, sym_g: PolynomialField,
                    truth_f: FieldFn, truth_g: FieldFn) -> list[dict[str, float]]:
    """Off-support grid. Not a selection input."""
    lo = np.quantile(train_xy, 0.05, axis=0)
    hi = np.quantile(train_xy, 0.95, axis=0)
    pad = 0.5 * (hi - lo)
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 9)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 9)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    rows = []
    for t in times:
        tt = np.full(len(xy), t)
        features = np.concatenate([xy, tt.reshape(-1, 1)], axis=1)
        f_teacher = teacher_f(xy, tt)
        g_teacher = teacher_g(xy, tt)
        f_sym = sym_f.predict(features)
        g_sym = sym_g.predict(features)
        f_truth = truth_f(xy, tt)
        g_truth = truth_g(xy, tt)
        rows.append({
            "t": float(t),
            "drift_symbolic_vs_teacher_nrmse": nrmse(f_teacher, f_sym),
            "drift_symbolic_vs_teacher_rms": rms(f_teacher, f_sym),
            "drift_teacher_vs_truth_nrmse": nrmse(f_truth, f_teacher),
            "drift_teacher_vs_truth_rms": rms(f_truth, f_teacher),
            "drift_symbolic_vs_truth_nrmse": nrmse(f_truth, f_sym),
            "drift_symbolic_vs_truth_rms": rms(f_truth, f_sym),
            "diffusion_symbolic_vs_teacher_nrmse": nrmse(g_teacher, g_sym),
            "diffusion_symbolic_vs_teacher_rms": rms(g_teacher, g_sym),
            "diffusion_teacher_vs_truth_nrmse": nrmse(g_truth, g_teacher),
            "diffusion_teacher_vs_truth_rms": rms(g_truth, g_teacher),
            "diffusion_symbolic_vs_truth_nrmse": nrmse(g_truth, g_sym),
            "diffusion_symbolic_vs_truth_rms": rms(g_truth, g_sym),
        })
    return rows


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


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def git_head() -> str:
    """Current commit, or an empty string if git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


# ---------------------------------------------------------------- main

def main() -> dict[str, Any]:
    """Train, distill, validate once, and write the day-12 outputs."""
    started = time.time()
    cfg = load_config()
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    data_cfg = cfg["data"]
    mu = float(data_cfg["mu"])
    g0, g2, gy = float(data_cfg["g0"]), float(data_cfg["g2"]), float(data_cfg["gy"])
    dataset = make_dataset(cfg)
    mean, std = normalization_from_train(dataset["train"]["paths"])
    paths_z = (dataset["train"]["paths"] - mean) / std

    print("training FDM teacher", flush=True)
    fdm_model, fdm_log = train_model("fdm", paths_z, cfg)
    print("training marginal baseline", flush=True)
    marginal_model, marginal_log = train_model("marginal", paths_z, cfg)

    def pack(model: NeuralSDE) -> dict[str, Any]:
        return {"drift": export_layers(model.drift),
                "diffusion": export_layers(model.diffusion),
                "mean": mean.tolist(), "std": std.tolist(),
                "T": float(data_cfg["T"]), "width": int(cfg["teacher"]["width"])}

    fdm_weights = pack(fdm_model)
    marginal_weights = pack(marginal_model)
    parity = parity_error(fdm_model, fdm_weights, int(cfg["seed"]) + 7)
    if parity > 1e-6:
        raise RuntimeError(f"numpy/torch field mismatch {parity}")

    # Shape probe. The network is queried in the normalized coordinate.
    probe_z = torch.zeros(8, 2, dtype=torch.float64)
    probe_t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        probe_f, probe_g = fdm_model(probe_z, probe_t)
    probe = {"z": list(probe_z.shape), "t_norm": list(probe_t.shape),
             "drift": list(probe_f.shape), "diffusion": list(probe_g.shape),
             "numpy_torch_max_abs": parity}

    def fdm_f(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_field(fdm_weights, xy, t, "drift")

    def fdm_g(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_field(fdm_weights, xy, t, "diffusion")

    def marg_f(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_field(marginal_weights, xy, t, "drift")

    def marg_g(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_field(marginal_weights, xy, t, "diffusion")

    def truth_f(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return true_drift(xy, mu)

    def truth_g(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return true_diffusion(xy, g0, g2, gy)

    times = dataset["times"]
    queries = {name: query_bundle(dataset[name]["paths"], times, {
        "drift": fdm_f, "diffusion": fdm_g, "drift_truth": truth_f,
        "diffusion_truth": truth_g, "drift_marginal": marg_f,
        "diffusion_marginal": marg_g}) for name in ("train", "validation", "test")}

    dist = cfg["distillation"]
    degrees = [int(v) for v in dist["degrees"]]
    thresholds = [float(v) for v in dist["thresholds"]]
    ridge = float(dist["ridge"])
    drift_field, drift_meta = select_field(
        queries["train"]["features"], queries["train"]["drift"],
        queries["validation"]["features"], queries["validation"]["drift"],
        degrees, thresholds, ridge)
    g_field, g_meta = select_field(
        queries["train"]["features"], queries["train"]["diffusion"],
        queries["validation"]["features"], queries["validation"]["diffusion"],
        degrees, thresholds, ridge)

    def evaluate_split(name: str) -> dict[str, Any]:
        feat = queries[name]["features"]
        drift_hat = drift_field.predict(feat)
        g_hat = g_field.predict(feat)
        return {
            "drift_symbolic_vs_teacher": field_block(queries[name]["drift"], drift_hat),
            "drift_teacher_vs_truth": field_block(queries[name]["drift_truth"],
                                                  queries[name]["drift"]),
            "drift_symbolic_vs_truth": field_block(queries[name]["drift_truth"], drift_hat),
            "diffusion_symbolic_vs_teacher": field_block(queries[name]["diffusion"], g_hat),
            "diffusion_teacher_vs_truth": field_block(queries[name]["diffusion_truth"],
                                                      queries[name]["diffusion"]),
            "diffusion_symbolic_vs_truth": field_block(queries[name]["diffusion_truth"], g_hat),
            "sigma_symbolic_vs_teacher": field_block(queries[name]["diffusion"] ** 2, g_hat ** 2),
            "marginal_drift_vs_truth": field_block(queries[name]["drift_truth"],
                                                   queries[name]["drift_marginal"]),
            "marginal_diffusion_vs_truth": field_block(queries[name]["diffusion_truth"],
                                                       queries[name]["diffusion_marginal"]),
            "fraction_symbolic_g_nonpositive": float(np.mean(np.any(g_hat <= 0.0, axis=1))),
        }

    # Test is read here, once, after selection. Validation numbers above
    # chose the polynomials. The test block is not used to choose them.
    heldout = {"validation": evaluate_split("validation"), "test": evaluate_split("test")}

    ceiling_f = fit_polynomial_field(queries["train"]["features"], queries["train"]["drift"],
                                     3, 0.0, ridge)
    ceiling_g = fit_polynomial_field(queries["train"]["features"], queries["train"]["diffusion"],
                                     3, 0.0, ridge)
    oracle_f = fit_polynomial_field(queries["train"]["features"], queries["train"]["drift_truth"],
                                    3, 0.0, ridge)
    oracle_g = fit_polynomial_field(queries["train"]["features"],
                                    queries["train"]["diffusion_truth"], 3, 0.0, ridge)
    feat_te = queries["test"]["features"]
    ceilings = {
        "drift_ols_vs_teacher": nrmse(queries["test"]["drift"], ceiling_f.predict(feat_te)),
        "diffusion_ols_vs_teacher": nrmse(queries["test"]["diffusion"], ceiling_g.predict(feat_te)),
        "drift_oracle_vs_truth": nrmse(queries["test"]["drift_truth"], oracle_f.predict(feat_te)),
        "diffusion_oracle_vs_truth": nrmse(queries["test"]["diffusion_truth"],
                                           oracle_g.predict(feat_te)),
    }

    vif = {"drift_library_degree_3": vif_max(queries["train"]["features"], 3),
           "note": "One library in (x, y, t). The same value gates both fields."}
    # Diffusion uses the same features, so the VIF is the same number.
    vif["diffusion_library_degree_3"] = vif["drift_library_degree_3"]

    dt = float(dataset["dt"])
    f_norm = np.linalg.norm(queries["test"]["drift"], axis=1)
    g_norm = np.linalg.norm(queries["test"]["diffusion"], axis=1)
    snr = {"mean_abs_f_dt": float(np.mean(f_norm) * dt),
           "mean_g_sqrt_dt": float(np.mean(g_norm) * np.sqrt(dt))}
    snr["ratio_of_means"] = snr["mean_abs_f_dt"] / max(snr["mean_g_sqrt_dt"], 1e-15)

    # Paired simulation on the test Wiener paths.
    radius = float(cfg["simulation"]["failure_radius"])
    excursion_radius = float(cfg["simulation"]["excursion_radius"])
    test = dataset["test"]

    def run_paired(drift_fn: Callable, diffusion_fn: Callable) -> dict[str, Any]:
        return simulate_timed(drift_fn, diffusion_fn, test["x0"], dt,
                              test["increments"], radius, excursion_radius)

    sim_truth = run_paired(truth_f, truth_g)
    sim_teacher = run_paired(fdm_f, fdm_g)
    sim_symbolic = run_paired(
        lambda xy, t: drift_field.predict(np.concatenate([xy, np.full((len(xy), 1), t)], axis=1)),
        lambda xy, t: g_field.predict(np.concatenate([xy, np.full((len(xy), 1), t)], axis=1)))
    truth_resim_gap = float(np.max(np.abs(sim_truth["states"] - test["paths"])))

    def paired_metrics(sim: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
        cov_s, n_s = terminal_cov(sim["states"])
        cov_r, n_r = terminal_cov(reference["states"])
        cov_gap = None if cov_s is None or cov_r is None else rel_frob(cov_s, cov_r)
        return {"counts": failure_counts(sim),
                "mean_rmse": mean_rmse(mean_path(sim["states"]), mean_path(reference["states"])),
                "terminal_cov_rel_frob": cov_gap,
                "terminal_finite_n": [n_s, n_r]}

    paired = {"truth_resimulation_max_abs": truth_resim_gap,
              "teacher_vs_truth": paired_metrics(sim_teacher, sim_truth),
              "symbolic_vs_teacher": paired_metrics(sim_symbolic, sim_teacher),
              "symbolic_vs_truth": paired_metrics(sim_symbolic, sim_truth),
              "truth_counts": failure_counts(sim_truth)}

    # Discretization gate. Fine Wiener increments sum to the coarse ones.
    fine_factor = int(cfg["simulation"]["fine_factor"])
    fine_dt = dt / fine_factor
    n_fine = dataset["n_steps"] * fine_factor
    disc_rng = np.random.default_rng(int(cfg["simulation"]["discretization_seed"]))
    fine_inc = disc_rng.standard_normal((len(test["x0"]), n_fine, 2))
    coarse_inc = coarse_increments(fine_inc, fine_factor)
    sim_fine = simulate_timed(fdm_f, fdm_g, test["x0"], fine_dt, fine_inc,
                              radius, excursion_radius)
    sim_coarse = simulate_timed(fdm_f, fdm_g, test["x0"], dt, coarse_inc,
                                radius, excursion_radius)
    both = np.isfinite(sim_fine["final"]).all(axis=1) & np.isfinite(sim_coarse["final"]).all(axis=1)
    if both.any():
        terminal_gap = sim_coarse["final"][both] - sim_fine["final"][both]
        disc_rmse = float(np.sqrt(np.mean(np.sum(terminal_gap ** 2, axis=1))))
    else:
        disc_rmse = float("nan")
    discretization = {"terminal_rmse": disc_rmse,
                      "retained_fraction": float(np.mean(both)),
                      "fine_failure_rate": sim_fine["failure_rate"],
                      "coarse_failure_rate": sim_coarse["failure_rate"]}

    # Unconditional law comparison against the observed test paths.
    law_seed = int(cfg["simulation"]["law_seed"])
    n_law = len(test["x0"])
    law_truth = fresh_paths(truth_f, truth_g, n_law, dt, dataset["n_steps"],
                            law_seed, radius, excursion_radius)
    law_teacher = fresh_paths(fdm_f, fdm_g, n_law, dt, dataset["n_steps"],
                              law_seed, radius, excursion_radius)
    law_symbolic = fresh_paths(
        lambda xy, t: drift_field.predict(np.concatenate([xy, np.full((len(xy), 1), t)], axis=1)),
        lambda xy, t: g_field.predict(np.concatenate([xy, np.full((len(xy), 1), t)], axis=1)),
        n_law, dt, dataset["n_steps"], law_seed, radius, excursion_radius)
    pairs = cfg["hypotheses"]["H5_law"]["time_pairs"]
    sigma = float(cfg["teacher"]["rbf_sigma"])
    observed = test["paths"]
    law = {
        "truth_counts": failure_counts(law_truth),
        "teacher_counts": failure_counts(law_teacher),
        "symbolic_counts": failure_counts(law_symbolic),
        "mmd2_truth_vs_observed": two_time_mmd(law_truth["states"], observed, pairs,
                                               mean, std, sigma),
        "mmd2_teacher_vs_observed": two_time_mmd(law_teacher["states"], observed, pairs,
                                                 mean, std, sigma),
        "mmd2_symbolic_vs_observed": two_time_mmd(law_symbolic["states"], observed, pairs,
                                                  mean, std, sigma),
        "mean_rmse_teacher_vs_observed": mean_rmse(mean_path(law_teacher["states"]),
                                                   mean_path(observed)),
        "mean_rmse_symbolic_vs_observed": mean_rmse(mean_path(law_symbolic["states"]),
                                                    mean_path(observed)),
        "mean_rmse_truth_vs_observed": mean_rmse(mean_path(law_truth["states"]),
                                                 mean_path(observed)),
    }

    probe_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    probe_t = np.zeros(len(probe_xy))
    point_probes = []
    for row, t_value in zip(probe_xy, probe_t):
        point_probes.append({
            "x": row.tolist(), "t": float(t_value),
            "truth_f": truth_f(row.reshape(1, 2), np.array([t_value]))[0].tolist(),
            "teacher_f": fdm_f(row.reshape(1, 2), np.array([t_value]))[0].tolist(),
            "symbolic_f": drift_field.predict(np.array([[row[0], row[1], t_value]]))[0].tolist(),
            "truth_g": truth_g(row.reshape(1, 2), np.array([t_value]))[0].tolist(),
            "teacher_g": fdm_g(row.reshape(1, 2), np.array([t_value]))[0].tolist(),
            "symbolic_g": g_field.predict(np.array([[row[0], row[1], t_value]]))[0].tolist(),
        })

    grid = diagnostic_grid(dataset["train"]["paths"].reshape(-1, 2),
                           [0.0, float(data_cfg["T"]) / 2.0, float(data_cfg["T"])],
                           fdm_f, fdm_g, drift_field, g_field, truth_f, truth_g)

    hyp = cfg["hypotheses"]
    test_m = heldout["test"]
    h1 = hyp["H1_drift"]
    h2 = hyp["H2_diffusion"]
    h4 = hyp["H4_simulation"]
    h5 = hyp["H5_law"]
    h6 = hyp["H6_marginal_baseline"]
    h7 = hyp["H7_discretization_gate"]
    fdm_g_nrmse = test_m["diffusion_teacher_vs_truth"]["nrmse"]
    marg_g_nrmse = test_m["marginal_diffusion_vs_truth"]["nrmse"]
    h7_ok = (discretization["terminal_rmse"]
             <= h7["teacher_coarse_vs_fine_terminal_rmse_max"]
             and discretization["retained_fraction"] >= h7["min_retained_fraction"])
    fail_teacher = sim_teacher["failure_rate"]
    fail_symbolic = sim_symbolic["failure_rate"]
    fail_truth = sim_truth["failure_rate"]
    h4_failure_ok = (fail_teacher <= h4["failure_rate_max"]
                     and fail_symbolic <= h4["failure_rate_max"]
                     and fail_truth <= h4["failure_rate_max"]
                     and abs(fail_symbolic - fail_teacher) <= h4["failure_rate_gap_max"])
    h4_excursion_gap = abs(sim_symbolic["excursion_rate"] - sim_teacher["excursion_rate"])
    h4_dist_ok = (paired["symbolic_vs_teacher"]["mean_rmse"]
                  <= h4["paired_mean_rmse_symbolic_vs_teacher_max"]
                  and paired["teacher_vs_truth"]["mean_rmse"]
                  <= h4["paired_mean_rmse_teacher_vs_truth_max"]
                  and paired["symbolic_vs_teacher"]["terminal_cov_rel_frob"] is not None
                  and paired["symbolic_vs_teacher"]["terminal_cov_rel_frob"]
                  <= h4["terminal_cov_rel_frob_symbolic_vs_teacher_max"])
    h4_gate_open = (max(fail_teacher, fail_symbolic, fail_truth) <= h4["failure_gate"]
                    and h7_ok)
    if h4_gate_open:
        h4_verdict = _pass(h4_failure_ok and h4_excursion_gap <= h4["excursion_rate_gap_max"]
                           and h4_dist_ok)
    else:
        h4_verdict = "indeterminate"
    law_fail = max(law_teacher["failure_rate"], law_symbolic["failure_rate"],
                   law_truth["failure_rate"])
    if law_fail > h5["failure_gate"]:
        h5_verdict = "indeterminate"
    else:
        h5_verdict = _pass(law["mmd2_teacher_vs_observed"] <= h5["teacher_mmd2_max"]
                           and law["mmd2_symbolic_vs_observed"] <= h5["symbolic_mmd2_max"]
                           and law["mean_rmse_teacher_vs_observed"]
                           <= h5["mean_path_rmse_teacher_vs_observed_max"]
                           and law["mean_rmse_symbolic_vs_observed"]
                           <= h5["mean_path_rmse_symbolic_vs_observed_max"])
    vif_value = vif["drift_library_degree_3"]
    verdicts = {
        "H1": _pass(test_m["drift_symbolic_vs_teacher"]["nrmse"]
                    <= h1["symbolic_vs_teacher_test_nrmse_max"]
                    and test_m["drift_teacher_vs_truth"]["nrmse"]
                    <= h1["teacher_vs_truth_test_nrmse_max"]
                    and test_m["drift_symbolic_vs_truth"]["nrmse"]
                    <= h1["symbolic_vs_truth_test_nrmse_max"]),
        "H2": _pass(test_m["diffusion_symbolic_vs_teacher"]["nrmse"]
                    <= h2["symbolic_vs_teacher_test_nrmse_max"]
                    and test_m["diffusion_teacher_vs_truth"]["nrmse"]
                    <= h2["teacher_vs_truth_test_nrmse_max"]
                    and test_m["diffusion_symbolic_vs_truth"]["nrmse"]
                    <= h2["symbolic_vs_truth_test_nrmse_max"]
                    and test_m["sigma_symbolic_vs_teacher"]["nrmse"]
                    <= h2["symbolic_vs_teacher_sigma_nrmse_max"]),
        "H2c": _pass(test_m["fraction_symbolic_g_nonpositive"]
                     <= hyp["H2c_range"]["fraction_symbolic_g_nonpositive_max"]),
        "H3": ("indeterminate" if vif_value > hyp["H3_collinearity_gate"]["vif_max"]
               else "pass"),
        "H4_failure": _pass(h4_failure_ok),
        "H4_excursion_gap": _pass(h4_excursion_gap <= h4["excursion_rate_gap_max"]),
        "H4": h4_verdict,
        "H5": h5_verdict,
        "H6": _pass(marg_g_nrmse - fdm_g_nrmse
                    >= h6["diffusion_nrmse_improvement_min"]),
        "H7": _pass(h7_ok),
    }

    results: dict[str, Any] = {
        "day": 12,
        "paper": cfg["paper"]["title"],
        "venue": cfg["paper"]["journal"],
        "seed": cfg["seed"],
        "git_head": git_head(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "probe": probe,
        "fdm_training": fdm_log,
        "marginal_training": marginal_log,
        "selection": {"drift": drift_meta, "diffusion": g_meta},
        "heldout": heldout,
        "ceilings": ceilings,
        "vif": vif,
        "snr": snr,
        "paired_simulation": paired,
        "discretization": discretization,
        "law": law,
        "grid": grid,
        "point_probes": point_probes,
        "query_dropped": {name: queries[name]["dropped"]
                          for name in ("train", "validation", "test")},
        "h6_diffusion_nrmse": {"fdm": fdm_g_nrmse, "marginal": marg_g_nrmse,
                               "improvement": marg_g_nrmse - fdm_g_nrmse},
        "h6_drift_nrmse": {"fdm": test_m["drift_teacher_vs_truth"]["nrmse"],
                           "marginal": test_m["marginal_drift_vs_truth"]["nrmse"]},
        "excursion_absolute": {"truth": sim_truth["excursion_rate"],
                               "teacher": sim_teacher["excursion_rate"],
                               "symbolic": sim_symbolic["excursion_rate"],
                               "gap_symbolic_minus_teacher": h4_excursion_gap},
        "verdicts": verdicts,
        "elapsed_seconds": float(time.time() - started),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drift_field.save(OUT_DIR / "symbolic_f.json")
    g_field.save(OUT_DIR / "symbolic_g.json")
    (OUT_DIR / "teacher_fdm_weights.json").write_text(json.dumps(fdm_weights))
    (OUT_DIR / "teacher_marginal_weights.json").write_text(json.dumps(marginal_weights))
    expanded_f = expand_to_original(drift_field)
    expanded_g = expand_to_original(g_field)
    tex = (
        "Symbolic SDE in the original van der Pol coordinates (x, y, t).\n"
        "The neural fields take time, so t is a coordinate of the power library.\n"
        "Coefficients are the expanded polynomial, not a claim of unique recovery.\n\n$$\n"
        + tex_equations(expanded_f, ("f_{1}(x, y, t)", "f_{2}(x, y, t)"),
                        var_names=("x", "y", "t"))
        + "\n$$\n\n$$\n"
        + tex_equations(expanded_g, ("g_{1}(x, y, t)", "g_{2}(x, y, t)"),
                        var_names=("x", "y", "t"))
        + "\n$$\n\n"
        "\\[\n\\begin{aligned}\n"
        "dx &= f_{1}(x, y, t)\\,dt + g_{1}(x, y, t)\\,dW^{1},\\\\\n"
        "dy &= f_{2}(x, y, t)\\,dt + g_{2}(x, y, t)\\,dW^{2}.\n"
        "\\end{aligned}\n\\]\n"
    )
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": [{",".join(map(str, exp)): coef for exp, coef in poly.items()}
                  for poly in expanded_f],
        "diffusion": [{",".join(map(str, exp)): coef for exp, coef in poly.items()}
                      for poly in expanded_g],
    }, indent=2))
    (OUT_DIR / "results.json").write_text(json.dumps(to_jsonable(results), indent=2))
    print(json.dumps(to_jsonable({
        "verdicts": verdicts,
        "selection": results["selection"],
        "test_drift_sym_teacher": test_m["drift_symbolic_vs_teacher"],
        "test_drift_teacher_truth": test_m["drift_teacher_vs_truth"],
        "test_g_sym_teacher": test_m["diffusion_symbolic_vs_teacher"],
        "test_g_teacher_truth": test_m["diffusion_teacher_vs_truth"],
        "h6": results["h6_diffusion_nrmse"],
        "law_mmd": {k: law[k] for k in law if k.startswith("mmd")},
        "vif": vif_value,
        "elapsed_seconds": results["elapsed_seconds"],
    }), indent=2))
    return results


if __name__ == "__main__":
    main()
