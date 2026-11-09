"""Day 17: Du, Honnappa, and Rao (UAI 2026) neural diffusion intensity -> SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day17.json``):

1. Data: Section 4.1. The latent intensity is the CIR equation
   ``dZ = 0.3(80-Z) dt + sqrt(Z) dW``, ``Z0 = 5``, on ``[0, 4]``.
   Observations are bin-wise Poisson counts of that intensity. The coordinate
   is a latent event rate, not a gene-regulatory state.
2. Teacher: a tanh MLP prior drift ``b_theta(z, t)`` and a remaining-count
   posterior correction, trained by the pathwise ELBO (Theorem 3.1). The
   diffusion coefficient stays the paper's ``sqrt(z)``. It is not a network.
3. Distillation: direct queries of the prior drift. The library is powers of
   ``(z, t)``. A second fit, powers of ``z`` only, probes ``sqrt(z)``.
   Trajectory-disjoint splits, validation-only selection, test once.
4. Simulation keeps failed-trajectory mass and separates non-finite states,
   crossings of the intensity floor, and escapes.

Run from the repository root:
``python -m src.daily_neural_sde.day17_du_ndm``

Outputs land in ``outputs/daily_neural_sde/day17/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day17.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day17"
DTYPE = torch.float64

DriftFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
SigmaFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-17 config."""
    return json.loads(path.read_text())


def n_steps(cfg: dict[str, Any]) -> int:
    """Euler steps. ``T`` must be an integer multiple of ``dt``."""
    steps = int(cfg["data"]["n_steps"])
    t = float(cfg["data"]["T"])
    dt = float(cfg["data"]["dt"])
    if abs(steps * dt - t) > 1e-9:
        raise ValueError("n_steps * dt must equal T")
    return steps


def true_drift(z: np.ndarray, kappa: float = 0.3, theta: float = 80.0) -> np.ndarray:
    """CIR drift ``kappa * (theta - z)``, shape (n,)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    return kappa * (theta - z)


def true_sigma(z: np.ndarray, sigma_scale: float = 1.0) -> np.ndarray:
    """CIR diffusion ``sigma_scale * sqrt(z)`` for ``z > 0``, shape (n,)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if np.any(z <= 0):
        raise ValueError("CIR diffusion is queried on a nonpositive state")
    return sigma_scale * np.sqrt(z)


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def rms(values: np.ndarray) -> float:
    """Root mean square of every entry."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array ** 2)))


def glorot_(tensor: torch.Tensor) -> None:
    """Glorot uniform. Biases and 1-D tensors are set to zero."""
    if tensor.ndim < 2:
        tensor.zero_()
        return
    fan_in, fan_out = tensor.shape[1], tensor.shape[0]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


class MLP(nn.Module):
    """Tanh MLP. ``depth`` is the number of hidden layers.

    Input shape (n, in_dim), output shape (n, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * int(depth) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (n, in_dim) to (n, out_dim)."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class IntensityModel(nn.Module):
    """Prior drift and remaining-count correction.

    Drift inputs are standardized ``(z, t)``. Correction inputs are
    standardized ``(z, t, r)``. Both outputs are in original units: the drift
    is ``b``, and the correction head is the scalar whose product with
    ``sigma`` is ``u`` in equation (3).
    """

    def __init__(self, hidden: int, depth: int, project_floor: float) -> None:
        super().__init__()
        self.drift_net = MLP(2, 1, hidden, depth)
        self.corr_net = MLP(3, 1, hidden, depth)
        self.project_floor = float(project_floor)
        self.register_buffer("drift_mean", torch.zeros(2, dtype=DTYPE))
        self.register_buffer("drift_scale", torch.ones(2, dtype=DTYPE))
        self.register_buffer("corr_mean", torch.zeros(3, dtype=DTYPE))
        self.register_buffer("corr_scale", torch.ones(3, dtype=DTYPE))

    def set_scalers(self, drift_mean: np.ndarray, drift_scale: np.ndarray,
                    corr_mean: np.ndarray, corr_scale: np.ndarray) -> None:
        """Install train-only affine scalers. Arrays have shape (2,) and (3,)."""
        self.drift_mean.copy_(torch.as_tensor(drift_mean, dtype=DTYPE))
        self.drift_scale.copy_(torch.as_tensor(drift_scale, dtype=DTYPE))
        self.corr_mean.copy_(torch.as_tensor(corr_mean, dtype=DTYPE))
        self.corr_scale.copy_(torch.as_tensor(corr_scale, dtype=DTYPE))

    def drift(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Prior drift, shape matching ``z``."""
        raw = torch.stack(((z - self.drift_mean[0]) / self.drift_scale[0],
                           (t - self.drift_mean[1]) / self.drift_scale[1]), dim=-1)
        return self.drift_net(raw).squeeze(-1)

    def correction_raw(self, z: torch.Tensor, t: torch.Tensor,
                       remain: torch.Tensor) -> torch.Tensor:
        """MLP factor in ``u = sigma * MLP``. Shape matches ``z``."""
        raw = torch.stack((
            (z - self.corr_mean[0]) / self.corr_scale[0],
            (t - self.corr_mean[1]) / self.corr_scale[1],
            (remain - self.corr_mean[2]) / self.corr_scale[2]), dim=-1)
        return self.corr_net(raw).squeeze(-1)


def build_model(cfg: dict[str, Any]) -> IntensityModel:
    """Float64 model with Glorot weights and a shrunk last layer."""
    model = IntensityModel(int(cfg["model"]["hidden"]), int(cfg["model"]["depth"]),
                           float(cfg["data"]["project_floor"]))
    model = model.to(dtype=DTYPE)
    with torch.no_grad():
        for parameter in model.parameters():
            glorot_(parameter)
        scale = float(cfg["model"]["last_layer_scale"])
        model.drift_net.layers[-1].weight.mul_(scale)
        model.corr_net.layers[-1].weight.mul_(scale)
    return model


def simulate_truth_batch(n: int, rng: np.random.Generator,
                         cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Euler paths and bin-wise Poisson counts. Shapes (n, steps).

    A path is marked failed if it leaves ``(failure_floor, failure_radius)``
    or becomes non-finite. Counts on failed paths are zero and unused.
    """
    data = cfg["data"]
    steps = n_steps(cfg)
    dt = float(data["dt"])
    kappa, theta = float(data["kappa"]), float(data["theta"])
    scale = float(data["sigma_scale"])
    floor = float(data["failure_floor"])
    radius = float(data["failure_radius"])
    z = np.full(n, float(data["z0"]), dtype=np.float64)
    left = np.empty((n, steps), dtype=np.float64)
    failed = np.zeros(n, dtype=bool)
    for step in range(steps):
        left[:, step] = z
        active = ~failed
        shock = rng.normal(0.0, 1.0, size=n)
        nxt = z.copy()
        nxt[active] = (z[active] + true_drift(z[active], kappa, theta) * dt
                       + scale * np.sqrt(z[active]) * math.sqrt(dt) * shock[active])
        bad = active & (~np.isfinite(nxt) | (nxt < floor) | (nxt > radius))
        failed |= bad
        nxt[failed] = np.nan
        z = nxt
    lam = np.where(failed[:, None], 0.0, left * dt)
    counts = rng.poisson(lam).astype(np.int64)
    counts[failed] = 0
    return {"left": left, "counts": counts, "failed": failed}


def make_split(n: int, rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, Any]:
    """Draw ``n`` trajectories that stay inside the floor and the radius."""
    factor = int(cfg["data"]["max_resample_factor"])
    kept_left: list[np.ndarray] = []
    kept_counts: list[np.ndarray] = []
    rejected = 0
    attempts = 0
    while sum(len(chunk) for chunk in kept_left) < n:
        attempts += 1
        if attempts > factor:
            raise RuntimeError("Resample cap hit while generating CIR paths")
        need = n - sum(len(chunk) for chunk in kept_left)
        batch = simulate_truth_batch(need, rng, cfg)
        keep = ~batch["failed"]
        rejected += int((~keep).sum())
        if keep.any():
            kept_left.append(batch["left"][keep])
            kept_counts.append(batch["counts"][keep])
    left = np.concatenate(kept_left, axis=0)[:n]
    counts = np.concatenate(kept_counts, axis=0)[:n]
    return {"left": left, "counts": counts, "rejected": rejected}


def make_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Trajectory-disjoint train, validation, and test from one seed stream."""
    rng = np.random.default_rng(int(cfg["seeds"]["data"]))
    data = cfg["data"]
    train = make_split(int(data["n_train"]), rng, cfg)
    val = make_split(int(data["n_val"]), rng, cfg)
    test = make_split(int(data["n_test"]), rng, cfg)
    return {"train": train, "val": val, "test": test}


def remaining_counts(counts: np.ndarray) -> np.ndarray:
    """Events in the current bin and later bins. Shape (n, steps)."""
    flipped = np.cumsum(counts[:, ::-1], axis=1)
    return np.ascontiguousarray(flipped[:, ::-1])


def fit_scalers(train: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Mean and scale of train ``(z, t)`` and ``(z, t, r)`` only."""
    steps = n_steps(cfg)
    dt = float(cfg["data"]["dt"])
    z = train["left"].reshape(-1)
    t = np.tile(np.arange(steps, dtype=np.float64) * dt, train["left"].shape[0])
    remain = remaining_counts(train["counts"]).reshape(-1).astype(np.float64)
    drift_x = np.stack([z, t], axis=1)
    corr_x = np.stack([z, t, remain], axis=1)

    def mean_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return mean, scale

    drift_mean, drift_scale = mean_scale(drift_x)
    corr_mean, corr_scale = mean_scale(corr_x)
    return {"drift_mean": drift_mean, "drift_scale": drift_scale,
            "corr_mean": corr_mean, "corr_scale": corr_scale}


def rollout(model: IntensityModel, z0: np.ndarray, counts: np.ndarray, times: np.ndarray,
            dt: float, noise: torch.Tensor, project: bool, drift_source: str,
            corr_source: str, kappa: float, theta: float,
            sigma_scale: float) -> dict[str, torch.Tensor]:
    """Euler rollout of the variational SDE.

    ``z0`` is (batch,), ``counts`` is (batch, steps), ``noise`` is
    (batch, paths, steps) of standard normals. Returns the ELBO per
    (batch, path) and the left states (batch, paths, steps).
    """
    batch, paths, steps = noise.shape
    if counts.shape != (batch, steps):
        raise ValueError("counts shape mismatch")
    z = torch.as_tensor(z0, dtype=DTYPE).reshape(batch, 1).expand(batch, paths).clone()
    count_t = torch.as_tensor(counts, dtype=DTYPE)
    remain = torch.as_tensor(remaining_counts(counts), dtype=DTYPE)
    time_t = torch.as_tensor(times, dtype=DTYPE)
    lefts = []
    loglik = torch.zeros(batch, paths, dtype=DTYPE)
    kl = torch.zeros(batch, paths, dtype=DTYPE)
    floor = model.project_floor
    for step in range(steps):
        t = time_t[step].expand(batch, paths)
        r = remain[:, step].reshape(batch, 1).expand(batch, paths)
        if drift_source == "network":
            drift = model.drift(z, t)
        elif drift_source == "truth":
            drift = torch.as_tensor(kappa, dtype=DTYPE) * (torch.as_tensor(theta, dtype=DTYPE) - z)
        elif drift_source == "zero":
            drift = torch.zeros_like(z)
        else:
            raise ValueError(drift_source)
        sigma = torch.as_tensor(sigma_scale, dtype=DTYPE) * torch.sqrt(torch.clamp(z, min=floor))
        if corr_source == "network":
            raw = model.correction_raw(z, t, r)
        elif corr_source == "zero":
            raw = torch.zeros_like(z)
        else:
            raise ValueError(corr_source)
        # Remaining-count option: u = sigma * MLP. Equation (3) adds sigma * u,
        # and the Girsanov penalty is 0.5 u^2. sigma^2 = z when sigma_scale is 1.
        control = sigma * raw
        lefts.append(z)
        loglik = loglik + count_t[:, step].reshape(batch, 1) * torch.log(z) - z * dt
        kl = kl + 0.5 * control ** 2 * dt
        z = z + (drift + sigma * control) * dt + sigma * math.sqrt(dt) * noise[:, :, step]
        if project:
            z = torch.clamp(z, min=floor)
    return {"elbo": loglik - kl, "left": torch.stack(lefts, dim=-1)}


def _batch_noise(batch: int, paths: int, steps: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(batch, paths, steps, generator=generator, dtype=DTYPE)


def train_model(model: IntensityModel, data: dict[str, Any], scalers: dict[str, np.ndarray],
                cfg: dict[str, Any]) -> dict[str, Any]:
    """Maximize the ELBO. The checkpoint is the best validation ELBO."""
    model.set_scalers(scalers["drift_mean"], scalers["drift_scale"],
                      scalers["corr_mean"], scalers["corr_scale"])
    model.train()
    train, val = data["train"], data["val"]
    steps = n_steps(cfg)
    dt = float(cfg["data"]["dt"])
    times = np.arange(steps, dtype=np.float64) * dt
    kappa = float(cfg["data"]["kappa"])
    theta = float(cfg["data"]["theta"])
    sigma_scale = float(cfg["data"]["sigma_scale"])
    m_paths = int(cfg["model"]["n_brownian"])
    batch_size = int(cfg["model"]["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["model"]["lr"]))
    train_gen = torch.Generator()
    train_gen.manual_seed(int(cfg["seeds"]["train_noise"]))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = -float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    n_train = train["left"].shape[0]
    z0 = np.full(n_train, float(cfg["data"]["z0"]))
    z0_val = np.full(val["left"].shape[0], float(cfg["data"]["z0"]))
    py_rng = np.random.default_rng(int(cfg["seeds"]["train_noise"]))
    for epoch in range(int(cfg["model"]["epochs"])):
        order = py_rng.permutation(n_train)
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            noise = _batch_noise(len(index), m_paths, steps, train_gen)
            out = rollout(model, z0[index], train["counts"][index], times, dt, noise,
                          project=True, drift_source="network", corr_source="network",
                          kappa=kappa, theta=theta, sigma_scale=sigma_scale)
            loss = -out["elbo"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(param.grad).all() for param in model.parameters()
                       if param.grad is not None):
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["model"]["grad_clip"]))
            optimizer.step()
            total += float(out["elbo"].detach().sum())
            seen += int(out["elbo"].numel())
        model.eval()
        val_gen = torch.Generator()
        val_gen.manual_seed(int(cfg["seeds"]["val_noise"]))
        with torch.no_grad():
            val_noise = _batch_noise(len(z0_val), m_paths, steps, val_gen)
            val_out = rollout(model, z0_val, val["counts"], times, dt, val_noise,
                              project=True, drift_source="network", corr_source="network",
                              kappa=kappa, theta=theta, sigma_scale=sigma_scale)
            val_elbo = float(val_out["elbo"].mean())
        train_elbo = total / max(seen, 1)
        history.append({"epoch": epoch, "train_elbo": train_elbo, "val_elbo": val_elbo})
        if math.isfinite(val_elbo) and val_elbo > best_val:
            best_val = val_elbo
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == int(cfg["model"]["epochs"]) - 1:
            print(f"epoch {epoch:03d} train {train_elbo:.3f} val {val_elbo:.3f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_val_elbo": best_val, "history": history}


def reference_elbo(model: IntensityModel, split: dict[str, Any], cfg: dict[str, Any],
                   seed: int, drift_source: str, corr_source: str) -> float:
    """Mean ELBO of one split under a fixed drift and correction source."""
    steps = n_steps(cfg)
    dt = float(cfg["data"]["dt"])
    times = np.arange(steps, dtype=np.float64) * dt
    z0 = np.full(split["left"].shape[0], float(cfg["data"]["z0"]))
    generator = torch.Generator()
    generator.manual_seed(seed)
    noise = _batch_noise(len(z0), int(cfg["model"]["n_brownian"]), steps, generator)
    model.eval()
    with torch.no_grad():
        out = rollout(model, z0, split["counts"], times, dt, noise, project=True,
                      drift_source=drift_source, corr_source=corr_source,
                      kappa=float(cfg["data"]["kappa"]), theta=float(cfg["data"]["theta"]),
                      sigma_scale=float(cfg["data"]["sigma_scale"]))
    return float(out["elbo"].mean())


def query_frame(split: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Left states, times, and the analytic drift and diffusion. Shapes (n,)."""
    steps = n_steps(cfg)
    dt = float(cfg["data"]["dt"])
    n_traj = split["left"].shape[0]
    z = split["left"].reshape(-1)
    # C-order flatten of (trajectory, time) repeats the time grid once per trajectory.
    t = np.tile(np.arange(steps, dtype=np.float64) * dt, (n_traj, 1)).reshape(-1)
    kappa = float(cfg["data"]["kappa"])
    theta = float(cfg["data"]["theta"])
    return {"z": z, "t": t, "state": np.stack([z, t], axis=1),
            "drift": true_drift(z, kappa, theta),
            "sigma": true_sigma(z, float(cfg["data"]["sigma_scale"])),
            "traj_index": np.repeat(np.arange(n_traj), steps)}


def drift_numpy(model: IntensityModel, state: np.ndarray) -> np.ndarray:
    """Teacher prior drift at rows ``(z, t)``. Shape (n,)."""
    state = np.atleast_2d(np.asarray(state, dtype=np.float64))
    z = torch.as_tensor(state[:, 0], dtype=DTYPE)
    t = torch.as_tensor(state[:, 1], dtype=DTYPE)
    model.eval()
    with torch.no_grad():
        values = model.drift(z, t)
    return values.cpu().numpy()


def state_dict_numpy(model: IntensityModel) -> dict[str, list[Any]]:
    """JSON-ready weights, including the train scalers."""
    payload = {}
    for key, value in model.state_dict().items():
        payload[key] = value.detach().cpu().numpy().tolist()
    return payload


def vif_max(x: np.ndarray, degree: int) -> float:
    """Max VIF of a polynomial library. The constant column is excluded."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    powers = monomial_powers(x.shape[1], degree)
    theta = evaluate_powers((x - mean) / scale, powers)
    keep = [j for j, power in enumerate(powers) if int(power.sum()) > 0]
    theta = theta[:, keep]
    if theta.shape[1] == 0:
        return 1.0
    worst = 1.0
    for j in range(theta.shape[1]):
        others = np.delete(theta, j, axis=1)
        if others.shape[1] == 0:
            continue
        coef, *_ = np.linalg.lstsq(others, theta[:, j], rcond=None)
        residual = theta[:, j] - others @ coef
        total = float(np.sum((theta[:, j] - theta[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual ** 2)) / max(total, 1e-15)
        worst = max(worst, 1.0 / max(1.0 - r2, 1e-12))
    return float(worst)


def select_field(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, cfg: dict[str, Any]) -> tuple[PolynomialField, dict[str, Any]]:
    """Minimize validation NRMSE. Ties prefer the smaller degree, then threshold."""
    best: PolynomialField | None = None
    best_key: tuple[float, int, float] | None = None
    records = []
    ridge = float(cfg["distill"]["ridge"])
    y_train_2d = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    y_val_flat = np.asarray(y_val, dtype=np.float64).reshape(-1)
    for degree in cfg["distill"]["degrees"]:
        for threshold in cfg["distill"]["thresholds"]:
            field = fit_polynomial_field(x_train, y_train_2d, int(degree), float(threshold), ridge)
            score = nrmse(y_val_flat, field.predict(x_val)[:, 0])
            record = {"degree": int(degree), "threshold": float(threshold), "score": score,
                      "active_terms": int(np.count_nonzero(field.coefficients))}
            records.append(record)
            key = (score, int(degree), float(threshold))
            if best_key is None or key < best_key:
                best_key = key
                best = field
    assert best is not None and best_key is not None
    return best, {"selected": {"degree": best_key[1], "threshold": best_key[2],
                               "score": best_key[0],
                               "active_terms": int(np.count_nonzero(best.coefficients))},
                  "candidates": records}


def cubic_ratio(field_target: np.ndarray, state: np.ndarray, degree: int,
                threshold: float, ridge: float) -> float:
    """RMS of the selected field's degree-3 part, divided by the target RMS.

    Zero when the selected degree is below 3.
    """
    if degree < 3:
        return 0.0
    target = np.asarray(field_target, dtype=np.float64).reshape(-1, 1)
    field = fit_polynomial_field(state, target, degree, threshold, ridge)
    powers = field.powers
    coef = field.coefficients.copy()
    for j, power in enumerate(powers):
        if int(np.sum(power)) < 3:
            coef[:, j] = 0.0
    trimmed = PolynomialField(powers, coef, field.mean, field.scale, {})
    return rms(trimmed.predict(state)[:, 0]) / max(rms(target[:, 0]), 1e-12)


def simulate_scalar(drift_fn: DriftFn, sigma_fn: SigmaFn, z0: np.ndarray, dt: float,
                    increments: np.ndarray, floor: float, radius: float,
                    project_floor: float | None = None) -> dict[str, Any]:
    """Unprojected Euler-Maruyama unless ``project_floor`` is set.

    ``increments`` has shape (n, steps) of standard normals. Failures are
    latched as ``non_finite``, ``below_floor``, or ``outside_radius``.
    """
    z = np.array(z0, dtype=np.float64, copy=True).reshape(-1)
    n, steps = increments.shape
    if len(z) != n:
        raise ValueError("z0 length mismatch")
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    observed = [z.copy()]
    sqrt_dt = math.sqrt(dt)
    for step in range(steps):
        active = ~failed
        t = np.full(int(active.sum()), step * dt)
        nxt = z.copy()
        if active.any():
            drift = np.asarray(drift_fn(z[active], t), dtype=np.float64).reshape(-1)
            sigma = np.asarray(sigma_fn(z[active]), dtype=np.float64).reshape(-1)
            nxt[active] = z[active] + drift * dt + sigma * sqrt_dt * increments[active, step]
            if project_floor is not None:
                nxt[active] = np.maximum(nxt[active], project_floor)
        nonfinite = ~np.isfinite(nxt)
        below = np.isfinite(nxt) & (nxt < floor)
        outside = np.isfinite(nxt) & (np.abs(nxt) > radius)
        newly = (~failed) & (nonfinite | below | outside)
        for idx in np.flatnonzero(newly):
            if nonfinite[idx]:
                failure_class[idx] = "non_finite"
            elif below[idx]:
                failure_class[idx] = "below_floor"
            else:
                failure_class[idx] = "outside_radius"
        failed |= newly
        nxt[failed] = np.nan
        z = nxt
        observed.append(z.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class, "failure_rate": float(np.mean(failed))}


def class_counts(failure_class: list[str | None]) -> dict[str, int]:
    """Counts of latched failure classes. Survivors are omitted."""
    counts = {"non_finite": 0, "below_floor": 0, "outside_radius": 0}
    for label in failure_class:
        if label is not None:
            counts[label] += 1
    return counts


def finite_path_rmse(left: np.ndarray, right: np.ndarray, fail_left: np.ndarray,
                     fail_right: np.ndarray) -> float:
    """RMSE on trajectories that both survive. NaN if none do."""
    both = ~(fail_left | fail_right)
    if not both.any():
        return float("nan")
    return rms(left[both] - right[both])


def terminal_var(states: np.ndarray, failed: np.ndarray) -> float | None:
    """Variance of finite terminal states. None if fewer than two survive."""
    values = states[~failed, -1]
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    return float(np.var(values))


def rel_var(left: float | None, right: float | None) -> float | None:
    """Absolute relative difference of two variances."""
    if left is None or right is None or not math.isfinite(right) or abs(right) < 1e-12:
        return None
    return abs(left - right) / abs(right)


def mean_curve(states: np.ndarray) -> np.ndarray:
    """Time-wise mean. NaN marks a time at which every trajectory has failed."""
    with np.errstate(all="ignore"):
        return np.nanmean(states, axis=0)


def gaussian_nll(increment: np.ndarray, drift: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
    """One-dimensional Euler-Gauss NLL, shape (n,). The ``log(2 pi)`` term is kept."""
    residual = increment - drift * dt
    scale = sigma * math.sqrt(dt)
    quad = (residual / scale) ** 2
    return 0.5 * (quad + np.log(2.0 * math.pi * scale ** 2))


def time_sensitivity(predict: Callable[[np.ndarray], np.ndarray], state: np.ndarray,
                     shuffled_time: np.ndarray) -> float:
    """RMS of ``f(z, t) - f(z, t')`` divided by the RMS of ``f(z, t)``.

    ``shuffled_time`` has shape (n,) and is independent of the path time.
    """
    shuffled = np.array(state, dtype=np.float64, copy=True)
    shuffled[:, 1] = np.asarray(shuffled_time, dtype=np.float64)
    base = np.asarray(predict(state), dtype=np.float64).reshape(-1)
    moved = np.asarray(predict(shuffled), dtype=np.float64).reshape(-1)
    return rms(base - moved) / max(rms(base), 1e-12)


def product_grid(z_values: np.ndarray, horizon: float, n_t: int) -> np.ndarray:
    """Independent ``(z, t)`` pairs, shape (n_z * n_t, 2)."""
    times = np.linspace(0.0, horizon, n_t)
    zz, tt = np.meshgrid(z_values, times, indexing="ij")
    return np.stack([zz.ravel(), tt.ravel()], axis=1)


def bootstrap_support(x: np.ndarray, y: np.ndarray, traj: np.ndarray, degree: int,
                      threshold: float, ridge: float, n_boot: int,
                      rng: np.random.Generator) -> dict[str, Any]:
    """Trajectory bootstrap of the selected STLSQ support."""
    y_2d = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    chosen = fit_polynomial_field(x, y_2d, degree, threshold, ridge)
    chosen_on = np.abs(chosen.coefficients[0]) > 0
    ids = np.unique(traj)
    matches = 0
    coefs = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        parts_x = []
        parts_y = []
        for item in draw:
            take = traj == item
            parts_x.append(x[take])
            parts_y.append(y_2d[take])
        field = fit_polynomial_field(np.concatenate(parts_x), np.concatenate(parts_y),
                                     degree, threshold, ridge)
        active = np.abs(field.coefficients[0]) > 0
        matches += int(np.array_equal(active, chosen_on))
        coefs.append(field.coefficients[0])
    stacked = np.stack(coefs, axis=0)
    return {"support_frequency": matches / max(n_boot, 1),
            "coef_std_max": float(np.max(stacked.std(axis=0))),
            "n_boot": n_boot}


def posterior_mean_rmse(model: IntensityModel, split: dict[str, Any], cfg: dict[str, Any],
                        corr_source: str, seed: int) -> dict[str, Any]:
    """RMSE of the mean latent path against the data-generating left states."""
    steps = n_steps(cfg)
    dt = float(cfg["data"]["dt"])
    times = np.arange(steps, dtype=np.float64) * dt
    z0 = np.full(split["left"].shape[0], float(cfg["data"]["z0"]))
    generator = torch.Generator()
    generator.manual_seed(seed)
    m_paths = int(cfg["distill"]["posterior_paths"])
    noise = _batch_noise(len(z0), m_paths, steps, generator)
    model.eval()
    with torch.no_grad():
        out = rollout(model, z0, split["counts"], times, dt, noise, project=True,
                      drift_source="network", corr_source=corr_source,
                      kappa=float(cfg["data"]["kappa"]), theta=float(cfg["data"]["theta"]),
                      sigma_scale=float(cfg["data"]["sigma_scale"]))
    left = out["left"].cpu().numpy()
    # A sample fails when it crosses the evaluation floor or the radius or is non-finite.
    floor = float(cfg["data"]["failure_floor"])
    radius = float(cfg["data"]["failure_radius"])
    bad = ~np.isfinite(left) | (left < floor) | (left > radius)
    # Latch failures forward so a later bad step does not re-enter the mean.
    latched = np.cumsum(bad, axis=-1) > 0
    left = np.where(latched, np.nan, left)
    failed = latched[..., -1]
    mean_path = np.nanmean(left, axis=1)
    truth = split["left"]
    usable = np.isfinite(mean_path)
    if not usable.any():
        score = float("nan")
    else:
        score = rms(mean_path[usable] - truth[usable])
    return {"rmse": score, "failure_rate": float(np.mean(failed)),
            "all_samples_failed_fraction": float(np.mean(~usable))}


def drift_pack(predict: Callable[[np.ndarray], np.ndarray], truth: np.ndarray,
               teacher: np.ndarray, state: np.ndarray) -> dict[str, float]:
    """NRMSE and residual RMS of one drift field against truth and the teacher."""
    pred = predict(state)
    return {"sym_teacher_nrmse": nrmse(teacher, pred),
            "teacher_truth_nrmse": nrmse(truth, teacher),
            "sym_truth_nrmse": nrmse(truth, pred),
            "sym_teacher_rms": rms(pred - teacher),
            "teacher_truth_rms": rms(teacher - truth),
            "sym_truth_rms": rms(pred - truth),
            "teacher_rms": rms(teacher),
            "truth_rms": rms(truth)}


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Thresholds are read from ``cfg`` and not modified."""
    hyp = cfg["hypotheses"]
    h1, h2 = hyp["H1_drift"], hyp["H2_diffusion_probe"]
    h4, h5 = hyp["H4_simulation"], hyp["H5_law"]
    h6, h7, h8 = hyp["H6_time_and_grid"], hyp["H7_discretization_gate"], hyp["H8_posterior"]
    verdicts: dict[str, Any] = {
        "H1": _pass(
            metrics["drift_test"]["sym_teacher_nrmse"] <= h1["symbolic_vs_teacher_test_nrmse_max"]
            and metrics["drift_test"]["teacher_truth_nrmse"] <= h1["teacher_vs_truth_test_nrmse_max"]
            and metrics["drift_test"]["sym_truth_nrmse"] <= h1["symbolic_vs_truth_test_nrmse_max"]),
        "H2": _pass(
            metrics["sigma_test_nrmse"] <= h2["symbolic_vs_sigma_test_nrmse_max"]
            and metrics["sigma_nonpositive_fraction"] <= h2["fraction_nonpositive_max"]),
        "H2_local_shadow": bool(
            metrics["sigma_low_z_nrmse"]
            > h2["low_z_over_ontraj_factor"] * max(metrics["sigma_test_nrmse"], 1e-8)),
        "H2_is_square_root": False,
        "H3": ("indeterminate" if metrics["vif_selected"] > hyp["H3_collinearity_gate"]["vif_max"]
               else "pass"),
        "H3_closed_form": False,
    }
    verdicts["H4"] = _simulation_verdict(h4, metrics["fail_truth"], metrics["fail_teacher"],
                                         metrics["fail_symbolic"], metrics["paired_sym_teacher"],
                                         metrics["paired_teacher_truth"], metrics["var_sym_teacher"],
                                         metrics["var_teacher_truth"],
                                         metrics["drift_test"]["teacher_truth_nrmse"])
    verdicts["H4b"] = _simulation_verdict(h4, metrics["fail_truth"], metrics["fail_teacher"],
                                          metrics["fail_poly"], metrics["paired_poly_teacher"],
                                          metrics["paired_teacher_truth"], metrics["var_poly_teacher"],
                                          metrics["var_teacher_truth"],
                                          metrics["drift_test"]["teacher_truth_nrmse"])
    h7_ok = (math.isfinite(metrics["disc_rmse"])
             and metrics["disc_rmse"] <= h7["teacher_coarse_vs_fine_terminal_rmse_max"]
             and metrics["disc_retained"] >= h7["min_retained_fraction"])
    verdicts["H7"] = _pass(h7_ok)
    law_fail = max(metrics["law_fail_truth"], metrics["law_fail_teacher"], metrics["law_fail_symbolic"])
    floor_blocks = (not math.isfinite(metrics["floor_mean"])
                    or metrics["floor_mean"] > h5["mean_path_rmse_teacher_max"]
                    or metrics["floor_var"] is None
                    or not math.isfinite(metrics["floor_var"])
                    or metrics["floor_var"] > h5["terminal_var_rel_teacher_max"])
    law_ok = (math.isfinite(metrics["law_mean_teacher"])
              and math.isfinite(metrics["law_mean_symbolic"])
              and metrics["law_var_teacher"] is not None
              and metrics["law_var_symbolic"] is not None
              and metrics["law_mean_teacher"] <= h5["mean_path_rmse_teacher_max"]
              and metrics["law_mean_symbolic"] <= h5["mean_path_rmse_symbolic_max"]
              and metrics["law_var_teacher"] <= h5["terminal_var_rel_teacher_max"]
              and metrics["law_var_symbolic"] <= h5["terminal_var_rel_symbolic_max"])
    verdicts["H5"] = "indeterminate" if law_fail > h5["failure_gate"] or floor_blocks else _pass(law_ok)
    if metrics["time_teacher"] > h6["time_sensitivity_teacher_max"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "teacher_uses_time"
    elif metrics["time_symbolic"] > h6["time_sensitivity_symbolic_max"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "symbolic_uses_time"
    elif metrics["grid_teacher_truth_nrmse"] > h6["grid_teacher_vs_truth_nrmse_max"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "teacher_grid"
    elif metrics["grid_sym_teacher_nrmse"] > h6["grid_symbolic_vs_teacher_nrmse_max"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "symbolic_grid"
    else:
        verdicts["H6"] = "pass"
        verdicts["H6_reason"] = "time_homogeneous_on_and_off_path"
    if metrics["post_fail"] > h8["failure_gate"] or metrics["prior_fail"] > h8["failure_gate"]:
        verdicts["H8"] = "indeterminate"
    else:
        improved = (math.isfinite(metrics["post_rmse"]) and math.isfinite(metrics["prior_rmse"])
                    and metrics["post_rmse"] + h8["min_improvement_over_prior"] <= metrics["prior_rmse"])
        verdicts["H8"] = _pass(metrics["post_rmse"] <= h8["posterior_mean_path_rmse_max"] and improved)
    blind = metrics["step_ratio"] < float(cfg["distill"]["step_ratio_drift_blind"])
    nll_gap = abs(metrics["nll_teacher"] - metrics["nll_truth"])
    verdicts["drift_step_ratio_blind"] = bool(blind)
    verdicts["nll_hides_drift"] = bool(
        nll_gap < float(cfg["distill"]["nll_floor_gap"])
        and metrics["drift_test"]["teacher_truth_nrmse"] > h1["teacher_vs_truth_test_nrmse_max"])
    verdicts["coefficients_are_closed_form"] = False
    verdicts["diffusion_teacher_is_neural"] = False
    return verdicts


def _simulation_verdict(h4: dict[str, Any], fail_truth: float, fail_teacher: float,
                        fail_model: float, paired_model: float, paired_truth: float,
                        var_model: float | None, var_truth: float | None,
                        drift_nrmse: float) -> str:
    """Shared gate for the square-root diffusion model and the polynomial probe."""
    fail_ok = (fail_truth <= h4["failure_rate_max"] and fail_teacher <= h4["failure_rate_max"]
               and fail_model <= h4["failure_rate_max"]
               and abs(fail_model - fail_teacher) <= h4["failure_rate_gap_max"])
    gate = (max(fail_truth, fail_teacher, fail_model) <= h4["failure_gate"]
            and var_model is not None and var_truth is not None)
    path_ok = (math.isfinite(paired_model) and math.isfinite(paired_truth)
               and paired_model <= h4["paired_path_rmse_symbolic_vs_teacher_max"]
               and paired_truth <= h4["paired_path_rmse_teacher_vs_truth_max"])
    var_ok = (var_model is not None and var_truth is not None
              and var_model <= h4["terminal_var_rel_symbolic_vs_teacher_max"]
              and var_truth <= h4["terminal_var_rel_teacher_vs_truth_max"])
    drift_ok = drift_nrmse <= h4["drift_teacher_vs_truth_test_nrmse_max"]
    if not gate:
        return "indeterminate"
    return _pass(fail_ok and path_ok and var_ok and drift_ok)


def git_head() -> str:
    """Current commit, or ``unknown`` if git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def jsonify(obj: Any) -> Any:
    """Recursively replace NumPy and torch values with JSON-native ones."""
    if isinstance(obj, dict):
        return {str(key): jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def polynomial_sigma_fn(field: PolynomialField) -> SigmaFn:
    """Diffusion probe. The input state is ``(z, t)``; only ``z`` is used."""

    def evaluate(z: np.ndarray, _t: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        return field.predict(z.reshape(-1, 1))[:, 0]

    return evaluate


def paper_sigma_fn(cfg: dict[str, Any]) -> SigmaFn:
    """Paper diffusion. Nonpositive states return NaN and fail the step."""
    scale = float(cfg["data"]["sigma_scale"])

    def evaluate(z: np.ndarray, _t: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        out = np.full(len(z), np.nan)
        ok = z > 0
        out[ok] = scale * np.sqrt(z[ok])
        return out

    return evaluate


def make_drift_fn(kind: str, model: IntensityModel, field: PolynomialField | None,
                  cfg: dict[str, Any]) -> DriftFn:
    """``truth``, ``teacher``, or ``symbolic`` drift of shape (n,)."""
    kappa = float(cfg["data"]["kappa"])
    theta = float(cfg["data"]["theta"])

    def evaluate(z: np.ndarray, t: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        if kind == "truth":
            return true_drift(z, kappa, theta)
        if kind == "teacher":
            return drift_numpy(model, np.stack([z, t], axis=1))
        if kind == "symbolic":
            assert field is not None
            return field.predict(np.stack([z, t], axis=1))[:, 0]
        raise ValueError(kind)

    return evaluate


def run_simulation(drift_fn: DriftFn, sigma_fn: SigmaFn, z0: np.ndarray, dt: float,
                   increments: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluation Euler scheme. No projection."""
    return simulate_scalar(drift_fn, sigma_fn, z0, dt, increments,
                           floor=float(cfg["data"]["failure_floor"]),
                           radius=float(cfg["data"]["failure_radius"]))


def main() -> None:
    """Train the teacher, distill it, and write the day-17 artifacts."""
    started = time.time()
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    data = make_dataset(cfg)
    scalers = fit_scalers(data["train"], cfg)
    model = build_model(cfg)
    # Shape probe before training, on the train scaler.
    model.set_scalers(scalers["drift_mean"], scalers["drift_scale"],
                      scalers["corr_mean"], scalers["corr_scale"])
    probe = torch.tensor([5.0, 20.0, 40.0], dtype=DTYPE)
    probe_t = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE)
    probe_r = torch.tensor([10.0, 4.0, 1.0], dtype=DTYPE)
    with torch.no_grad():
        probe_drift = model.drift(probe, probe_t)
        probe_raw = model.correction_raw(probe, probe_t, probe_r)
    print("probe drift", tuple(probe_drift.shape), "correction", tuple(probe_raw.shape), flush=True)
    training = train_model(model, data, scalers, cfg)
    frames = {name: query_frame(data[name], cfg) for name in ("train", "val", "test")}
    teacher = {name: drift_numpy(model, frames[name]["state"]) for name in frames}
    drift_field, drift_search = select_field(
        frames["train"]["state"], teacher["train"], frames["val"]["state"], teacher["val"], cfg)
    sigma_field, sigma_search = select_field(
        frames["train"]["z"].reshape(-1, 1), frames["train"]["sigma"],
        frames["val"]["z"].reshape(-1, 1), frames["val"]["sigma"], cfg)
    drift_test = drift_pack(lambda state: drift_field.predict(state)[:, 0],
                            frames["test"]["drift"], teacher["test"], frames["test"]["state"])
    drift_val = drift_pack(lambda state: drift_field.predict(state)[:, 0],
                           frames["val"]["drift"], teacher["val"], frames["val"]["state"])
    sigma_test_pred = sigma_field.predict(frames["test"]["z"].reshape(-1, 1))[:, 0]
    sigma_test_nrmse = nrmse(frames["test"]["sigma"], sigma_test_pred)
    low_z = np.linspace(float(cfg["distill"]["low_z"][0]), float(cfg["distill"]["low_z"][1]),
                        int(cfg["distill"]["low_z_n"]))
    low_pred = sigma_field.predict(low_z.reshape(-1, 1))[:, 0]
    low_truth = true_sigma(low_z, float(cfg["data"]["sigma_scale"]))
    degree = int(drift_search["selected"]["degree"])
    threshold = float(drift_search["selected"]["threshold"])
    ridge = float(cfg["distill"]["ridge"])
    oracle_teacher = fit_polynomial_field(frames["train"]["state"], teacher["train"].reshape(-1, 1),
                                          int(cfg["distill"]["oracle_degree"]), 0.0, ridge)
    oracle_truth = fit_polynomial_field(frames["train"]["state"], frames["train"]["drift"].reshape(-1, 1),
                                        int(cfg["distill"]["oracle_degree"]), 0.0, ridge)
    z_lo, z_hi = np.quantile(frames["train"]["z"], cfg["distill"]["grid_z_quantiles"])
    grid_z = np.linspace(float(z_lo), float(z_hi), int(cfg["distill"]["grid_n_z"]))
    grid = product_grid(grid_z, float(cfg["data"]["T"]), int(cfg["distill"]["grid_n_t"]))
    grid_truth = true_drift(grid[:, 0], float(cfg["data"]["kappa"]), float(cfg["data"]["theta"]))
    grid_teacher = drift_numpy(model, grid)
    grid_sym = drift_field.predict(grid)[:, 0]
    shuffle_rng = np.random.default_rng(int(cfg["seeds"]["time_shuffle"]))
    shuffled_time = shuffle_rng.uniform(0.0, float(cfg["data"]["T"]), size=len(frames["test"]["state"]))
    time_teacher = time_sensitivity(lambda state: drift_numpy(model, state), frames["test"]["state"],
                                    shuffled_time)
    time_symbolic = time_sensitivity(lambda state: drift_field.predict(state)[:, 0],
                                     frames["test"]["state"], shuffled_time)
    boot = bootstrap_support(frames["train"]["state"], teacher["train"], frames["train"]["traj_index"],
                             degree, threshold, ridge, int(cfg["distill"]["bootstrap"]),
                             np.random.default_rng(int(cfg["seeds"]["bootstrap"])))
    # One-step Gaussian NLL on test increments. This is not the training ELBO.
    test_left = data["test"]["left"]
    dt = float(cfg["data"]["dt"])
    steps = n_steps(cfg)
    times = np.arange(steps, dtype=np.float64) * dt
    # Reconstruct the terminal state is unnecessary: use consecutive left states.
    # The last increment is not stored. Use steps 0..S-2 -> left[:, 1:].
    z_now = test_left[:, :-1].reshape(-1)
    z_next = test_left[:, 1:].reshape(-1)
    t_now = np.tile(times[:-1], test_left.shape[0])
    increment = z_next - z_now
    sigma_now = true_sigma(z_now, float(cfg["data"]["sigma_scale"]))
    nll_truth = gaussian_nll(increment, true_drift(z_now, float(cfg["data"]["kappa"]),
                                                   float(cfg["data"]["theta"])), sigma_now, dt)
    nll_teacher = gaussian_nll(increment, drift_numpy(model, np.stack([z_now, t_now], axis=1)),
                               sigma_now, dt)
    step_ratio = float(np.mean(np.abs(true_drift(frames["train"]["z"], float(cfg["data"]["kappa"]),
                                                 float(cfg["data"]["theta"]))) * dt
                               / (frames["train"]["sigma"] * math.sqrt(dt))))
    # Simulations.
    def initial(n: int) -> np.ndarray:
        return np.full(n, float(cfg["data"]["z0"]))

    paired_n = int(cfg["data"]["paired_paths"])
    paired_rng = np.random.default_rng(int(cfg["seeds"]["paired"]))
    paired_inc = paired_rng.normal(0.0, 1.0, size=(paired_n, steps))
    sigma_paper = paper_sigma_fn(cfg)
    sigma_poly = polynomial_sigma_fn(sigma_field)
    sims = {}
    for name, kind in (("truth", "truth"), ("teacher", "teacher"), ("symbolic", "symbolic")):
        sims[name] = run_simulation(make_drift_fn(kind, model, drift_field, cfg), sigma_paper,
                                    initial(paired_n), dt, paired_inc, cfg)
    sims["poly"] = run_simulation(make_drift_fn("symbolic", model, drift_field, cfg), sigma_poly,
                                  initial(paired_n), dt, paired_inc, cfg)
    law_n = int(cfg["data"]["law_paths"])
    law = {}
    for name, kind, seed in (
            ("truth", "truth", cfg["seeds"]["law_truth"]),
            ("floor", "truth", cfg["seeds"]["law_floor"]),
            ("teacher", "teacher", cfg["seeds"]["law_teacher"]),
            ("symbolic", "symbolic", cfg["seeds"]["law_symbolic"])):
        inc = np.random.default_rng(int(seed)).normal(0.0, 1.0, size=(law_n, steps))
        law[name] = run_simulation(make_drift_fn(kind, model, drift_field, cfg), sigma_paper,
                                   initial(law_n), dt, inc, cfg)
    # Discretization: fine Brownian pieces summed into the coarse increment.
    fine_factor = int(cfg["hypotheses"]["H7_discretization_gate"]["fine_factor"])
    fine_rng = np.random.default_rng(int(cfg["seeds"]["discretization"]))
    fine_inc = fine_rng.normal(0.0, 1.0, size=(paired_n, steps * fine_factor))
    coarse_inc = fine_inc.reshape(paired_n, steps, fine_factor).sum(axis=-1) / math.sqrt(fine_factor)
    disc_fine = run_simulation(make_drift_fn("teacher", model, drift_field, cfg), sigma_paper,
                               initial(paired_n), dt / fine_factor, fine_inc, cfg)
    disc_coarse = run_simulation(make_drift_fn("teacher", model, drift_field, cfg), sigma_paper,
                                 initial(paired_n), dt, coarse_inc, cfg)
    disc_both = ~(disc_fine["failed"] | disc_coarse["failed"])
    disc_rmse = (rms(disc_fine["states"][disc_both, -1] - disc_coarse["states"][disc_both, -1])
                 if disc_both.any() else float("nan"))
    post = posterior_mean_rmse(model, data["test"], cfg, "network", int(cfg["seeds"]["posterior"]))
    prior = posterior_mean_rmse(model, data["test"], cfg, "zero", int(cfg["seeds"]["prior_rollout"]))
    elbo_test = reference_elbo(model, data["test"], cfg, int(cfg["seeds"]["test_noise"]),
                               "network", "network")
    elbo_truth_prior = reference_elbo(model, data["val"], cfg, int(cfg["seeds"]["val_noise"]),
                                      "truth", "zero")
    elbo_truth_corr = reference_elbo(model, data["val"], cfg, int(cfg["seeds"]["val_noise"]),
                                     "truth", "network")
    expanded_drift = expand_to_original(drift_field)
    expanded_sigma = expand_to_original(sigma_field)
    drift_tex = tex_equations(expanded_drift, ("b",), ("z", "t"))
    sigma_tex = tex_equations(expanded_sigma, (r"\sigma",), ("z",))
    on_rms = drift_test["teacher_truth_rms"]
    grid_rms = rms(grid_teacher - grid_truth)
    metrics = {
        "drift_test": drift_test,
        "drift_val_not_used_for_teacher_selection": drift_val,
        "sigma_test_nrmse": sigma_test_nrmse,
        "sigma_test_rms": rms(sigma_test_pred - frames["test"]["sigma"]),
        "sigma_truth_rms": rms(frames["test"]["sigma"]),
        "sigma_low_z_nrmse": nrmse(low_truth, low_pred),
        "sigma_low_z_rms": rms(low_pred - low_truth),
        "sigma_nonpositive_fraction": float(np.mean(sigma_test_pred <= 0.0)),
        "vif_selected": vif_max(frames["train"]["state"], degree),
        "vif_degree3": vif_max(frames["train"]["state"], int(cfg["distill"]["vif_degree"])),
        "vif_sigma_selected": vif_max(frames["train"]["z"].reshape(-1, 1),
                                      int(sigma_search["selected"]["degree"])),
        "oracle_teacher_nrmse": nrmse(teacher["test"], oracle_teacher.predict(frames["test"]["state"])[:, 0]),
        "oracle_truth_nrmse": nrmse(frames["test"]["drift"],
                                    oracle_truth.predict(frames["test"]["state"])[:, 0]),
        "cubic_rms_ratio": cubic_ratio(teacher["train"], frames["train"]["state"], degree, threshold, ridge),
        "fail_truth": sims["truth"]["failure_rate"],
        "fail_teacher": sims["teacher"]["failure_rate"],
        "fail_symbolic": sims["symbolic"]["failure_rate"],
        "fail_poly": sims["poly"]["failure_rate"],
        "paired_sym_teacher": finite_path_rmse(sims["symbolic"]["states"], sims["teacher"]["states"],
                                               sims["symbolic"]["failed"], sims["teacher"]["failed"]),
        "paired_teacher_truth": finite_path_rmse(sims["teacher"]["states"], sims["truth"]["states"],
                                                 sims["teacher"]["failed"], sims["truth"]["failed"]),
        "paired_poly_teacher": finite_path_rmse(sims["poly"]["states"], sims["teacher"]["states"],
                                                sims["poly"]["failed"], sims["teacher"]["failed"]),
        "var_sym_teacher": rel_var(terminal_var(sims["symbolic"]["states"], sims["symbolic"]["failed"]),
                                   terminal_var(sims["teacher"]["states"], sims["teacher"]["failed"])),
        "var_teacher_truth": rel_var(terminal_var(sims["teacher"]["states"], sims["teacher"]["failed"]),
                                    terminal_var(sims["truth"]["states"], sims["truth"]["failed"])),
        "var_poly_teacher": rel_var(terminal_var(sims["poly"]["states"], sims["poly"]["failed"]),
                                   terminal_var(sims["teacher"]["states"], sims["teacher"]["failed"])),
        "law_fail_truth": law["truth"]["failure_rate"],
        "law_fail_teacher": law["teacher"]["failure_rate"],
        "law_fail_symbolic": law["symbolic"]["failure_rate"],
        "law_mean_teacher": rms(mean_curve(law["teacher"]["states"]) - mean_curve(law["truth"]["states"])),
        "law_mean_symbolic": rms(mean_curve(law["symbolic"]["states"]) - mean_curve(law["truth"]["states"])),
        "floor_mean": rms(mean_curve(law["floor"]["states"]) - mean_curve(law["truth"]["states"])),
        "law_var_teacher": rel_var(terminal_var(law["teacher"]["states"], law["teacher"]["failed"]),
                                  terminal_var(law["truth"]["states"], law["truth"]["failed"])),
        "law_var_symbolic": rel_var(terminal_var(law["symbolic"]["states"], law["symbolic"]["failed"]),
                                   terminal_var(law["truth"]["states"], law["truth"]["failed"])),
        "floor_var": rel_var(terminal_var(law["floor"]["states"], law["floor"]["failed"]),
                            terminal_var(law["truth"]["states"], law["truth"]["failed"])),
        "time_teacher": time_teacher,
        "time_symbolic": time_symbolic,
        "grid_teacher_truth_nrmse": nrmse(grid_truth, grid_teacher),
        "grid_sym_teacher_nrmse": nrmse(grid_teacher, grid_sym),
        "grid_sym_truth_nrmse": nrmse(grid_truth, grid_sym),
        "grid_teacher_truth_rms": grid_rms,
        "grid_sym_teacher_rms": rms(grid_sym - grid_teacher),
        "ontraj_teacher_truth_rms": on_rms,
        "disc_rmse": disc_rmse,
        "disc_retained": float(np.mean(disc_both)),
        "post_rmse": post["rmse"],
        "post_fail": post["failure_rate"],
        "prior_rmse": prior["rmse"],
        "prior_fail": prior["failure_rate"],
        "step_ratio": step_ratio,
        "nll_teacher": float(np.mean(nll_teacher)),
        "nll_truth": float(np.mean(nll_truth)),
    }
    metrics["grid_over_ontraj"] = grid_rms / max(on_rms, 1e-12)
    verdicts = decide(cfg, metrics)
    counts_train = data["train"]["counts"].sum(axis=1)
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "platform": platform.platform(),
        "git_head": git_head(),
        "seconds": time.time() - started,
    }
    equations = {
        "drift_tex": drift_tex,
        "sigma_polynomial_tex": sigma_tex,
        "sigma_paper_tex": r"\sigma(z) = \sqrt{z}",
        "symbolic_sde_tex": (
            "\\begin{aligned}\n"
            "dZ_t &= b(Z_t, t)\\,dt + \\sigma(Z_t)\\,dW_t \\\\\n"
            + drift_tex.replace("\\begin{aligned}\n", "").replace("\\end{aligned}", "")
            + sigma_tex.replace("\\begin{aligned}\n", "").replace("\\end{aligned}", "")
            + "\\end{aligned}"
        ),
    }
    summary = {
        "environment": environment,
        "training": {"best_epoch": training["best_epoch"], "best_val_elbo": training["best_val_elbo"],
                     "final_train_elbo": training["history"][-1]["train_elbo"],
                     "final_val_elbo": training["history"][-1]["val_elbo"],
                     "history": training["history"]},
        "elbo_test_once": elbo_test,
        "elbo_val_truth_drift_zero_correction": elbo_truth_prior,
        "elbo_val_truth_drift_learned_correction": elbo_truth_corr,
        "data": {
            "rejected_train": data["train"]["rejected"],
            "rejected_val": data["val"]["rejected"],
            "rejected_test": data["test"]["rejected"],
            "z_train_min": float(frames["train"]["z"].min()),
            "z_train_max": float(frames["train"]["z"].max()),
            "event_count_mean": float(counts_train.mean()),
            "event_count_var": float(counts_train.var()),
            "index_of_dispersion": float(counts_train.var() / max(counts_train.mean(), 1e-12)),
            "scalers": {key: value.tolist() for key, value in scalers.items()},
        },
        "drift_search": drift_search,
        "sigma_search": sigma_search,
        "bootstrap": boot,
        "failure_classes": {name: class_counts(sims[name]["failure_class"]) for name in sims},
        "metrics": metrics,
        "verdicts": verdicts,
        "equations": equations,
        "probe_shapes": {"drift": list(probe_drift.shape), "correction": list(probe_raw.shape)},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(jsonify(summary), indent=2))
    (OUT_DIR / "teacher_ndm_weights.json").write_text(json.dumps(state_dict_numpy(model)))
    drift_field.save(OUT_DIR / "symbolic_b.json")
    sigma_field.save(OUT_DIR / "symbolic_sigma.json")
    (OUT_DIR / "symbolic_sde.tex").write_text(equations["symbolic_sde_tex"] + "\n")
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": [{str(exp): coef for exp, coef in poly.items()} for poly in expanded_drift],
        "sigma": [{str(exp): coef for exp, coef in poly.items()} for poly in expanded_sigma],
    }, indent=2))
    print(json.dumps(jsonify({"verdicts": verdicts, "metrics_brief": {
        "drift_test": drift_test, "sigma_test_nrmse": sigma_test_nrmse,
        "best_epoch": training["best_epoch"], "best_val_elbo": training["best_val_elbo"],
        "seconds": environment["seconds"]}}), indent=2))


if __name__ == "__main__":
    main()
