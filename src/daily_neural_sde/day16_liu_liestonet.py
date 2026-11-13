"""Day 16: Liu et al. (ICML 2026) LieStoNet -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day16.json``):

1. Data: Example 4 of Table 2. The Ito SDE is
   ``dx = (a1/x) dt + dW1``, ``dy = a2 dt + dW2`` with ``a1 = a2 = 1``.
   The closed form is the data law. It is not a gene-regulatory equation.
2. Teacher: the paper's stage-1 surrogate. A swish MLP drift and a swish
   Cholesky MLP are trained by the Euler-Gauss increment likelihood plus
   the notebook's JVP, variance, and whitening penalties (Appendix P).
3. Distillation: direct teacher queries, trajectory-disjoint splits, power
   library in ``(x, y)`` only, validation-only selection, test once.
4. The analytic scaling generator is a probe of the learned drift. Stage-2
   generator training is outside the cut.
5. Simulation keeps failed-trajectory mass and classifies non-finite states,
   nonpositive Cholesky diagonals, crossings of the x floor, and escapes.

Run from the repository root:
``python -m src.daily_neural_sde.day16_liu_liestonet``

Outputs land in ``outputs/daily_neural_sde/day16/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day16.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day16"
DTYPE = torch.float64

DriftFn = Callable[[np.ndarray], np.ndarray]
CholFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-16 config."""
    return json.loads(path.read_text())


def n_steps(cfg: dict[str, Any], dt: float | None = None) -> int:
    """Number of Euler intervals. ``T`` must be a multiple of ``dt``."""
    t = float(cfg["data"]["T"])
    step = float(cfg["data"]["dt"] if dt is None else dt)
    steps = int(round(t / step))
    if abs(steps * step - t) > 1e-9:
        raise ValueError("T is not a multiple of dt")
    return steps


def true_drift(x: np.ndarray, a1: float = 1.0, a2: float = 1.0) -> np.ndarray:
    """Example-4 drift, shape (n, 2).

    ``f = (a1 / x, a2)``. ``x`` must stay nonzero.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.stack([a1 / x[:, 0], np.full(len(x), a2)], axis=1)


def true_chol(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Constant diffusion matrix, shape (n, 2, 2).

    The Cholesky factor of ``sigma**2 I`` is ``sigma I``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    eye = np.zeros((len(x), 2, 2), dtype=np.float64)
    eye[:, 0, 0] = sigma
    eye[:, 1, 1] = sigma
    return eye


def chol_parts(chol: np.ndarray) -> np.ndarray:
    """Lower-triangular entries ``(l11, l21, l22)``, shape (n, 3)."""
    chol = np.asarray(chol, dtype=np.float64)
    return np.stack([chol[:, 0, 0], chol[:, 1, 0], chol[:, 1, 1]], axis=1)


def parts_to_chol(parts: np.ndarray) -> np.ndarray:
    """Assemble ``(l11, l21, l22)`` into a lower-triangular factor, shape (n, 2, 2)."""
    parts = np.atleast_2d(np.asarray(parts, dtype=np.float64))
    chol = np.zeros((len(parts), 2, 2), dtype=np.float64)
    chol[:, 0, 0] = parts[:, 0]
    chol[:, 1, 0] = parts[:, 1]
    chol[:, 1, 1] = parts[:, 2]
    return chol


def covariance_from_chol(chol: np.ndarray) -> np.ndarray:
    """``Sigma = L L^T``, shape (n, 2, 2)."""
    chol = np.asarray(chol, dtype=np.float64)
    return chol @ np.swapaxes(chol, -1, -2)


def ito_x_squared_drift(x: np.ndarray, a1: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Ito drift of ``x**2``.

    ``d(x**2) = (2 a1 + sigma**2) dt + 2 x sigma dW`` when ``dx = (a1/x) dt + sigma dW``.
    Shape (n,).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.full(len(x), 2.0 * a1 + sigma ** 2)


def central_jacobian(field: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                     eps: float) -> np.ndarray:
    """Central-difference Jacobian, shape (n, outputs, d).

    ``field`` maps (n, d) to (n, outputs).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    columns = []
    for j in range(x.shape[1]):
        step = np.zeros_like(x)
        step[:, j] = eps
        columns.append((field(x + step) - field(x - step)) / (2.0 * eps))
    return np.stack(columns, axis=-1)


def scaling_residual(drift: np.ndarray, jacobian: np.ndarray, x: np.ndarray,
                     t: np.ndarray, a2: float) -> np.ndarray:
    """Drift determining residual of ``v3 = 2t d_t + x d_x + (y + a2 t) d_y``.

    ``drift`` is (n, 2) and ``jacobian`` is (n, 2, 2), with the last axis the
    state derivative. The second derivatives of this generator vanish, so the
    diffusion term of the determining equation is absent. Returns (n, 2).
    """
    drift = np.asarray(drift, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    xi_x = x[:, 0]
    xi_y = x[:, 1] + a2 * t
    directional = xi_x[:, None] * jacobian[:, :, 0] + xi_y[:, None] * jacobian[:, :, 1]
    residual = -drift - directional
    residual[:, 1] = residual[:, 1] + a2
    return residual


def translation_y_residual(jacobian: np.ndarray) -> np.ndarray:
    """Drift residual of ``v2 = d_y``, which is ``-df/dy``. Shape (n, 2)."""
    return -np.asarray(jacobian, dtype=np.float64)[:, :, 1]


def mean_abs(values: np.ndarray) -> float:
    """Mean absolute value of every entry."""
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


def gaussian_nll_with_constant(increment: np.ndarray, drift: np.ndarray,
                               chol: np.ndarray, dt: float) -> np.ndarray:
    """Full Euler-Gauss NLL per transition, shape (n,).

    Covariance of the increment is ``L L^T dt``. The ``log(2 pi)`` term is kept.
    """
    increment = np.asarray(increment, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    chol = np.asarray(chol, dtype=np.float64)
    residual = increment - drift * dt
    scale = math.sqrt(dt)
    solved = np.linalg.solve(chol * scale, residual)
    quad = np.sum(solved ** 2, axis=1)
    logdet = 2.0 * np.sum(np.log(np.clip(np.diagonal(chol, axis1=1, axis2=2) * scale, 1e-12, None)),
                          axis=1)
    return 0.5 * (quad + logdet + 2.0 * math.log(2.0 * math.pi))


def notebook_nll(increment: np.ndarray, drift: np.ndarray, chol: np.ndarray,
                 dt: float) -> np.ndarray:
    """Notebook NLL: the full Gaussian NLL without ``log(2 pi)`` per component.

    Shape (n,). This is the training objective's likelihood term.
    """
    full = gaussian_nll_with_constant(increment, drift, chol, dt)
    return full - math.log(2.0 * math.pi)


class MLP(nn.Module):
    """Swish MLP. ``depth`` counts hidden layers, matching Appendix P.

    ``dims = [in] + [hidden] * depth + [out]``. Swish is applied on every
    layer except the last. Input and output shapes are (n, in) and (n, out).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * int(depth) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (n, in) to (n, out)."""
        hidden = x
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < last:
                hidden = torch.nn.functional.silu(hidden)
        return hidden


class Surrogate(nn.Module):
    """Stage-1 neural SDE: drift MLP and lower-triangular Cholesky MLP.

    Both networks read the standardized state ``(x, y)``. Diffusion diagonals
    are ``softplus + sigma_floor``. The factor ``L`` has shape (n, 2, 2) and
    the instantaneous covariance is ``L L^T``.
    """

    def __init__(self, hidden: int, depth: int, sigma_floor: float) -> None:
        super().__init__()
        self.sigma_floor = float(sigma_floor)
        self.drift = MLP(2, 2, hidden, depth)
        self.chol = MLP(2, 3, hidden, depth)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return drift (n, 2) and L (n, 2, 2) at standardized states."""
        drift = self.drift(z)
        raw = self.chol(z)
        l11 = torch.nn.functional.softplus(raw[:, 0]) + self.sigma_floor
        l21 = raw[:, 1]
        l22 = torch.nn.functional.softplus(raw[:, 2]) + self.sigma_floor
        zeros = torch.zeros_like(l11)
        factor = torch.stack([
            torch.stack([l11, zeros], dim=-1),
            torch.stack([l21, l22], dim=-1),
        ], dim=-2)
        return drift, factor


def glorot_(module: nn.Module) -> None:
    """Uniform Glorot weights and zero biases, as in the released notebook."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(layer.weight, -limit, limit)
            nn.init.zeros_(layer.bias)


def build_model(cfg: dict[str, Any]) -> Surrogate:
    """Float64 surrogate with Glorot initialization."""
    model = Surrogate(int(cfg["model"]["hidden"]), int(cfg["model"]["depth"]),
                      float(cfg["model"]["sigma_floor"])).double()
    glorot_(model)
    return model


def softplus_numpy(raw: np.ndarray) -> np.ndarray:
    """Stable softplus, matching ``torch.nn.functional.softplus``."""
    raw = np.asarray(raw, dtype=np.float64)
    return np.logaddexp(0.0, raw)


def mlp_numpy(weights: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """Replay one MLP. ``weights`` are ``(W, b)`` with ``W`` shaped (out, in)."""
    hidden = np.asarray(x, dtype=np.float64)
    last = len(weights) - 1
    for index, (weight, bias) in enumerate(weights):
        hidden = hidden @ weight.T + bias
        if index < last:
            hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
    return hidden


def numpy_forward(state: dict[str, np.ndarray], x: np.ndarray, mean: np.ndarray,
                  scale: np.ndarray, sigma_floor: float,
                  depth: int) -> tuple[np.ndarray, np.ndarray]:
    """NumPy replay of :class:`Surrogate`. Returns drift (n, 2) and L (n, 2, 2)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    z = (x - mean) / scale
    drift_layers = _paired_layers(state, "drift", depth)
    chol_layers = _paired_layers(state, "chol", depth)
    drift = mlp_numpy(drift_layers, z)
    raw = mlp_numpy(chol_layers, z)
    l11 = softplus_numpy(raw[:, 0]) + sigma_floor
    l21 = raw[:, 1]
    l22 = softplus_numpy(raw[:, 2]) + sigma_floor
    return drift, parts_to_chol(np.stack([l11, l21, l22], axis=1))


def _paired_layers(state: dict[str, np.ndarray], prefix: str,
                   depth: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect ``(weight, bias)`` pairs. ``depth`` hidden layers means ``depth + 1`` linears."""
    n_linear = depth + 1
    layers = []
    for index in range(n_linear):
        layers.append((state[f"{prefix}.layers.{index}.weight"],
                       state[f"{prefix}.layers.{index}.bias"]))
    return layers


def state_dict_numpy(model: Surrogate) -> dict[str, np.ndarray]:
    """Detach a float64 state dict into NumPy arrays."""
    return {key: value.detach().cpu().numpy().astype(np.float64)
            for key, value in model.state_dict().items()}


class FrozenTeacher:
    """Neural drift and Cholesky factor in original coordinates.

    ``mean`` and ``scale`` have shape (2,) and were fit on training states.
    ``predict`` returns drift (n, 2), L (n, 2, 2), and parts (n, 3).
    """

    def __init__(self, model: Surrogate, mean: np.ndarray, scale: np.ndarray) -> None:
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self._mean = torch.as_tensor(self.mean, dtype=DTYPE)
        self._scale = torch.as_tensor(self.scale, dtype=DTYPE)
        self.model.eval()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Original-coordinate fields at states ``x`` of shape (n, 2)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = torch.as_tensor((x - self.mean) / self.scale, dtype=DTYPE)
        with torch.no_grad():
            drift, factor = self.model(z)
        drift_np = drift.detach().cpu().numpy()
        factor_np = factor.detach().cpu().numpy()
        return drift_np, factor_np, chol_parts(factor_np)


def _simulate_truth_path(x0: np.ndarray, increments: np.ndarray, dt: float,
                         a1: float, a2: float, sigma: float, x_floor: float,
                         radius: float) -> tuple[np.ndarray, bool]:
    """One analytic path. Returns states (steps+1, 2) and a failure flag.

    On failure the states after the hitting time are NaN. The flag is true
    when any state is non-finite, below the floor, or outside the ball.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    states = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    failed = (not np.isfinite(x).all()) or x[0] < x_floor or float(np.linalg.norm(x)) > radius
    for step in range(increments.shape[0]):
        if failed:
            states.append(np.full(2, np.nan))
            continue
        drift = np.array([a1 / x[0], a2])
        x = x + drift * dt + sigma * sqrt_dt * increments[step]
        failed = (not np.isfinite(x).all()) or x[0] < x_floor or float(np.linalg.norm(x)) > radius
        states.append(np.full(2, np.nan) if failed else x.copy())
    return np.stack(states, axis=0), failed


def make_dataset(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Trajectory-disjoint surviving paths, generated in train, val, test order.

    A path is rejected and redrawn when it hits the floor, leaves the ball, or
    becomes non-finite. Rejection counts are returned. The analytic SDE is not
    clamped.
    """
    data = cfg["data"]
    dt = float(data["dt"])
    steps = n_steps(cfg)
    a1, a2, sigma = float(data["a1"]), float(data["a2"]), float(data["sigma"])
    floor = float(data["x_floor"])
    radius = float(data["failure_radius"])
    cap = int(data["max_resample"])
    out: dict[str, Any] = {}
    for name, count in (("train", int(data["n_train"])),
                        ("val", int(data["n_val"])),
                        ("test", int(data["n_test"]))):
        paths = []
        attempts = 0
        rejected = 0
        while len(paths) < count:
            if attempts > count * cap:
                raise RuntimeError(f"too many rejected {name} trajectories")
            attempts += 1
            x0 = np.array([
                rng.uniform(float(data["x0_low"]), float(data["x0_high"])),
                rng.uniform(float(data["y0_low"]), float(data["y0_high"])),
            ])
            increments = rng.normal(size=(steps, 2))
            states, failed = _simulate_truth_path(
                x0, increments, dt, a1, a2, sigma, floor, radius)
            if failed:
                rejected += 1
                continue
            paths.append(states)
        out[name] = {"states": np.stack(paths, axis=0), "attempts": attempts,
                     "rejected": rejected}
    return out


def transitions_from_paths(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nonterminal states and increments. Shapes (n_traj * steps, 2)."""
    current = states[:, :-1, :].reshape(-1, states.shape[-1])
    nxt = states[:, 1:, :].reshape(-1, states.shape[-1])
    return current, nxt - current


def fit_standardizer(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of finite nonterminal states, shape (2,) each."""
    current, _increment = transitions_from_paths(states)
    current = current[np.isfinite(current).all(axis=1)]
    mean = current.mean(axis=0)
    scale = current.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def _mvn_nll_torch(increment: torch.Tensor, drift: torch.Tensor, factor: torch.Tensor,
                   dt: float) -> torch.Tensor:
    """Notebook NLL per row, shape (n,). No ``log(2 pi)`` term."""
    residual = increment - drift * dt
    scaled = factor * math.sqrt(dt)
    solved = torch.linalg.solve_triangular(scaled, residual.unsqueeze(-1), upper=False).squeeze(-1)
    quad = torch.sum(solved * solved, dim=-1)
    diag = torch.diagonal(scaled, dim1=-2, dim2=-1).clamp_min(1e-12)
    logdet = 2.0 * torch.sum(torch.log(diag), dim=-1)
    return 0.5 * (quad + logdet)


def _chol_factor(raw: torch.Tensor, sigma_floor: float) -> torch.Tensor:
    """Map three raw outputs to a lower-triangular factor, shape (n, 2, 2)."""
    l11 = torch.nn.functional.softplus(raw[:, 0]) + sigma_floor
    l21 = raw[:, 1]
    l22 = torch.nn.functional.softplus(raw[:, 2]) + sigma_floor
    zeros = torch.zeros_like(l11)
    return torch.stack([
        torch.stack([l11, zeros], dim=-1),
        torch.stack([l21, l22], dim=-1),
    ], dim=-2)


def _penalties(drift_net: nn.Module, chol_net: nn.Module, z: torch.Tensor,
               increment: torch.Tensor, dt: float, sigma_floor: float,
               smooth_seed: int, smooth: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """NLL plus JVP smoothness, diagonal variance, and whitening.

    ``drift_net`` and ``chol_net`` are called separately so the caller can
    detach one of them. The JVP smoothness penalty is the derivative of the
    Cholesky entries with respect to the standardized state. It is evaluated
    on diffusion steps, which are the steps that can change ``L``. On a drift
    step the same penalty does not depend on the drift parameters.
    """
    drift = drift_net(z)
    factor = _chol_factor(chol_net(z), sigma_floor)
    nll = _mvn_nll_torch(increment, drift, factor, dt).mean()
    if smooth:
        def factor_entries(point: torch.Tensor) -> torch.Tensor:
            made = _chol_factor(chol_net(point), sigma_floor)
            return torch.stack([made[:, 0, 0], made[:, 1, 0], made[:, 1, 1]], dim=-1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(smooth_seed))
        direction = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        direction = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-12)
        _value, derivative = torch.func.jvp(factor_entries, (z,), (direction,))
        smooth_value = torch.mean(derivative * derivative)
    else:
        smooth_value = torch.zeros((), dtype=z.dtype)
    diag = torch.stack([factor[:, 0, 0], factor[:, 1, 1]], dim=-1)
    variance = torch.mean(torch.var(diag, dim=0, unbiased=False))
    solved = torch.linalg.solve_triangular(
        factor * math.sqrt(dt), (increment - drift * dt).unsqueeze(-1), upper=False).squeeze(-1)
    centered = solved - solved.mean(dim=0, keepdim=True)
    count = max(int(solved.shape[0]) - 1, 1)
    cov = centered.T @ centered / count
    eye = torch.eye(2, dtype=z.dtype)
    whiten = torch.sum(solved.mean(dim=0) ** 2) + torch.sum((cov - eye) ** 2)
    return nll, {"nll": nll, "smooth": smooth_value, "var": variance, "whiten": whiten}


def train_surrogate(model: Surrogate, train_x: np.ndarray, train_dy: np.ndarray,
                    val_x: np.ndarray, val_dy: np.ndarray, mean: np.ndarray,
                    scale: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Alternate drift and diffusion AdamW steps. Restore the best validation NLL.

    The validation score is the notebook NLL only. Regularizers and field error
    are logged and do not select the checkpoint.
    """
    model_cfg = cfg["model"]
    dt = float(cfg["data"]["dt"])
    floor = float(model_cfg["sigma_floor"])
    x_t = torch.as_tensor((train_x - mean) / scale, dtype=DTYPE)
    dy_t = torch.as_tensor(train_dy, dtype=DTYPE)
    opt_f = torch.optim.AdamW(model.drift.parameters(), lr=float(model_cfg["lr_f"]),
                              weight_decay=float(model_cfg["weight_decay"]))
    opt_l = torch.optim.AdamW(model.chol.parameters(), lr=float(model_cfg["lr_L"]),
                              weight_decay=float(model_cfg["weight_decay"]))
    best_nll = math.inf
    best_state = state_dict_numpy(model)
    best_step = 0
    history: list[dict[str, float]] = []
    batch = int(model_cfg["batch_size"])
    n_rows = len(train_x)
    generator = np.random.default_rng(int(cfg["seed"]) + 1)
    outer = int(model_cfg["outer_steps"])
    clip = float(model_cfg["grad_clip"])
    weights = (float(model_cfg["smooth_weight"]), float(model_cfg["var_weight"]),
               float(model_cfg["whiten_weight"]))

    def one_step(active: str, index: np.ndarray, seed_draw: int) -> dict[str, float]:
        z = x_t[index]
        increment = dy_t[index]
        if active == "drift":
            opt_f.zero_grad(set_to_none=True)
            drift_call = model.drift
            chol_call = lambda point: model.chol(point).detach()
            optimizer = opt_f
            parameters = list(model.drift.parameters())
            do_smooth = False
        else:
            opt_l.zero_grad(set_to_none=True)
            drift_call = lambda point: model.drift(point).detach()
            chol_call = model.chol
            optimizer = opt_l
            parameters = list(model.chol.parameters())
            do_smooth = True
        nll, parts = _penalties(drift_call, chol_call, z, increment, dt, floor,
                                seed_draw, do_smooth)
        total = (nll + weights[0] * parts["smooth"] + weights[1] * parts["var"]
                 + weights[2] * parts["whiten"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        return {key: float(value.detach()) for key, value in parts.items()}

    for step in range(1, outer + 1):
        take = min(batch, n_rows)
        choice = generator.choice(n_rows, size=take, replace=False)
        last: dict[str, float] = {}
        for inner in range(int(model_cfg["drift_steps_per_iter"])):
            last = one_step("drift", choice, int(cfg["seed"]) + 10000 * step + inner)
        for inner in range(int(model_cfg["diff_steps_per_iter"])):
            last = one_step("diff", choice, int(cfg["seed"]) + 10000 * step + 100 + inner)
        if step % int(model_cfg["eval_every"]) == 0 or step == 1 or step == outer:
            score = validation_nll(model, val_x, val_dy, mean, scale, dt)
            history.append({"step": step, "val_nll": score, **last})
            if score < best_nll:
                best_nll = score
                best_state = state_dict_numpy(model)
                best_step = step
            print(f"step {step} val_nll {score:.6f} train_nll {last['nll']:.6f}", flush=True)
    model.load_state_dict({key: torch.as_tensor(value, dtype=DTYPE)
                           for key, value in best_state.items()})
    return {"best_step": best_step, "best_val_nll": best_nll, "history_tail": history[-8:]}


def validation_nll(model: Surrogate, x: np.ndarray, increment: np.ndarray,
                   mean: np.ndarray, scale: np.ndarray, dt: float) -> float:
    """Mean notebook NLL. No regularizers."""
    model.eval()
    z = torch.as_tensor((x - mean) / scale, dtype=DTYPE)
    dy = torch.as_tensor(increment, dtype=DTYPE)
    with torch.no_grad():
        drift, factor = model(z)
        score = float(_mvn_nll_torch(dy, drift, factor, dt).mean())
    model.train()
    return score


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


def diffusion_selection_score(teacher_parts: np.ndarray, symbolic_parts: np.ndarray) -> float:
    """Validation score for the Cholesky factor.

    NRMSE on the two positive diagonals, plus the off-diagonal RMS scaled by
    the teacher diagonal RMS. The off-diagonal truth is zero, so its NRMSE is
    not the selection score.
    """
    diag = 0.5 * (nrmse(teacher_parts[:, 0], symbolic_parts[:, 0])
                  + nrmse(teacher_parts[:, 2], symbolic_parts[:, 2]))
    scale = max(float(np.sqrt(np.mean(teacher_parts[:, 0] ** 2))), 1e-6)
    off = float(np.sqrt(np.mean((symbolic_parts[:, 1] - teacher_parts[:, 1]) ** 2))) / scale
    return float(diag + off)


def select_field(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, cfg: dict[str, Any],
                 score_fn: Callable[[np.ndarray, np.ndarray], float]) -> tuple[PolynomialField, dict[str, Any]]:
    """STLSQ over degree and threshold. ``score_fn(truth, prediction)`` is minimized on validation."""
    best: PolynomialField | None = None
    best_key: tuple[float, int, float] | None = None
    best_meta: dict[str, Any] = {}
    candidates = []
    ridge = float(cfg["distill"]["ridge"])
    for degree in cfg["distill"]["degrees"]:
        for threshold in cfg["distill"]["thresholds"]:
            field = fit_polynomial_field(x_train, y_train, int(degree), float(threshold), ridge)
            prediction = field.predict(x_val)
            score = float(score_fn(y_val, prediction))
            record = {"degree": int(degree), "threshold": float(threshold), "score": score,
                      "active_terms": int(np.count_nonzero(field.coefficients))}
            candidates.append(record)
            key = (score, int(degree), float(threshold))
            if best_key is None or key < best_key:
                best_key = key
                best = field
                best_meta = record
    if best is None:
        raise RuntimeError("no candidate")
    best_meta = dict(best_meta)
    best_meta["candidates"] = candidates
    best_meta["library_size"] = int(best.metadata["library_size"])
    return best, best_meta


def pooled_nrmse_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Pooled NRMSE used to select the drift."""
    return nrmse(truth, prediction)


def percentile_grid(x: np.ndarray, size: int, quantiles: list[float], pad: float,
                    x_min: float) -> np.ndarray:
    """Axis-aligned grid on a padded central box. ``x`` is kept at least ``x_min``."""
    x = np.asarray(x, dtype=np.float64)
    low = np.quantile(x, quantiles[0], axis=0)
    high = np.quantile(x, quantiles[1], axis=0)
    span = high - low
    low = low - pad * span
    high = high + pad * span
    low[0] = max(float(low[0]), x_min)
    if high[0] <= low[0]:
        high[0] = low[0] + 1.0
    axes = [np.linspace(low[j], high[j], size) for j in range(x.shape[1])]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return mesh.reshape(-1, x.shape[1])


def probe_cloud(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed probe of ``(x, y)`` and times for the scaling identity. Not used for selection."""
    distill = cfg["distill"]
    xs = np.linspace(float(distill["probe_x"][0]), float(distill["probe_x"][1]),
                     int(distill["probe_n_x"]))
    ys = np.linspace(float(distill["probe_y"][0]), float(distill["probe_y"][1]),
                     int(distill["probe_n_y"]))
    ts = np.asarray(distill["probe_t"], dtype=np.float64)
    grid_x, grid_y, grid_t = np.meshgrid(xs, ys, ts, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return points, grid_t.ravel()


def near_floor_strip(train_x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """A strip closer to the singularity than the bulk of the training ``x``."""
    distill = cfg["distill"]
    count = int(distill["near_floor_n"])
    xs = np.linspace(float(distill["near_floor_x"][0]), float(distill["near_floor_x"][1]), count)
    y_low, y_high = np.quantile(train_x[:, 1], [0.1, 0.9])
    ys = np.linspace(float(y_low), float(y_high), count)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)


def simulate_matrix(drift_fn: DriftFn, chol_fn: CholFn, x0: np.ndarray, dt: float,
                    increments: np.ndarray, failure_radius: float,
                    x_floor: float) -> dict[str, Any]:
    """Euler-Maruyama with a lower-triangular diffusion factor.

    ``increments`` has shape (n, n_steps, 2) of standard normals. Failures are
    latched and classified as ``non_finite``, ``nonpositive_chol``,
    ``below_floor``, or ``outside_radius``. Failed trajectories stay NaN and
    keep their mass in the failure rate.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    n, d = x.shape
    n_steps_local = increments.shape[1]
    if increments.shape != (n, n_steps_local, d):
        raise ValueError("increments shape mismatch")
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n

    def latch(mask: np.ndarray, label: str) -> None:
        newly = (~failed) & mask
        for index in np.flatnonzero(newly):
            failure_class[index] = label
        failed[newly] = True

    finite = np.isfinite(x).all(axis=1)
    latch(~finite, "non_finite")
    latch(finite & (x[:, 0] < x_floor), "below_floor")
    finite_now = np.isfinite(x).all(axis=1) & ~failed
    norms = np.linalg.norm(np.where(np.isfinite(x), x, 0.0), axis=1)
    latch(finite_now & (norms > failure_radius), "outside_radius")
    x[failed] = np.nan
    observed = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps_local):
        active = np.flatnonzero(~failed)
        if len(active):
            drift = np.asarray(drift_fn(x[active]), dtype=np.float64)
            factor = np.asarray(chol_fn(x[active]), dtype=np.float64)
            nonfinite = ((~np.isfinite(drift).all(axis=1))
                         | (~np.isfinite(factor).all(axis=(1, 2))))
            nonpositive = (~nonfinite) & ((factor[:, 0, 0] <= 0.0) | (factor[:, 1, 1] <= 0.0))
            good = (~nonfinite) & (~nonpositive)
            if nonfinite.any():
                mask = np.zeros(n, dtype=bool)
                mask[active[nonfinite]] = True
                latch(mask, "non_finite")
            if nonpositive.any():
                mask = np.zeros(n, dtype=bool)
                mask[active[nonpositive]] = True
                latch(mask, "nonpositive_chol")
            if good.any():
                noise = sqrt_dt * increments[active[good], step, :]
                update = np.einsum("mij,mj->mi", factor[good], noise)
                x[active[good]] = x[active[good]] + drift[good] * dt + update
        finite = np.isfinite(x).all(axis=1)
        latch(~failed & ~finite, "non_finite")
        finite = np.isfinite(x).all(axis=1)
        latch(~failed & finite & (x[:, 0] < x_floor), "below_floor")
        finite = np.isfinite(x).all(axis=1)
        norms = np.linalg.norm(np.where(np.isfinite(x), x, 0.0), axis=1)
        latch(~failed & finite & (norms > failure_radius), "outside_radius")
        x[failed] = np.nan
        observed.append(x.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class, "failure_rate": float(np.mean(failed))}


def failure_counts(classes: list[str | None]) -> dict[str, int]:
    """Counts of the four pre-registered failure classes."""
    names = ("non_finite", "nonpositive_chol", "below_floor", "outside_radius")
    return {name: sum(item == name for item in classes) for name in names}


def paired_path_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Time-averaged RMSE over trajectories finite in both ensembles."""
    errors = []
    for time_index in range(left.shape[1]):
        ok = np.isfinite(left[:, time_index]).all(axis=1) & np.isfinite(right[:, time_index]).all(axis=1)
        if ok.any():
            errors.append(float(np.mean((left[ok, time_index] - right[ok, time_index]) ** 2)))
    if not errors:
        return math.nan
    return float(math.sqrt(float(np.mean(errors))))


def mean_path(states: np.ndarray) -> np.ndarray:
    """Time-wise nanmean, shape (n_obs, d)."""
    with np.errstate(all="ignore"):
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
    """``||left - right||_F / ||right||_F``."""
    if left is None or right is None:
        return None
    return float(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1e-8))


def monomial_table(field: PolynomialField) -> list[dict[str, float]]:
    """Original-coordinate coefficients per output, keyed by exponent text."""
    tables = []
    for poly in expand_to_original(field):
        tables.append({",".join(str(power) for power in exp): float(coef)
                       for exp, coef in sorted(poly.items())})
    return tables


def degree_component(poly: dict[tuple[int, ...], float], x: np.ndarray,
                     degree: int) -> np.ndarray:
    """Evaluate only the terms of total degree ``degree``. Shape (n,)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.zeros(len(x))
    for exp, coef in poly.items():
        if sum(exp) != degree:
            continue
        term = np.full(len(x), coef)
        for k, power in enumerate(exp):
            if power:
                term = term * x[:, k] ** power
        out += term
    return out


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Thresholds are read from ``cfg`` and not modified."""
    hyp = cfg["hypotheses"]
    h1, h2 = hyp["H1_drift"], hyp["H2_diffusion"]
    h4, h5 = hyp["H4_simulation"], hyp["H5_law"]
    h6, h7 = hyp["H6_scaling"], hyp["H7_discretization_gate"]
    verdicts: dict[str, Any] = {
        "H1": _pass(
            all(value <= h1["symbolic_vs_teacher_test_nrmse_max"]
                for value in metrics["drift_sym_teacher"])
            and all(value <= h1["teacher_vs_truth_test_nrmse_max"]
                    for value in metrics["drift_teacher_truth"])
            and all(value <= h1["symbolic_vs_truth_test_nrmse_max"]
                    for value in metrics["drift_sym_truth"])),
        "H2": _pass(
            all(value <= h2["diag_symbolic_vs_teacher_nrmse_max"]
                for value in metrics["diag_sym_teacher"])
            and all(value <= h2["diag_teacher_vs_truth_nrmse_max"]
                    for value in metrics["diag_teacher_truth"])
            and all(value <= h2["diag_symbolic_vs_truth_nrmse_max"]
                    for value in metrics["diag_sym_truth"])
            and metrics["l21_rms_teacher"] <= h2["l21_rms_teacher_max"]
            and metrics["l21_rms_sym_teacher"] <= h2["l21_rms_symbolic_vs_teacher_max"]
            and metrics["l21_rms_sym_truth"] <= h2["l21_rms_symbolic_vs_truth_max"]),
        "H2c": _pass(metrics["fraction_nonpositive"]
                     <= hyp["H2c_range"]["fraction_symbolic_diagonal_nonpositive_max"]),
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
               and metrics["paired_sym_teacher"] <= h4["paired_path_rmse_symbolic_vs_teacher_max"]
               and metrics["paired_teacher_truth"] <= h4["paired_path_rmse_teacher_vs_truth_max"])
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
    v2 = metrics["v2_teacher"]
    v3_teacher = metrics["v3_teacher"]
    v3_symbolic = metrics["v3_symbolic"]
    v3_oracle = metrics["v3_oracle"]
    if not all(math.isfinite(value) for value in (v2, v3_teacher, v3_symbolic, v3_oracle)):
        verdicts["H6"] = "indeterminate"
        verdicts["H6_reason"] = "missing_residual"
    elif v2 > h6["v2_teacher_mean_abs_max"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "spurious_y_dependence"
    elif v3_teacher > h6["v3_teacher_mean_abs_max"]:
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = "teacher_breaks_scaling"
    elif v3_symbolic > h6["v3_symbolic_mean_abs_max"]:
        relative = abs(v3_symbolic - v3_oracle) / max(v3_oracle, 1e-6)
        verdicts["H6"] = "fail"
        verdicts["H6_reason"] = ("library_breaks_scaling_identity"
                                 if relative <= h6["oracle_relative_gap_max"]
                                 else "symbolic_breaks_scaling")
        verdicts["H6_oracle_relative_gap"] = relative
    else:
        verdicts["H6"] = "pass"
        verdicts["H6_reason"] = "scaling_identity_held"
    verdicts["coefficients_are_closed_form"] = False
    return verdicts


def git_head() -> str:
    """Current commit, or ``unknown`` if git is unavailable."""
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


def step_scales(states: np.ndarray, dt: float, a1: float, a2: float, sigma: float) -> dict[str, float]:
    """Mean ``|f| dt`` and mean ``|sigma| sqrt(dt)`` on finite nonterminal states."""
    current, _increment = transitions_from_paths(states)
    current = current[np.isfinite(current).all(axis=1)]
    drift = true_drift(current, a1, a2)
    diffusion_step = math.sqrt(2.0) * sigma * math.sqrt(dt)
    drift_step = float(np.mean(np.linalg.norm(drift, axis=1)) * dt)
    return {"mean_abs_drift_dt": drift_step,
            "mean_abs_sigma_sqrt_dt": diffusion_step,
            "drift_over_diffusion_step": drift_step / max(diffusion_step, 1e-8)}


def field_bundle(teacher: FrozenTeacher, states: np.ndarray, a1: float, a2: float,
                 sigma: float) -> dict[str, np.ndarray]:
    """Teacher and truth fields on finite nonterminal states."""
    current, increment = transitions_from_paths(states)
    finite = np.isfinite(current).all(axis=1) & np.isfinite(increment).all(axis=1)
    current = current[finite]
    increment = increment[finite]
    drift, factor, parts = teacher.predict(current)
    truth_d = true_drift(current, a1, a2)
    truth_c = true_chol(current, sigma)
    return {"features": current, "increment": increment, "drift": drift, "chol": factor,
            "parts": parts, "drift_truth": truth_d, "parts_truth": chol_parts(truth_c)}


def identity_stats(field: Callable[[np.ndarray], np.ndarray], points: np.ndarray,
                   times: np.ndarray, a2: float, eps: float) -> dict[str, float]:
    """Mean absolute v2 and v3 drift residuals of a callable field."""
    jacobian = central_jacobian(field, points, eps)
    drift = field(points)
    v2 = translation_y_residual(jacobian)
    # Average the per-time clouds by evaluating the residual at each point's time.
    v3 = scaling_residual(drift, jacobian, points, times, a2)
    return {"v2": mean_abs(v2), "v3": mean_abs(v3)}


def discretization_check(drift_fn: DriftFn, chol_fn: CholFn, x0: np.ndarray,
                         cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    """Terminal RMSE of coarse EM versus ``k`` fine steps with the same Wiener path."""
    data = cfg["data"]
    k = int(cfg["hypotheses"]["H7_discretization_gate"]["fine_factor"])
    dt = float(data["dt"])
    steps = n_steps(cfg)
    fine_noise = rng.normal(size=(len(x0), steps, k, 2))
    coarse_noise = fine_noise.sum(axis=2) / math.sqrt(k)
    radius = float(data["failure_radius"])
    floor = float(data["x_floor"])
    coarse = simulate_matrix(drift_fn, chol_fn, x0, dt, coarse_noise, radius, floor)
    fine = simulate_matrix(drift_fn, chol_fn, x0, dt / k,
                           fine_noise.reshape(len(x0), steps * k, 2), radius, floor)
    both = (np.isfinite(coarse["states"][:, -1, :]).all(axis=1)
            & np.isfinite(fine["states"][:, -1, :]).all(axis=1))
    if not both.any():
        return {"disc_rmse": math.nan, "disc_retained": 0.0}
    rmse = float(np.sqrt(np.mean((coarse["states"][both, -1, :]
                                  - fine["states"][both, -1, :]) ** 2)))
    return {"disc_rmse": rmse, "disc_retained": float(np.mean(both))}


def rms_matrix(left: np.ndarray, right: np.ndarray) -> float:
    """Root mean square of the difference of two arrays."""
    gap = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(gap ** 2)))


def main() -> dict[str, Any]:
    """Train the LieStoNet stage-1 surrogate, distill it, and write the outputs."""
    started = time.time()
    cfg = load_config()
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = cfg["data"]
    a1, a2, sigma = float(data["a1"]), float(data["a2"]), float(data["sigma"])
    dt = float(data["dt"])

    dataset = make_dataset(cfg, data_rng)
    print("rejections", {name: dataset[name]["rejected"] for name in ("train", "val", "test")},
          flush=True)
    mean, scale = fit_standardizer(dataset["train"]["states"])
    train_x, train_dy = transitions_from_paths(dataset["train"]["states"])
    val_x, val_dy = transitions_from_paths(dataset["val"]["states"])
    train_ok = np.isfinite(train_x).all(axis=1) & np.isfinite(train_dy).all(axis=1)
    val_ok = np.isfinite(val_x).all(axis=1) & np.isfinite(val_dy).all(axis=1)
    train_x, train_dy = train_x[train_ok], train_dy[train_ok]
    val_x, val_dy = val_x[val_ok], val_dy[val_ok]

    model = build_model(cfg)
    train_info = train_surrogate(model, train_x, train_dy, val_x, val_dy, mean, scale, cfg)
    teacher = FrozenTeacher(model, mean, scale)
    floor_nll = float(np.mean(notebook_nll(
        val_dy, true_drift(val_x, a1, a2), true_chol(val_x, sigma), dt)))
    print("best", train_info["best_step"], train_info["best_val_nll"], "floor", floor_nll, flush=True)

    # Validation field error is recorded and is not used to choose the teacher or the polynomial.
    val_drift, _val_chol, val_parts = teacher.predict(val_x)
    val_field = {
        "drift_nrmse": component_nrmse(true_drift(val_x, a1, a2), val_drift),
        "diag_nrmse": [nrmse(np.full(len(val_x), sigma), val_parts[:, 0]),
                       nrmse(np.full(len(val_x), sigma), val_parts[:, 2])],
        "l21_rms": float(np.sqrt(np.mean(val_parts[:, 1] ** 2))),
    }

    queries = {name: field_bundle(teacher, dataset[name]["states"], a1, a2, sigma)
               for name in ("train", "val", "test")}
    drift_field, drift_meta = select_field(
        queries["train"]["features"], queries["train"]["drift"],
        queries["val"]["features"], queries["val"]["drift"], cfg, pooled_nrmse_score)
    chol_field, chol_meta = select_field(
        queries["train"]["features"], queries["train"]["parts"],
        queries["val"]["features"], queries["val"]["parts"], cfg, diffusion_selection_score)
    # Test is evaluated once, after both selections.
    test = queries["test"]
    drift_hat = drift_field.predict(test["features"])
    parts_hat = chol_field.predict(test["features"])
    drift_numbers = {
        "sym_teacher": component_nrmse(test["drift"], drift_hat),
        "teacher_truth": component_nrmse(test["drift_truth"], test["drift"]),
        "sym_truth": component_nrmse(test["drift_truth"], drift_hat),
        "resid_sym_teacher": residual_rms(test["drift"], drift_hat),
        "resid_teacher_truth": residual_rms(test["drift_truth"], test["drift"]),
        "resid_sym_truth": residual_rms(test["drift_truth"], drift_hat),
        "teacher_rms": column_rms(test["drift"]),
        "truth_rms": column_rms(test["drift_truth"]),
    }
    diag_index = [0, 2]
    diag_numbers = {
        "sym_teacher": [nrmse(test["parts"][:, j], parts_hat[:, j]) for j in diag_index],
        "teacher_truth": [nrmse(test["parts_truth"][:, j], test["parts"][:, j]) for j in diag_index],
        "sym_truth": [nrmse(test["parts_truth"][:, j], parts_hat[:, j]) for j in diag_index],
        "resid_teacher_truth": [float(np.sqrt(np.mean((test["parts"][:, j] - test["parts_truth"][:, j]) ** 2)))
                                for j in diag_index],
    }
    l21_rms_teacher = float(np.sqrt(np.mean(test["parts"][:, 1] ** 2)))
    l21_rms_sym_teacher = float(np.sqrt(np.mean((parts_hat[:, 1] - test["parts"][:, 1]) ** 2)))
    l21_rms_sym_truth = float(np.sqrt(np.mean((parts_hat[:, 1] - test["parts_truth"][:, 1]) ** 2)))
    fraction_nonpositive = float(np.mean((parts_hat[:, 0] <= 0.0) | (parts_hat[:, 2] <= 0.0)))

    oracle_truth = fit_polynomial_field(
        queries["train"]["features"], queries["train"]["drift_truth"],
        int(cfg["distill"]["oracle_degree"]), 0.0, float(cfg["distill"]["ridge"]))
    oracle_teacher = fit_polynomial_field(
        queries["train"]["features"], queries["train"]["drift"],
        int(cfg["distill"]["oracle_degree"]), 0.0, float(cfg["distill"]["ridge"]))
    oracle_chol = fit_polynomial_field(
        queries["train"]["features"], queries["train"]["parts_truth"],
        int(cfg["distill"]["oracle_degree"]), 0.0, float(cfg["distill"]["ridge"]))
    oracle_drift_truth = component_nrmse(test["drift_truth"], oracle_truth.predict(test["features"]))
    oracle_drift_teacher = component_nrmse(test["drift"], oracle_teacher.predict(test["features"]))
    oracle_chol_truth = component_nrmse(test["parts_truth"], oracle_chol.predict(test["features"]))

    points, times = probe_cloud(cfg)
    eps = float(cfg["distill"]["finite_difference_eps"])

    def truth_field(state: np.ndarray) -> np.ndarray:
        return true_drift(state, a1, a2)

    ident_truth = identity_stats(truth_field, points, times, a2, eps)
    ident_teacher = identity_stats(lambda state: teacher.predict(state)[0], points, times, a2, eps)
    ident_symbolic = identity_stats(drift_field.predict, points, times, a2, eps)
    ident_oracle = identity_stats(oracle_truth.predict, points, times, a2, eps)
    if ident_truth["v3"] > 1e-4:
        raise RuntimeError("analytic scaling residual is not numerically zero")

    expanded_drift = expand_to_original(drift_field)
    cubic = np.stack([degree_component(expanded_drift[j], test["features"], 3)
                      for j in range(2)], axis=1)
    full = drift_field.predict(test["features"])
    cubic_ratio = float(np.sqrt(np.mean(cubic[:, 0] ** 2)) / max(float(np.sqrt(np.mean(full[:, 0] ** 2))), 1e-8))

    grid = percentile_grid(queries["train"]["features"], int(cfg["distill"]["grid_size"]),
                           list(cfg["distill"]["grid_quantile"]),
                           float(cfg["distill"]["grid_pad_fraction"]),
                           float(cfg["distill"]["grid_x_min"]))
    grid_truth = true_drift(grid, a1, a2)
    grid_teacher = teacher.predict(grid)[0]
    grid_symbolic = drift_field.predict(grid)
    grid_rms = {"sym_teacher": rms_matrix(grid_teacher, grid_symbolic),
                "teacher_truth": rms_matrix(grid_truth, grid_teacher),
                "sym_truth": rms_matrix(grid_truth, grid_symbolic)}
    strip = near_floor_strip(queries["train"]["features"], cfg)
    strip_rms = {"teacher_truth": rms_matrix(true_drift(strip, a1, a2), teacher.predict(strip)[0]),
                 "sym_truth": rms_matrix(true_drift(strip, a1, a2), drift_field.predict(strip)),
                 "sym_teacher": rms_matrix(teacher.predict(strip)[0], drift_field.predict(strip))}

    radius = float(data["failure_radius"])
    floor = float(data["x_floor"])
    test_x0 = dataset["test"]["states"][:, 0, :]
    pair_rng = np.random.default_rng(seed + 21)
    increments = pair_rng.normal(size=(len(test_x0), n_steps(cfg), 2))

    def truth_drift_fn(state: np.ndarray) -> np.ndarray:
        return true_drift(state, a1, a2)

    def truth_chol_fn(state: np.ndarray) -> np.ndarray:
        return true_chol(state, sigma)

    def teacher_drift_fn(state: np.ndarray) -> np.ndarray:
        return teacher.predict(state)[0]

    def teacher_chol_fn(state: np.ndarray) -> np.ndarray:
        return teacher.predict(state)[1]

    def symbolic_chol_fn(state: np.ndarray) -> np.ndarray:
        return parts_to_chol(chol_field.predict(state))

    sim_truth = simulate_matrix(truth_drift_fn, truth_chol_fn, test_x0, dt, increments, radius, floor)
    sim_teacher = simulate_matrix(teacher_drift_fn, teacher_chol_fn, test_x0, dt, increments, radius, floor)
    sim_symbolic = simulate_matrix(drift_field.predict, symbolic_chol_fn, test_x0, dt, increments, radius, floor)
    cov_truth, _n_truth = terminal_cov(sim_truth["states"])
    cov_teacher, _n_teacher = terminal_cov(sim_teacher["states"])
    cov_symbolic, _n_symbolic = terminal_cov(sim_symbolic["states"])

    law_rng = np.random.default_rng(seed + 31)
    floor_rng_a = np.random.default_rng(seed + 41)
    floor_rng_b = np.random.default_rng(seed + 42)
    law_x0 = np.repeat(np.asarray(data["law_x0"], dtype=np.float64)[None, :],
                       int(data["law_paths"]), axis=0)
    law_noise = law_rng.normal(size=(len(law_x0), n_steps(cfg), 2))
    law_truth = simulate_matrix(truth_drift_fn, truth_chol_fn, law_x0, dt, law_noise, radius, floor)
    law_teacher = simulate_matrix(teacher_drift_fn, teacher_chol_fn, law_x0, dt, law_noise, radius, floor)
    law_symbolic = simulate_matrix(drift_field.predict, symbolic_chol_fn, law_x0, dt, law_noise, radius, floor)
    floor_a = simulate_matrix(truth_drift_fn, truth_chol_fn, law_x0, dt,
                              floor_rng_a.normal(size=law_noise.shape), radius, floor)
    floor_b = simulate_matrix(truth_drift_fn, truth_chol_fn, law_x0, dt,
                              floor_rng_b.normal(size=law_noise.shape), radius, floor)
    law_cov_truth, _ = terminal_cov(law_truth["states"])
    law_cov_teacher, _ = terminal_cov(law_teacher["states"])
    law_cov_symbolic, _ = terminal_cov(law_symbolic["states"])
    floor_cov_a, _ = terminal_cov(floor_a["states"])
    floor_cov_b, _ = terminal_cov(floor_b["states"])

    disc = discretization_check(teacher_drift_fn, teacher_chol_fn, test_x0, cfg,
                                np.random.default_rng(seed + 51))
    train_scales = step_scales(dataset["train"]["states"], dt, a1, a2, sigma)
    test_scales = step_scales(dataset["test"]["states"], dt, a1, a2, sigma)
    on_traj_resid = float(np.sqrt(np.mean((test["drift"] - test["drift_truth"]) ** 2)))

    metrics = {
        "drift_sym_teacher": drift_numbers["sym_teacher"],
        "drift_teacher_truth": drift_numbers["teacher_truth"],
        "drift_sym_truth": drift_numbers["sym_truth"],
        "diag_sym_teacher": diag_numbers["sym_teacher"],
        "diag_teacher_truth": diag_numbers["teacher_truth"],
        "diag_sym_truth": diag_numbers["sym_truth"],
        "l21_rms_teacher": l21_rms_teacher,
        "l21_rms_sym_teacher": l21_rms_sym_teacher,
        "l21_rms_sym_truth": l21_rms_sym_truth,
        "fraction_nonpositive": fraction_nonpositive,
        "vif": vif_max(queries["train"]["features"], int(cfg["distill"]["vif_degree"])),
        "fail_truth": sim_truth["failure_rate"],
        "fail_teacher": sim_teacher["failure_rate"],
        "fail_symbolic": sim_symbolic["failure_rate"],
        "paired_sym_teacher": paired_path_rmse(sim_symbolic["states"], sim_teacher["states"]),
        "paired_teacher_truth": paired_path_rmse(sim_teacher["states"], sim_truth["states"]),
        "cov_sym_teacher": rel_frob(cov_symbolic, cov_teacher),
        "cov_teacher_truth": rel_frob(cov_teacher, cov_truth),
        "disc_rmse": disc["disc_rmse"],
        "disc_retained": disc["disc_retained"],
        "law_fail_truth": law_truth["failure_rate"],
        "law_fail_teacher": law_teacher["failure_rate"],
        "law_fail_symbolic": law_symbolic["failure_rate"],
        "law_mean_teacher": mean_path_rmse(mean_path(law_teacher["states"]), mean_path(law_truth["states"])),
        "law_mean_symbolic": mean_path_rmse(mean_path(law_symbolic["states"]), mean_path(law_truth["states"])),
        "law_cov_teacher": rel_frob(law_cov_teacher, law_cov_truth),
        "law_cov_symbolic": rel_frob(law_cov_symbolic, law_cov_truth),
        "floor_mean": mean_path_rmse(mean_path(floor_a["states"]), mean_path(floor_b["states"])),
        "floor_cov": rel_frob(floor_cov_a, floor_cov_b),
        "v2_teacher": ident_teacher["v2"],
        "v3_teacher": ident_teacher["v3"],
        "v3_symbolic": ident_symbolic["v3"],
        "v3_oracle": ident_oracle["v3"],
    }
    verdicts = decide(cfg, metrics)
    flags = {
        "nll_at_floor": abs(train_info["best_val_nll"] - floor_nll)
        <= float(cfg["interpretation"]["nll_floor_gap_max"]),
        "nll_gap": abs(train_info["best_val_nll"] - floor_nll),
        "drift_step_ratio_blind": test_scales["drift_over_diffusion_step"]
        < float(cfg["interpretation"]["step_ratio_drift_blind"]),
        "oracle_f1_above_closed_form": oracle_drift_truth[0]
        > float(cfg["interpretation"]["oracle_closed_form_nrmse_max"]),
        "cubic_ratio_f1": cubic_ratio,
        "cubic_ratio_small": cubic_ratio < float(cfg["interpretation"]["cubic_rms_ratio_small"]),
        "grid_over_ontraj": grid_rms["teacher_truth"] / max(on_traj_resid, 1e-8),
        "coefficients_are_closed_form": False,
    }
    symbolic_tex = tex_equations(expanded_drift, ("f_{x}", "f_{y}"), ("x", "y"), digits=4)
    symbolic_tex += "\n" + tex_equations(expand_to_original(chol_field),
                                        (r"\ell_{11}", r"\ell_{21}", r"\ell_{22}"),
                                        ("x", "y"), digits=4)
    probe = torch.zeros(3, 2, dtype=DTYPE)
    with torch.no_grad():
        probe_drift, probe_factor = model((probe - teacher._mean) / teacher._scale)
    replay_drift, replay_factor = numpy_forward(
        state_dict_numpy(model), probe.numpy(), mean, scale, float(cfg["model"]["sigma_floor"]),
        int(cfg["model"]["depth"]))
    shape_log = {
        "query": list(probe.shape),
        "drift": list(probe_drift.shape),
        "chol": list(probe_factor.shape),
        "numpy_max_abs_drift": float(np.max(np.abs(probe_drift.numpy() - replay_drift))),
        "numpy_max_abs_chol": float(np.max(np.abs(probe_factor.numpy() - replay_factor))),
    }
    if shape_log["numpy_max_abs_drift"] > 1e-8 or shape_log["numpy_max_abs_chol"] > 1e-8:
        raise RuntimeError("NumPy replay does not match the torch surrogate")
    results = {
        "paper": cfg["paper"],
        "seed": seed,
        "deviations_locked_in_config": True,
        "rejections": {name: {"attempts": dataset[name]["attempts"],
                              "rejected": dataset[name]["rejected"]}
                       for name in ("train", "val", "test")},
        "training": train_info,
        "val_field_not_used_for_selection": val_field,
        "nll_floor_val": floor_nll,
        "drift_selection": drift_meta,
        "chol_selection": chol_meta,
        "drift": drift_numbers,
        "diffusion_diag": diag_numbers,
        "l21_rms_teacher": l21_rms_teacher,
        "l21_rms_sym_teacher": l21_rms_sym_teacher,
        "l21_rms_sym_truth": l21_rms_sym_truth,
        "fraction_nonpositive": fraction_nonpositive,
        "oracle_drift_vs_truth": oracle_drift_truth,
        "oracle_drift_vs_teacher": oracle_drift_teacher,
        "oracle_chol_vs_truth": oracle_chol_truth,
        "identity": {"truth": ident_truth, "teacher": ident_teacher,
                     "symbolic": ident_symbolic, "oracle_truth": ident_oracle},
        "grid_rms": grid_rms,
        "near_floor_rms": strip_rms,
        "on_trajectory_teacher_truth_rms": on_traj_resid,
        "vif": metrics["vif"],
        "simulation": {
            "failure_rate": {"truth": sim_truth["failure_rate"],
                             "teacher": sim_teacher["failure_rate"],
                             "symbolic": sim_symbolic["failure_rate"]},
            "paired_sym_teacher": metrics["paired_sym_teacher"],
            "paired_teacher_truth": metrics["paired_teacher_truth"],
            "cov_sym_teacher": metrics["cov_sym_teacher"],
            "cov_teacher_truth": metrics["cov_teacher_truth"],
        },
        "law": {
            "mean_teacher": metrics["law_mean_teacher"],
            "mean_symbolic": metrics["law_mean_symbolic"],
            "cov_teacher": metrics["law_cov_teacher"],
            "cov_symbolic": metrics["law_cov_symbolic"],
            "floor_mean": metrics["floor_mean"],
            "floor_cov": metrics["floor_cov"],
            "failure_rate": {"truth": law_truth["failure_rate"],
                             "teacher": law_teacher["failure_rate"],
                             "symbolic": law_symbolic["failure_rate"]},
        },
        "discretization": disc,
        "step_scales_train": train_scales,
        "step_scales_test": test_scales,
        "flags": flags,
        "failure_classes": {
            "truth": failure_counts(sim_truth["failure_class"]),
            "teacher": failure_counts(sim_teacher["failure_class"]),
            "symbolic": failure_counts(sim_symbolic["failure_class"]),
        },
        "verdicts": verdicts,
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
        "metrics_for_decide": metrics,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    (OUT_DIR / "symbolic_sde.tex").write_text(symbolic_tex + "\n")
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": monomial_table(drift_field),
        "chol": monomial_table(chol_field),
    }, indent=2))
    drift_field.save(OUT_DIR / "symbolic_f.json")
    chol_field.save(OUT_DIR / "symbolic_L.json")
    payload = {"sigma_floor": float(cfg["model"]["sigma_floor"]),
               "hidden": int(cfg["model"]["hidden"]),
               "depth": int(cfg["model"]["depth"]),
               "mean": mean.tolist(), "scale": scale.tolist(),
               "state_dict": {key: value.tolist() for key, value in state_dict_numpy(model).items()}}
    (OUT_DIR / "teacher_liestonet_weights.json").write_text(json.dumps(payload))
    print(json.dumps(jsonify({
        "verdicts": verdicts,
        "drift_teacher_truth": drift_numbers["teacher_truth"],
        "drift_sym_teacher": drift_numbers["sym_teacher"],
        "v3": {"teacher": ident_teacher["v3"], "symbolic": ident_symbolic["v3"],
               "oracle": ident_oracle["v3"]},
        "runtime": results["runtime_seconds"],
    }), indent=2))
    return results


if __name__ == "__main__":
    main()
