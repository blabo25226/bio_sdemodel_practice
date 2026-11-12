"""Day 20: Howard et al. (ICML 2026) control-consistency diffusion bridge.

Pipeline (settings frozen in ``configs/daily_neural_sde_day20.json``):

1. System: the normal cell-differentiation bridge in Appendix E.4. The
   reference drift is the closed-form Wang et al. (2011) field. Diffusion
   is the constant ``sigma = 0.1`` and is not a network.
2. Teacher: a scalar potential network. Its state gradient is the neural
   bridge adjustment ``eta``. The controlled drift is the reference drift,
   plus the Brownian-bridge guide, plus ``sigma * eta`` (equation 16 and
   Appendix B.1). Training uses self-consistency (4.1) with the average
   schedule (equation 49). Paths are simulated under stop-gradient.
3. Distillation: direct queries of ``eta``. The library is powers of
   ``(t, x_1, x_2)`` only. Validation chooses degree and threshold.
   Test is scored once. A second fit replaces the whole drift by the same
   library and is not allowed to choose the primary polynomial.

Run from the repository root:
``python -m src.daily_neural_sde.day20_howard_ccdb``

Outputs land in ``outputs/daily_neural_sde/day20/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day20.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day20"
REPORT_DIR = ROOT / "reports" / "daily_neural_sde" / "day20_howard_ccdb"
DTYPE = torch.float64
HILL_EPS = 2.0 ** -4
VAR_NAMES = ("t", "x_{1}", "x_{2}")
FieldFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-20 config."""
    return json.loads(path.read_text())


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def rms(values: np.ndarray) -> float:
    """Root mean square of every finite entry. Empty input is NaN."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
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


def reference_drift(x: np.ndarray) -> np.ndarray:
    """Wang et al. (2011) cell drift. ``x`` has shape ``(n, 2)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x1, x2 = x[:, 0], x[:, 1]
    b1 = x1 ** 4 / (HILL_EPS + x1 ** 4) + HILL_EPS / (HILL_EPS + x2 ** 4) - x1
    b2 = x2 ** 4 / (HILL_EPS + x2 ** 4) + HILL_EPS / (HILL_EPS + x1 ** 4) - x2
    return np.column_stack([b1, b2])


def reference_drift_torch(x: torch.Tensor) -> torch.Tensor:
    """Torch twin of :func:`reference_drift`. ``x`` has shape ``(n, 2)``."""
    x1, x2 = x[:, 0], x[:, 1]
    b1 = x1 ** 4 / (HILL_EPS + x1 ** 4) + HILL_EPS / (HILL_EPS + x2 ** 4) - x1
    b2 = x2 ** 4 / (HILL_EPS + x2 ** 4) + HILL_EPS / (HILL_EPS + x1 ** 4) - x2
    return torch.stack([b1, b2], dim=1)


def terminal_control(x: np.ndarray, x_terminal: np.ndarray, dt: float, sigma: float,
                     drift: np.ndarray | None = None) -> np.ndarray:
    """Control implied by a deterministic jump onto ``x_terminal``.

    ``x`` has shape ``(n, 2)``. The returned control has the same shape.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if drift is None:
        drift = reference_drift(x)
    jump = (np.asarray(x_terminal, dtype=np.float64) - x) / dt
    return (jump - drift) / (sigma ** 2)


class PotentialNet(nn.Module):
    """Scalar potential with a sinusoidal time embedding.

    The bridge adjustment is the state gradient of this potential, so the
    learned vector field is a gradient in the two state coordinates.
    """

    def __init__(self, width: int, depth: int, n_freq: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.n_freq = int(n_freq)
        n_time = 1 + 2 * self.n_freq
        layers: list[nn.Module] = []
        in_dim = n_time + 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def time_features(self, t: torch.Tensor, horizon: float) -> torch.Tensor:
        """Map times ``(n,)`` to ``(n, 1 + 2 n_freq)``."""
        cols = [t / horizon]
        for freq in range(1, self.n_freq + 1):
            angle = 2.0 * math.pi * freq * t / horizon
            cols.append(torch.sin(angle))
            cols.append(torch.cos(angle))
        return torch.stack(cols, dim=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor, horizon: float) -> torch.Tensor:
        """Potential values. ``t`` is ``(n,)`` and ``x`` is ``(n, 2)``."""
        features = torch.cat([self.time_features(t, horizon), x], dim=1)
        return self.net(features).squeeze(-1)


def eta_value(net: PotentialNet, t: torch.Tensor, x: torch.Tensor, horizon: float) -> torch.Tensor:
    """State gradient of the potential, detached from the parameters.

    ``t`` has shape ``(n,)``, ``x`` has shape ``(n, 2)``, and the gradient
    has shape ``(n, 2)``.
    """
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        psi = net(t.detach(), x_in, horizon)
        grad = torch.autograd.grad(psi.sum(), x_in, create_graph=False)[0]
    return grad.detach()


def eta_train(net: PotentialNet, t: torch.Tensor, x: torch.Tensor, horizon: float) -> torch.Tensor:
    """State gradient that keeps the parameter graph. Shape ``(n, 2)``."""
    x_in = x.detach().requires_grad_(True)
    psi = net(t.detach(), x_in, horizon)
    return torch.autograd.grad(psi.sum(), x_in, create_graph=True)[0]


def vjp_reference(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``(Jacobian of b)^T u``. Both arguments have shape ``(n, 2)``."""
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        value = reference_drift_torch(x_in)
        grad = torch.autograd.grad(value, x_in, grad_outputs=u.detach(), create_graph=False)[0]
    return grad.detach()


def control_from_eta(eta: torch.Tensor, x: torch.Tensor, x_terminal: torch.Tensor,
                     t: float, horizon: float, sigma: float) -> torch.Tensor:
    """Control ``u = (f - b) / sigma^2`` for the parameterized drift."""
    remain = horizon - t
    bridge = (x_terminal - x) / remain
    return bridge / (sigma ** 2) + eta / sigma


def consistency_loss(net: PotentialNet, noise: torch.Tensor, x0: torch.Tensor,
                     x_terminal: torch.Tensor, horizon: float, sigma: float,
                     radius: float) -> tuple[torch.Tensor, int]:
    """One stop-gradient self-consistency loss for property (4.1).

    ``noise`` has shape ``(batch, n_steps - 1, 2)``. The returned loss is a
    scalar tensor, and the integer is the number of paths that stayed finite
    and inside ``radius``.
    """
    batch, n_noise, _ = noise.shape
    n_steps = n_noise + 1
    dt = horizon / n_steps
    x = x0.expand(batch, 2).clone()
    states = [x.clone()]
    alive = torch.ones(batch, dtype=torch.bool)
    for k in range(n_steps - 1):
        t = k * dt
        eta = eta_value(net, torch.full((batch,), t, dtype=DTYPE), x, horizon)
        drift = reference_drift_torch(x) + (x_terminal - x) / (horizon - t) + sigma * eta
        x = x + drift * dt + sigma * math.sqrt(dt) * noise[:, k]
        bad = (~torch.isfinite(x).all(dim=1)) | (torch.linalg.vector_norm(x, dim=1) > radius)
        alive = alive & ~bad
        x = torch.where(alive[:, None], x, x0)
        states.append(x.clone())
    controls: list[torch.Tensor] = []
    for k, state in enumerate(states):
        t = k * dt
        eta = eta_value(net, torch.full((batch,), t, dtype=DTYPE), state, horizon)
        controls.append(control_from_eta(eta, state, x_terminal, t, horizon, sigma))
    jump = (x_terminal - states[-1]) / dt
    targets: list[torch.Tensor | None] = [None] * n_steps
    targets[-1] = (jump - reference_drift_torch(states[-1])) / (sigma ** 2)
    for k in reversed(range(n_steps - 1)):
        s = k * dt
        weight_next = (horizon - (s + dt)) / (horizon - s)
        weight_now = dt / (horizon - s)
        running = dt * vjp_reference(states[k], controls[k])
        nxt = targets[k + 1]
        if nxt is None:
            raise RuntimeError("missing self-consistency target")
        targets[k] = running + weight_now * controls[k + 1] + weight_next * nxt
    pieces = []
    for k, state in enumerate(states):
        t = k * dt
        eta = eta_train(net, torch.full((batch,), t, dtype=DTYPE), state, horizon)
        pred = control_from_eta(eta, state.detach(), x_terminal, t, horizon, sigma)
        target = targets[k]
        if target is None:
            raise RuntimeError("missing self-consistency target")
        pieces.append((pred - target.detach()).pow(2).sum(dim=1))
    stacked = torch.stack(pieces, dim=1)
    mask = alive.detach()
    if int(mask.sum()) == 0:
        zero = torch.zeros((), dtype=DTYPE, requires_grad=True)
        return zero, 0
    return stacked[mask].mean(), int(mask.sum())


def train_teacher(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit the potential. The scalar head starts at zero."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    torch.manual_seed(int(cfg["seeds"]["network"]))
    net = PotentialNet(int(model_cfg["width"]), int(model_cfg["depth"]),
                       int(model_cfg["n_freq"])).to(dtype=DTYPE)
    opt = torch.optim.Adam(net.parameters(), lr=float(model_cfg["lr"]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["seeds"]["network"]))
    horizon = float(data_cfg["horizon"])
    sigma = float(data_cfg["sigma"])
    radius = float(data_cfg["failure_radius"])
    n_steps = int(model_cfg["n_steps"])
    batch = int(model_cfg["batch_size"])
    x0 = torch.tensor(data_cfg["x0"], dtype=DTYPE)
    x_terminal = torch.tensor(data_cfg["x_terminal"], dtype=DTYPE)
    losses: list[float] = []
    alive_counts: list[int] = []
    for step in range(int(model_cfg["train_steps"])):
        noise = torch.randn(batch, n_steps - 1, 2, generator=generator, dtype=DTYPE)
        loss, n_alive = consistency_loss(net, noise, x0, x_terminal, horizon, sigma, radius)
        opt.zero_grad(set_to_none=True)
        if n_alive > 0 and torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(model_cfg["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        else:
            losses.append(float("nan"))
        alive_counts.append(n_alive)
        if step % 20 == 0 or step + 1 == int(model_cfg["train_steps"]):
            print(f"train step={step} loss={losses[-1]:.6g} alive={n_alive}/{batch}", flush=True)
    return {"model": net, "loss": losses, "alive": alive_counts}


def save_weights(net: PotentialNet, path: Path) -> None:
    """Save the potential weights as JSON."""
    payload = {key: value.detach().cpu().numpy().tolist() for key, value in net.state_dict().items()}
    path.write_text(json.dumps(payload, indent=2))


def load_weights(net: PotentialNet, path: Path) -> PotentialNet:
    """Load weights written by :func:`save_weights`."""
    payload = json.loads(path.read_text())
    state = {key: torch.tensor(value, dtype=DTYPE) for key, value in payload.items()}
    net.load_state_dict(state)
    return net


def teacher_eta_np(net: PotentialNet, t: float, x: np.ndarray, horizon: float) -> np.ndarray:
    """Evaluate ``eta`` at one time. ``x`` is ``(n, 2)`` and the result matches it."""
    if len(x) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    times = torch.full((len(x),), float(t), dtype=DTYPE)
    states = torch.tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE)
    return eta_value(net, times, states, horizon).cpu().numpy().astype(np.float64)


def query_eta_field(net: PotentialNet, features: np.ndarray, horizon: float) -> np.ndarray:
    """Evaluate ``eta`` on rows ``(t, x1, x2)``. Result has shape ``(n, 2)``."""
    features = np.asarray(features, dtype=np.float64)
    out = np.zeros((len(features), 2), dtype=np.float64)
    # Group identical times so each autograd call is a batch.
    times = features[:, 0]
    for t in np.unique(np.round(times, decimals=10)):
        mask = np.isclose(times, t)
        out[mask] = teacher_eta_np(net, float(t), features[mask, 1:], horizon)
    return out


def controlled_drift_np(eta: np.ndarray, x: np.ndarray, t: float, x_terminal: np.ndarray,
                        horizon: float, sigma: float) -> np.ndarray:
    """Full controlled drift ``b + bridge + sigma eta``. Shape ``(n, 2)``."""
    remain = horizon - t
    bridge = (np.asarray(x_terminal, dtype=np.float64) - x) / remain
    return reference_drift(x) + bridge + sigma * eta


def simulate_bridge(eta_fn: Callable[[float, np.ndarray], np.ndarray],
                    diffusion_fn: Callable[[float, np.ndarray], np.ndarray],
                    x0: np.ndarray, x_terminal: np.ndarray, horizon: float, sigma_drift: float,
                    n_steps: int, increments: np.ndarray, radius: float,
                    use_structure: bool = True) -> dict[str, Any]:
    """Euler bridge with a deterministic terminal jump.

    ``increments`` has shape ``(n, n_steps - 1, 2)`` and is used on every
    step except the last. Failures are latched. Classes are ``non_finite``,
    ``outside_radius``, and ``nonpositive_diffusion``. Failed rows stay NaN
    and keep their mass in ``failure_rate``.
    """
    increments = np.asarray(increments, dtype=np.float64)
    n = increments.shape[0]
    if increments.shape != (n, n_steps - 1, 2):
        raise ValueError("increments shape mismatch")
    dt = horizon / n_steps
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], n, axis=0)
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    observed = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    for k in range(n_steps - 1):
        t = k * dt
        active = np.flatnonzero(~failed)
        if len(active):
            state = x[active]
            eta = np.asarray(eta_fn(t, state), dtype=np.float64)
            diffusion = np.asarray(diffusion_fn(t, state), dtype=np.float64)
            if diffusion.ndim == 1:
                diffusion = np.repeat(diffusion[:, None], 2, axis=1)
            bad_diff = np.any(diffusion <= 0.0, axis=1) | ~np.isfinite(diffusion).all(axis=1)
            if use_structure:
                drift = controlled_drift_np(eta, state, t, x_terminal, horizon, sigma_drift)
            else:
                drift = eta
            step = drift * dt + diffusion * sqrt_dt * increments[active, k]
            updated = state + step
            bad_value = ~np.isfinite(updated).all(axis=1)
            bad_radius = np.linalg.norm(updated, axis=1) > radius
            newly = bad_diff | bad_value | bad_radius
            for local, idx in enumerate(active):
                if not newly[local]:
                    continue
                if bad_diff[local]:
                    failure_class[idx] = "nonpositive_diffusion"
                elif bad_value[local]:
                    failure_class[idx] = "non_finite"
                else:
                    failure_class[idx] = "outside_radius"
                failed[idx] = True
            x[active[~newly]] = updated[~newly]
        x[failed] = np.nan
        observed.append(x.copy())
    # Deterministic jump. A path that is already dead stays dead.
    active = np.flatnonzero(~failed)
    if len(active):
        x[active] = np.asarray(x_terminal, dtype=np.float64)
    x[failed] = np.nan
    observed.append(x.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "prejump_index": n_steps - 1}


def select_field(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                 cfg: dict[str, Any], name: str) -> tuple[PolynomialField, dict[str, Any]]:
    """Validation NRMSE, then smaller degree, then smaller threshold.

    ``y_train`` and ``y_val`` are one-dimensional.
    """
    ridge = float(cfg["distill"]["ridge"])
    best: PolynomialField | None = None
    best_key: tuple[float, int, float] | None = None
    best_meta: dict[str, Any] = {}
    candidates: list[dict[str, float]] = []
    for degree in cfg["distill"]["degrees"]:
        for threshold in cfg["distill"]["thresholds"]:
            field = fit_polynomial_field(x_train, y_train[:, None], int(degree), float(threshold), ridge)
            score = nrmse(y_val, field.predict(x_val)[:, 0])
            record = {"name": name, "degree": int(degree), "threshold": float(threshold),
                      "score": score, "active": int(np.count_nonzero(field.coefficients))}
            candidates.append(record)
            key = (score, int(degree), float(threshold))
            if best_key is None or key < best_key:
                best_key = key
                best = field
                best_meta = record
    if best is None:
        raise RuntimeError(f"no candidate for {name}")
    best_meta = dict(best_meta)
    best_meta["candidates"] = candidates
    return best, best_meta


def library_spectrum(x: np.ndarray, degree: int) -> dict[str, float]:
    """Numerical rank and condition of the degree-``degree`` library on ``x``."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    theta = evaluate_powers((x - mean) / scale, monomial_powers(x.shape[1], degree))
    singular = np.linalg.svd(theta, compute_uv=False)
    tiny = float(singular[-1])
    return {"rank": float(np.linalg.matrix_rank(theta, tol=1e-8)),
            "condition": float(singular[0] / max(tiny, 1e-30)),
            "terms": float(theta.shape[1])}


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


def component_pack(prediction: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    """NRMSE and residual RMS of one component against the teacher."""
    return {"sym_teacher_nrmse": nrmse(teacher, prediction),
            "sym_teacher_rms": rms(prediction - teacher),
            "teacher_rms": rms(teacher),
            "constant_floor": nrmse(teacher, np.full_like(teacher, np.mean(teacher)))}


def constant_floor_from_train(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """NRMSE of predicting the training mean on the test targets."""
    return nrmse(y_test, np.full_like(y_test, float(np.mean(y_train))))


def relative_rms(reference: np.ndarray, changed: np.ndarray) -> float:
    """``||changed - reference|| / ||reference||`` over finite rows."""
    return float(rms(changed - reference) / max(rms(reference), 1e-8))


def stack_fields(fields: list[PolynomialField], features: np.ndarray) -> np.ndarray:
    """Evaluate one scalar field per column. Result has shape ``(n, k)``."""
    columns = [field.predict(features)[:, 0] for field in fields]
    return np.column_stack(columns)


def finite_difference_curl(fn: FieldFn, features: np.ndarray, eps: float) -> np.ndarray:
    """Curl ``d eta_2 / d x_1 - d eta_1 / d x_2`` by central differences.

    ``features`` is ``(n, 3)`` in the order ``(t, x1, x2)``. ``fn`` returns
    ``(n, 2)``.
    """
    def shifted(axis: int, sign: float) -> np.ndarray:
        points = np.array(features, copy=True)
        points[:, axis] += sign * eps
        return fn(points)

    d2_dx1 = (shifted(1, 1.0)[:, 1] - shifted(1, -1.0)[:, 1]) / (2.0 * eps)
    d1_dx2 = (shifted(2, 1.0)[:, 0] - shifted(2, -1.0)[:, 0]) / (2.0 * eps)
    return d2_dx1 - d1_dx2


def make_grid(cfg: dict[str, Any]) -> np.ndarray:
    """Off-manifold query grid ``(t, x1, x2)``, not used for selection."""
    data = cfg["data"]
    tt, xx, yy = np.meshgrid(np.asarray(data["grid_t"], dtype=np.float64),
                             np.asarray(data["grid_x1"], dtype=np.float64),
                             np.asarray(data["grid_x2"], dtype=np.float64), indexing="ij")
    return np.column_stack([tt.ravel(), xx.ravel(), yy.ravel()])


def rollout_features(net: PotentialNet, cfg: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    """Simulate ``n`` teacher bridges and stack interior states as queries.

    Each finite trajectory contributes ``n_steps`` rows ``(t, x1, x2)``.
    The terminal jumped state is not a query: the bridge guide is undefined
    at ``t = T``.
    """
    data = cfg["data"]
    n_steps = int(cfg["model"]["n_steps"])
    horizon = float(data["horizon"])
    sigma = float(data["sigma"])
    dt = horizon / n_steps
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal((n, n_steps - 1, 2))

    def eta_fn(t: float, x: np.ndarray) -> np.ndarray:
        return teacher_eta_np(net, t, x, horizon)

    def diffusion_fn(t: float, x: np.ndarray) -> np.ndarray:
        del t
        return np.full(len(x), sigma)

    sim = simulate_bridge(eta_fn, diffusion_fn, np.asarray(data["x0"], dtype=np.float64),
                          np.asarray(data["x_terminal"], dtype=np.float64), horizon, sigma,
                          n_steps, increments, float(data["failure_radius"]), True)
    # observed index k is the state at time k * dt for k = 0..n_steps-1.
    # Index n_steps is the jump.
    finite = ~sim["failed"]
    rows = []
    owners = []
    for idx in np.flatnonzero(finite):
        for k in range(n_steps):
            state = sim["states"][idx, k]
            rows.append([k * dt, state[0], state[1]])
            owners.append(int(idx))
    features = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 3))
    return {"features": features, "owners": np.asarray(owners, dtype=int),
            "failure_rate": sim["failure_rate"], "n_finite": int(finite.sum()),
            "states": sim["states"], "failed": sim["failed"]}


def split_owners(owners: np.ndarray, n_train: int, n_val: int, n_test: int) -> dict[str, np.ndarray]:
    """Trajectory-disjoint masks. Owners are in simulation order."""
    unique = []
    seen = set()
    for owner in owners:
        if int(owner) not in seen:
            seen.add(int(owner))
            unique.append(int(owner))
    if len(unique) < n_train + n_val + n_test:
        raise RuntimeError(f"only {len(unique)} finite trajectories")
    train_ids = set(unique[:n_train])
    val_ids = set(unique[n_train:n_train + n_val])
    test_ids = set(unique[n_train + n_val:n_train + n_val + n_test])
    owner_arr = np.asarray(owners)
    return {"train": np.isin(owner_arr, list(train_ids)),
            "val": np.isin(owner_arr, list(val_ids)),
            "test": np.isin(owner_arr, list(test_ids)),
            "n_finite_trajectories": len(unique)}


def predict_structured_drift(eta_hat: np.ndarray, features: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Controlled drift with a polynomial ``eta``. Shape ``(n, 2)``."""
    data = cfg["data"]
    horizon = float(data["horizon"])
    sigma = float(data["sigma"])
    terminal = np.asarray(data["x_terminal"], dtype=np.float64)
    out = np.zeros_like(eta_hat)
    for t in np.unique(np.round(features[:, 0], decimals=10)):
        mask = np.isclose(features[:, 0], t)
        out[mask] = controlled_drift_np(eta_hat[mask], features[mask, 1:], float(t),
                                        terminal, horizon, sigma)
    return out


def path_rmse(left: np.ndarray, right: np.ndarray, left_failed: np.ndarray,
              right_failed: np.ndarray) -> dict[str, float]:
    """RMSE on trajectories that both simulations kept. The retained fraction is explicit."""
    ok = ~(left_failed | right_failed)
    if not ok.any():
        return {"rmse": float("nan"), "retained_fraction": 0.0}
    return {"rmse": float(np.sqrt(np.mean((left[ok] - right[ok]) ** 2))),
            "retained_fraction": float(np.mean(ok))}


def mean_curve(states: np.ndarray, failed: np.ndarray) -> np.ndarray | None:
    """Time-wise mean of surviving trajectories. ``states`` is ``(n, time, d)``."""
    rows = []
    for k in range(states.shape[1]):
        ok = ~failed & np.isfinite(states[:, k, 0])
        if not ok.any():
            return None
        rows.append(states[ok, k].mean(axis=0))
    return np.stack(rows, axis=0)


def mean_path_rmse(left: np.ndarray, right: np.ndarray, left_failed: np.ndarray,
                   right_failed: np.ndarray) -> float:
    """RMSE between two mean curves. NaN if either curve is missing."""
    a = mean_curve(left, left_failed)
    b = mean_curve(right, right_failed)
    if a is None or b is None:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def class_counts(names: list[str | None]) -> dict[str, int]:
    """Count latched failure classes. Survivors are omitted."""
    counts = {"non_finite": 0, "outside_radius": 0, "nonpositive_diffusion": 0}
    for name in names:
        if name in counts:
            counts[name] += 1
    return counts


def bootstrap_frequency(x: np.ndarray, y: np.ndarray, degree: int, threshold: float,
                        ridge: float, draws: int, seed: int) -> list[float]:
    """Selection frequency of each coefficient under row resampling of the training queries."""
    rng = np.random.default_rng(seed)
    powers = monomial_powers(x.shape[1], degree)
    counts = np.zeros(len(powers))
    n = len(x)
    for _ in range(draws):
        pick = rng.integers(0, n, size=n)
        field = fit_polynomial_field(x[pick], y[pick, None], degree, threshold, ridge)
        counts += np.abs(field.coefficients[0]) > 0.0
    return (counts / draws).tolist()


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Failure mass above the gate makes H4 and H5 indeterminate."""
    h1 = cfg["hypotheses"]["H1_eta"]
    h2 = cfg["hypotheses"]["H2_diffusion"]
    h3 = cfg["hypotheses"]["H3_collinearity_gate"]
    h4 = cfg["hypotheses"]["H4_paired_simulation"]
    h5 = cfg["hypotheses"]["H5_law"]
    h6 = cfg["hypotheses"]["H6_off_manifold"]
    h7 = cfg["hypotheses"]["H7_adjustment"]
    h8 = cfg["hypotheses"]["H8_pure_polynomial"]
    h9 = cfg["hypotheses"]["H9_curl"]
    eta_ok = []
    for pack in metrics["eta_test"]:
        beat = pack["sym_teacher_nrmse"] < pack["constant_floor"]
        eta_ok.append(pack["sym_teacher_nrmse"] <= h1["symbolic_vs_teacher_test_nrmse_max"] and beat)
    h4_inside = (metrics["fail_teacher"] <= h4["failure_rate_max"]
                 and metrics["fail_symbolic"] <= h4["failure_rate_max"]
                 and abs(metrics["fail_symbolic"] - metrics["fail_teacher"]) <= h4["failure_gap_max"])
    if h4_inside:
        h4_verdict = _pass(metrics["path_rmse"] <= h4["path_rmse_max"]
                           and metrics["prejump_rmse"] <= h4["prejump_rmse_max"])
    else:
        h4_verdict = "indeterminate"
    h5_inside = (metrics["law_fail_teacher"] <= h5["failure_rate_max"]
                 and metrics["law_fail_symbolic"] <= h5["failure_rate_max"]
                 and metrics["floor_mean"] <= h5["floor_mean_path_rmse_max"])
    if h5_inside:
        h5_verdict = _pass(metrics["law_mean_symbolic"] <= h5["symbolic_mean_path_rmse_max"])
    else:
        h5_verdict = "indeterminate"
    vif = metrics["vif_degree3"]
    h3_verdict = "indeterminate" if vif > h3["vif_max"] else "pass"
    time_gap = abs(metrics["time_teacher"] - metrics["time_symbolic"])
    return {
        "H1": _pass(all(eta_ok)),
        "H1_components": ["pass" if flag else "fail" for flag in eta_ok],
        "H1_matches_ceiling": bool(metrics["matches_ceiling"]),
        "H2": _pass(metrics["diffusion_nrmse"] <= h2["symbolic_vs_teacher_test_nrmse_max"]
                    and metrics["diffusion_nonpositive_fraction"] <= h2["nonpositive_fraction_max"]),
        "H3": h3_verdict,
        "H3_closed_form": False,
        "H4": h4_verdict,
        "H5": h5_verdict,
        "H6": _pass(metrics["grid_nrmse"] <= h6["grid_nrmse_max"]
                    and metrics["time_teacher"] >= h6["time_reversal_teacher_min"]
                    and time_gap <= h6["time_reversal_gap_max"]),
        "H6_amplified": bool(metrics["grid_residual_rms"]
                             > h6["amplification_factor"] * metrics["path_residual_rms"]),
        "H7": _pass(metrics["eta_rms"] >= h7["eta_rms_min"]
                    and metrics["baseline_path_rmse"] >= h7["baseline_path_rmse_min"]),
        "H8": _pass(metrics["pure_nrmse"] >= metrics["structured_nrmse"] + h8["gap_min"]),
        "H9": _pass(metrics["teacher_curl_rms"] <= h9["teacher_curl_rms_max"]
                    and metrics["symbolic_relative_curl"] <= h9["symbolic_relative_curl_max"]),
    }


def reverse_time(features: np.ndarray, horizon: float, dt: float) -> np.ndarray:
    """Send ``t`` to ``T - dt - t`` and keep the state."""
    out = np.array(features, copy=True)
    out[:, 0] = (horizon - dt) - out[:, 0]
    return out


def evaluate(cfg: dict[str, Any], net: PotentialNet, features: np.ndarray, masks: dict[str, np.ndarray],
             fields: dict[str, PolynomialField], pure_fields: list[PolynomialField],
             diffusion_field: PolynomialField, meta: dict[str, Any]) -> dict[str, Any]:
    """One-shot test, off-manifold grid, and paired simulations."""
    data = cfg["data"]
    horizon = float(data["horizon"])
    sigma = float(data["sigma"])
    n_steps = int(cfg["model"]["n_steps"])
    dt = horizon / n_steps
    terminal = np.asarray(data["x_terminal"], dtype=np.float64)
    x0 = np.asarray(data["x0"], dtype=np.float64)
    radius = float(data["failure_radius"])
    test = features[masks["test"]]
    train = features[masks["train"]]
    eta_test = query_eta_field(net, test, horizon)
    eta_train = query_eta_field(net, train, horizon)
    eta_hat = stack_fields([fields["eta1"], fields["eta2"]], test)
    packs = []
    for j in range(2):
        pack = component_pack(eta_hat[:, j], eta_test[:, j])
        pack["constant_floor"] = constant_floor_from_train(eta_train[:, j], eta_test[:, j])
        pack["sym_teacher_rms"] = rms(eta_hat[:, j] - eta_test[:, j])
        packs.append(pack)
    diffusion_hat = diffusion_field.predict(test)[:, 0]
    diffusion_target = np.full(len(test), sigma)
    # Pure polynomial of the full controlled drift, already selected on validation.
    teacher_drift = predict_structured_drift(eta_test, test, cfg)
    structured = predict_structured_drift(eta_hat, test, cfg)
    pure = stack_fields(pure_fields, test)
    grid = make_grid(cfg)
    eta_grid = query_eta_field(net, grid, horizon)
    eta_grid_hat = stack_fields([fields["eta1"], fields["eta2"]], grid)
    reversed_test = reverse_time(test, horizon, dt)
    eta_rev = query_eta_field(net, reversed_test, horizon)
    eta_rev_hat = stack_fields([fields["eta1"], fields["eta2"]], reversed_test)
    early = test[:, 0] <= float(cfg["distill"]["early_time_max"])
    eps = float(cfg["distill"]["curl_eps"])

    def teacher_fn(points: np.ndarray) -> np.ndarray:
        return query_eta_field(net, points, horizon)

    def symbolic_fn(points: np.ndarray) -> np.ndarray:
        return stack_fields([fields["eta1"], fields["eta2"]], points)

    teacher_curl = finite_difference_curl(teacher_fn, grid, eps)
    symbolic_curl = finite_difference_curl(symbolic_fn, grid, eps)
    ceiling = []
    for j in range(2):
        dense = fit_polynomial_field(train, eta_train[:, j:j + 1], 3, 0.0, float(cfg["distill"]["ridge"]))
        ceiling.append(nrmse(eta_test[:, j], dense.predict(test)[:, 0]))
    matches = []
    for j, name in enumerate(("eta1", "eta2")):
        selected = meta[name]
        matches.append(int(selected["degree"]) == 3 and float(selected["threshold"]) == 0.0
                       and abs(packs[j]["sym_teacher_nrmse"] - ceiling[j]) <= 1e-8)

    def run(eta_mode: str, diffusion_mode: str, increments: np.ndarray, structure: bool) -> dict[str, Any]:
        def eta_call(t: float, x: np.ndarray) -> np.ndarray:
            feat = np.column_stack([np.full(len(x), t), x])
            if eta_mode == "teacher":
                return teacher_eta_np(net, t, x, horizon)
            if eta_mode == "symbolic":
                return stack_fields([fields["eta1"], fields["eta2"]], feat)
            if eta_mode == "zero":
                return np.zeros((len(x), 2))
            if eta_mode == "pure":
                return stack_fields(pure_fields, feat)
            raise ValueError(eta_mode)

        def diffusion_call(t: float, x: np.ndarray) -> np.ndarray:
            if diffusion_mode == "teacher":
                return np.full(len(x), sigma)
            feat = np.column_stack([np.full(len(x), float(t)), x])
            return diffusion_field.predict(feat)[:, 0]

        return simulate_bridge(eta_call, diffusion_call, x0, terminal, horizon, sigma,
                               n_steps, increments, radius, structure)

    rng = np.random.default_rng(int(cfg["seeds"]["sim"]))
    paired_noise = rng.standard_normal((int(data["n_test"]), n_steps - 1, 2))
    teacher_sim = run("teacher", "teacher", paired_noise, True)
    symbolic_sim = run("symbolic", "symbolic", paired_noise, True)
    baseline_sim = run("zero", "teacher", paired_noise, True)
    pure_sim = run("pure", "symbolic", paired_noise, False)
    paired = path_rmse(teacher_sim["states"], symbolic_sim["states"],
                       teacher_sim["failed"], symbolic_sim["failed"])
    pre_idx = teacher_sim["prejump_index"]
    pre = path_rmse(teacher_sim["states"][:, pre_idx:pre_idx + 1],
                    symbolic_sim["states"][:, pre_idx:pre_idx + 1],
                    teacher_sim["failed"], symbolic_sim["failed"])
    base = path_rmse(teacher_sim["states"], baseline_sim["states"],
                     teacher_sim["failed"], baseline_sim["failed"])
    law_n = int(data["n_law"])
    law_teacher = run("teacher", "teacher",
                      np.random.default_rng(int(cfg["seeds"]["law_a"])).standard_normal((law_n, n_steps - 1, 2)),
                      True)
    law_floor = run("teacher", "teacher",
                    np.random.default_rng(int(cfg["seeds"]["law_b"])).standard_normal((law_n, n_steps - 1, 2)),
                    True)
    law_symbolic = run("symbolic", "symbolic",
                       np.random.default_rng(int(cfg["seeds"]["law_symbolic"])).standard_normal((law_n, n_steps - 1, 2)),
                       True)
    # Step ratio on the test queries: ||f|| dt / (sigma sqrt(dt)).
    step_ratio = float(np.mean(np.linalg.norm(teacher_drift, axis=1)) * dt / (sigma * math.sqrt(dt)))
    metrics: dict[str, Any] = {
        "eta_test": packs,
        "eta_rms": rms(eta_test),
        "path_residual_rms": rms(eta_hat - eta_test),
        "grid_nrmse": nrmse(eta_grid, eta_grid_hat),
        "grid_residual_rms": rms(eta_grid_hat - eta_grid),
        "diffusion_nrmse": nrmse(diffusion_target, diffusion_hat),
        "diffusion_nonpositive_fraction": float(np.mean(diffusion_hat <= 0.0)),
        "diffusion_mean": float(np.mean(diffusion_hat)),
        "vif_degree3": vif_max(train, int(cfg["distill"]["vif_degree"])),
        "spectrum_degree3": library_spectrum(train, 3),
        "time_teacher": relative_rms(eta_test, eta_rev),
        "time_symbolic": relative_rms(eta_hat, eta_rev_hat),
        "fail_teacher": teacher_sim["failure_rate"],
        "fail_symbolic": symbolic_sim["failure_rate"],
        "fail_baseline": baseline_sim["failure_rate"],
        "fail_pure": pure_sim["failure_rate"],
        "fail_teacher_class": class_counts(teacher_sim["failure_class"]),
        "fail_symbolic_class": class_counts(symbolic_sim["failure_class"]),
        "fail_pure_class": class_counts(pure_sim["failure_class"]),
        "path_rmse": paired["rmse"],
        "path_retained": paired["retained_fraction"],
        "prejump_rmse": pre["rmse"],
        "baseline_path_rmse": base["rmse"],
        "law_fail_teacher": law_teacher["failure_rate"],
        "law_fail_symbolic": law_symbolic["failure_rate"],
        "law_fail_floor": law_floor["failure_rate"],
        "floor_mean": mean_path_rmse(law_teacher["states"], law_floor["states"],
                                     law_teacher["failed"], law_floor["failed"]),
        "law_mean_symbolic": mean_path_rmse(law_teacher["states"], law_symbolic["states"],
                                            law_teacher["failed"], law_symbolic["failed"]),
        "pure_nrmse": nrmse(teacher_drift, pure),
        "structured_nrmse": nrmse(teacher_drift, structured),
        "pure_early_nrmse": nrmse(teacher_drift[early], pure[early]) if early.any() else float("nan"),
        "structured_early_nrmse": nrmse(teacher_drift[early], structured[early]) if early.any() else float("nan"),
        "ceiling_eta": ceiling,
        "matches_ceiling": bool(all(matches)),
        "teacher_curl_rms": rms(teacher_curl),
        "symbolic_curl_rms": rms(symbolic_curl),
        "symbolic_relative_curl": float(rms(symbolic_curl) / max(rms(eta_grid), 1e-8)),
        "step_ratio": step_ratio,
        "correlation_t_x1": float(np.corrcoef(train[:, 0], train[:, 1])[0, 1]),
        "correlation_t_x2": float(np.corrcoef(train[:, 0], train[:, 2])[0, 1]),
    }
    metrics["verdicts"] = decide(cfg, metrics)
    figure = {
        "teacher_states": teacher_sim["states"][:12],
        "symbolic_states": symbolic_sim["states"][:12],
        "baseline_states": baseline_sim["states"][:12],
        "teacher_failed": teacher_sim["failed"][:12],
        "grid_t": grid[:, 0],
        "grid_x": grid[:, 1:],
        "eta_grid": eta_grid,
        "eta_grid_hat": eta_grid_hat,
    }
    return {"metrics": metrics, "figure": figure,
            "expanded": {
                "eta1": expand_to_original(fields["eta1"]),
                "eta2": expand_to_original(fields["eta2"]),
                "diffusion": expand_to_original(diffusion_field),
                "pure1": expand_to_original(pure_fields[0]),
                "pure2": expand_to_original(pure_fields[1]),
            }}


def make_figure(figure: dict[str, Any], path: Path) -> None:
    """Test bridges and a time slice of the neural adjustment. Not used for selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    for idx in range(figure["teacher_states"].shape[0]):
        teacher = figure["teacher_states"][idx]
        symbolic = figure["symbolic_states"][idx]
        if np.isfinite(teacher).all():
            ax.plot(teacher[:, 0], teacher[:, 1], color="C0", alpha=0.35, lw=1.0)
        if np.isfinite(symbolic).all():
            ax.plot(symbolic[:, 0], symbolic[:, 1], color="C1", alpha=0.35, lw=1.0, ls="--")
    ax.scatter([0.1], [-0.1], color="black", s=18, zorder=3)
    ax.scatter([2.0], [-0.1], color="black", marker="x", s=28, zorder=3)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("paired bridges")
    slice_axis = axes[1]
    grid_x = figure["grid_x"]
    mask = np.isclose(figure["grid_t"], 2.0)
    sc = slice_axis.scatter(grid_x[mask, 0], grid_x[mask, 1], c=figure["eta_grid"][mask, 0],
                            cmap="coolwarm", s=36)
    fig.colorbar(sc, ax=slice_axis, fraction=0.046, pad=0.04)
    slice_axis.set_xlabel(r"$x_1$")
    slice_axis.set_ylabel(r"$x_2$")
    slice_axis.set_title(r"teacher $\eta_1$ at $t=2$")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_tex(expanded: dict[str, list[dict[tuple[int, ...], float]]]) -> str:
    """TeX for the structured symbolic bridge and the pure-polynomial ablation."""
    parts = [
        "% Day 20. The neural object is eta. b and the bridge guide are the paper's closed forms.",
        "% They are not terms of the power library. The pure polynomial is the ablation.",
        tex_equations(expanded["eta1"], (r"\hat\eta_1(t,x_1,x_2)",), VAR_NAMES),
        tex_equations(expanded["eta2"], (r"\hat\eta_2(t,x_1,x_2)",), VAR_NAMES),
        tex_equations(expanded["diffusion"], (r"\hat\sigma(t,x_1,x_2)",), VAR_NAMES),
        tex_equations(expanded["pure1"], (r"\hat f^{\mathrm{poly}}_1(t,x_1,x_2)",), VAR_NAMES),
        tex_equations(expanded["pure2"], (r"\hat f^{\mathrm{poly}}_2(t,x_1,x_2)",), VAR_NAMES),
    ]
    return "\n".join(parts)


def main() -> None:
    """Train the bridge adjustment, distill it, and write the one-shot test."""
    started = time.perf_counter()
    cfg = load_config()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if abs(float(cfg["data"]["hill_eps"]) - HILL_EPS) > 1e-15:
        raise RuntimeError("hill epsilon drifted from 2^{-4}")
    trained = train_teacher(cfg)
    net = trained["model"]
    data = cfg["data"]
    total = (int(data["n_train"]) + int(data["n_val"]) + int(data["n_test"])
             + int(data["n_extra_rollouts"]))
    rolled = rollout_features(net, cfg, total, int(cfg["seeds"]["queries"]))
    print(f"query trajectories finite={rolled['n_finite']} failure_rate={rolled['failure_rate']:.4g}",
          flush=True)
    masks = split_owners(rolled["owners"], int(data["n_train"]), int(data["n_val"]), int(data["n_test"]))
    features = rolled["features"]
    x_train, x_val = features[masks["train"]], features[masks["val"]]
    eta_train = query_eta_field(net, x_train, float(data["horizon"]))
    eta_val = query_eta_field(net, x_val, float(data["horizon"]))
    fields: dict[str, PolynomialField] = {}
    meta: dict[str, Any] = {}
    for j, name in enumerate(("eta1", "eta2")):
        fields[name], meta[name] = select_field(x_train, eta_train[:, j], x_val, eta_val[:, j], cfg, name)
        print(f"selected {name} degree={meta[name]['degree']} threshold={meta[name]['threshold']} "
              f"val={meta[name]['score']:.4g} active={meta[name]['active']}", flush=True)
    drift_train = predict_structured_drift(eta_train, x_train, cfg)
    drift_val = predict_structured_drift(eta_val, x_val, cfg)
    pure_fields = []
    for j, name in enumerate(("pure1", "pure2")):
        field, info = select_field(x_train, drift_train[:, j], x_val, drift_val[:, j], cfg, name)
        pure_fields.append(field)
        meta[name] = info
        print(f"selected {name} degree={info['degree']} threshold={info['threshold']} "
              f"val={info['score']:.4g} active={info['active']}", flush=True)
    diffusion_target_train = np.full(len(x_train), float(data["sigma"]))
    diffusion_target_val = np.full(len(x_val), float(data["sigma"]))
    diffusion_field, diffusion_meta = select_field(
        x_train, diffusion_target_train, x_val, diffusion_target_val, cfg, "diffusion")
    meta["diffusion"] = diffusion_meta
    print(f"selected diffusion degree={diffusion_meta['degree']} threshold={diffusion_meta['threshold']} "
          f"val={diffusion_meta['score']:.4g}", flush=True)
    frequencies = {}
    for j, name in enumerate(("eta1", "eta2")):
        frequencies[name] = bootstrap_frequency(
            x_train, eta_train[:, j], int(meta[name]["degree"]), float(meta[name]["threshold"]),
            float(cfg["distill"]["ridge"]), int(cfg["distill"]["bootstrap"]),
            int(cfg["seeds"]["bootstrap"]) + j)
    outcome = evaluate(cfg, net, features, masks, fields, pure_fields, diffusion_field, meta)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_weights(net, OUT_DIR / "teacher_ccdb_weights.json")
    for name, field in fields.items():
        field.save(OUT_DIR / f"symbolic_{name}.json")
    diffusion_field.save(OUT_DIR / "symbolic_diffusion.json")
    for j, field in enumerate(pure_fields):
        field.save(OUT_DIR / f"symbolic_pure{j + 1}.json")
    tex = render_tex(outcome["expanded"])
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (REPORT_DIR / "symbolic_sde.tex").write_text(tex)
    expanded_dump = {}
    for name, polys in outcome["expanded"].items():
        expanded_dump[name] = [{str(key): value for key, value in poly.items()} for poly in polys]
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps(expanded_dump, indent=2))
    make_figure(outcome["figure"], REPORT_DIR / "drift_grid.png")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "git_head": head,
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "query_failure_rate": rolled["failure_rate"],
        "n_finite_trajectories": masks["n_finite_trajectories"],
        "n_train_rows": int(masks["train"].sum()),
        "n_val_rows": int(masks["val"].sum()),
        "n_test_rows": int(masks["test"].sum()),
        "train_loss_first": trained["loss"][0],
        "train_loss_last": trained["loss"][-1],
        "train_alive_min": int(min(trained["alive"])),
        "selection": {name: {key: value for key, value in info.items() if key != "candidates"}
                      for name, info in meta.items()},
        "selection_candidates": {name: info["candidates"] for name, info in meta.items()},
        "bootstrap_frequency": frequencies,
        "metrics": jsonify(outcome["metrics"]),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    print(json.dumps(jsonify(outcome["metrics"]["verdicts"]), indent=2))
    print(f"wrote {OUT_DIR / 'results.json'} in {payload['seconds']:.1f}s")


if __name__ == "__main__":
    main()
