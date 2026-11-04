"""Day 11: Kong, Sun & Zhang (2020) SDE-Net -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day11.json``):

1. Data: a 2D four-blob XOR mixture. Each sample is one initial condition
   and therefore one trajectory. There is no closed-form ground-truth SDE;
   the neural SDE is the model, not an estimator of a known physical SDE.
2. Teacher: the paper's SDE
   ``dX_t = f(X_t, t) dt + g(X_0) dW_t`` with a tanh drift net, a
   sigmoid diffusion net of the initial state only, and a linear readout.
   Training alternates cross-entropy on one Euler-Maruyama path with
   ``mean g(ID) - mean g(OOD)``.
3. Distillation: direct teacher queries, trajectory-disjoint splits,
   power library only, validation-only selection, test once. Diffusion
   selection balances ID and OOD validation NRMSE. A degree-3 OLS ceiling
   is diagnostic only.
4. Simulation keeps failed-trajectory mass and separates an excursion
   radius from the failure radius. ID/OOD diffusion levels and terminal
   variance are the paper-axis checks; variance is gated by the failure
   rate.

Run from the repository root:
``python -m src.daily_neural_sde.day11_kong_sdenet``

Outputs land in ``outputs/daily_neural_sde/day11/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import (PolynomialField, evaluate_powers, expand_to_original,
                     fit_polynomial_field, monomial_powers, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day11.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day11"


# ---------------------------------------------------------------- data

def sample_xor(rng: np.random.Generator, n: int, sigma: float,
               centers: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balanced draws from the four XOR modes.

    Returns states ``(n, 2)`` and integer labels ``(n,)``. ``n`` must be
    divisible by the number of modes.
    """
    n_modes = len(centers)
    if n % n_modes != 0:
        raise ValueError("n must be divisible by the number of modes")
    per = n // n_modes
    xs = []
    ys = []
    for center, label in zip(centers, labels):
        xs.append(np.asarray(center, dtype=np.float64) + sigma * rng.standard_normal((per, 2)))
        ys.append(np.full(per, int(label), dtype=int))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    order = rng.permutation(n)
    return x[order], y[order]


def make_split(cfg: dict) -> dict[str, Any]:
    """Trajectory-disjoint ID samples. One x0 is one trajectory."""
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    centers = np.asarray(cfg["data"]["centers"], dtype=np.float64)
    labels = np.asarray(cfg["data"]["center_labels"], dtype=int)
    sigma = float(cfg["data"]["sigma"])
    out: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        x, y = sample_xor(rng, int(cfg["split"][name]), sigma, centers, labels)
        out[name] = {"x0": x, "y": y}
    return out


def perturb(rng: np.random.Generator, x0: np.ndarray, variance: float) -> np.ndarray:
    """Additive Gaussian perturbation, the paper's pseudo-OOD construction."""
    return x0 + np.sqrt(variance) * rng.standard_normal(x0.shape)


# ---------------------------------------------------------------- simulation

def simulate_sdenet(drift_fn: Callable[[np.ndarray, float], np.ndarray],
                    diffusion_fn: Callable[[np.ndarray], np.ndarray],
                    x0: np.ndarray, dt: float, increments: np.ndarray,
                    failure_radius: float, excursion_radius: float) -> dict[str, Any]:
    """Euler-Maruyama for ``dX = f(X, t) dt + g(X_0) dW``.

    ``drift_fn(x, t)`` maps ``x`` of shape ``(n, 2)`` and a scalar time to
    drift ``(n, 2)``. ``diffusion_fn(x0)`` maps the initial state ``(n, 2)``
    to a scalar diffusion ``(n,)`` and is called once; that value is held
    for the whole path (Kong et al., eq. 7). ``increments`` has shape
    ``(n, n_steps, 2)`` and is already the Wiener increment (the update adds
    ``g[:, None] * increments``).

    Failures latch and are classified ``non_finite`` or ``outside_radius``.
    Failed trajectories stay in the counts and are NaN after failure.
    An excursion is ``||X|| > excursion_radius`` on a step that has not yet
    failed; it is not itself a failure.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x0 must have shape (n, 2)")
    n = len(x)
    if increments.ndim != 3 or increments.shape[0] != n or increments.shape[2] != 2:
        raise ValueError("increments must have shape (n, n_steps, 2)")
    # Copy: a diffusion that returns a view of x0 must not change when the
    # state is later overwritten. g is held at the initial state.
    g0 = np.array(np.asarray(diffusion_fn(x), dtype=np.float64).reshape(-1), copy=True)
    if g0.shape != (n,):
        raise ValueError("diffusion_fn must return shape (n,)")
    failed = np.zeros(n, dtype=bool)
    excursion = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    n_steps = increments.shape[1]
    for step in range(n_steps):
        active = ~failed
        if active.any():
            drift = np.asarray(drift_fn(x[active], step * dt), dtype=np.float64)
            if drift.shape != (int(active.sum()), 2):
                raise ValueError("drift_fn must return shape (n_active, 2)")
            x[active] = x[active] + drift * dt + g0[active, None] * increments[active, step]
        finite = np.isfinite(x).all(axis=1)
        # Norm of non-finite rows is NaN; treat those as outside only via the
        # non-finite class, and measure excursions on finite rows.
        norm = np.linalg.norm(np.where(finite[:, None], x, 0.0), axis=1)
        excursion = excursion | (~failed & finite & (norm > excursion_radius))
        bad_nonfinite = ~finite
        bad_radius = finite & (norm > failure_radius)
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        x[failed] = np.nan
    return {"final": x, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion": excursion,
            "excursion_rate": float(np.mean(excursion))}


def aggregate_increments(fine: np.ndarray, block: int) -> np.ndarray:
    """Sum Wiener increments over contiguous blocks. Shape (n, steps, d)."""
    if fine.shape[1] % block != 0:
        raise ValueError("fine step count is not divisible by block")
    n, steps, d = fine.shape
    return fine.reshape(n, steps // block, block, d).sum(axis=2)


# ---------------------------------------------------------------- teacher

def _build_modules(width: int, sigma_max: float):
    """Construct the drift net, diffusion net, and linear head (torch)."""
    import torch
    import torch.nn as nn

    class Drift(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(3, width)
            self.fc2 = nn.Linear(width, width)
            self.fc3 = nn.Linear(width, 2)

        def forward(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
            # x: (n, 2), t: (n,) -> drift (n, 2)
            h = torch.tanh(self.fc1(torch.cat([x, t[:, None]], dim=1)))
            h = torch.tanh(self.fc2(h))
            return self.fc3(h)

    class Diffusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(2, width)
            self.fc2 = nn.Linear(width, width)
            self.fc3 = nn.Linear(width, 1)

        def forward(self, x0: "torch.Tensor") -> "torch.Tensor":
            # x0: (n, 2) -> scalar diffusion (n,)
            h = torch.tanh(self.fc1(x0))
            h = torch.tanh(self.fc2(h))
            return sigma_max * torch.sigmoid(self.fc3(h).squeeze(-1))

    return Drift(), Diffusion(), nn.Linear(2, 2)


def _state_dict_lists(module) -> dict[str, list]:
    return {k: v.detach().cpu().numpy().astype(np.float64).tolist()
            for k, v in module.state_dict().items()}


def train_teacher(cfg: dict, train_x: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    """Alternate the paper's task update and diffusion update.

    The task loss is cross-entropy of the linear head on one
    Euler-Maruyama terminal. ``g`` is stop-gradient inside that path, so
    the task loss trains the drift and the head only. The diffusion loss
    is ``mean g(ID) - mean g(OOD)`` and does not run through the path.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(cfg["seed"]))
    width = 64
    sigma_max = float(cfg["teacher"]["sigma_max"])
    drift, diffusion, head = _build_modules(width, sigma_max)
    opt_f = torch.optim.Adam(list(drift.parameters()) + list(head.parameters()), lr=1e-3)
    opt_g = torch.optim.Adam(diffusion.parameters(), lr=1e-2)
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    n = len(train_x)
    batch = int(cfg["teacher"]["batch_size"])
    steps = int(cfg["teacher"]["n_opt_steps"])
    n_em = int(cfg["teacher"]["n_steps_train"])
    dt = float(cfg["teacher"]["T"]) / n_em
    var = float(cfg["data"]["ood_noise_variance"])
    sqrt_dt = float(np.sqrt(dt))
    history: list[dict[str, float]] = []
    x_t = torch.tensor(train_x, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        xb = x_t[idx]
        yb = y_t[idx]
        noise = torch.tensor(rng.standard_normal((batch, n_em, 2)).astype(np.float32)
                             * sqrt_dt)
        x = xb
        g = diffusion(xb).detach()
        for k in range(n_em):
            t = torch.full((batch,), k * dt)
            x = x + drift(x, t) * dt + g[:, None] * noise[:, k]
        loss_f = F.cross_entropy(head(x), yb)
        opt_f.zero_grad()
        loss_f.backward()
        opt_f.step()

        ood = torch.tensor(perturb(rng, train_x[idx], var).astype(np.float32))
        loss_g = diffusion(xb).mean() - diffusion(ood).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if step % 100 == 0 or step == steps - 1:
            row = {"step": step, "loss_f": float(loss_f.item()),
                   "loss_g": float(loss_g.item()),
                   "g_id": float(diffusion(xb).mean().item()),
                   "g_ood": float(diffusion(ood).mean().item())}
            history.append(row)
            if step % 500 == 0:
                print("train", row, flush=True)
    drift.eval()
    diffusion.eval()
    head.eval()
    return {"drift": _state_dict_lists(drift), "diffusion": _state_dict_lists(diffusion),
            "head_weight": head.weight.detach().cpu().numpy().astype(np.float64).tolist(),
            "head_bias": head.bias.detach().cpu().numpy().astype(np.float64).tolist(),
            "width": width, "sigma_max": sigma_max, "history": history}


_CACHE: dict[str, Any] = {}


def _load_teacher(weights: dict):
    """Cache torch modules for the given weight object."""
    import torch

    if _CACHE.get("id") is not id(weights):
        drift, diffusion, head = _build_modules(int(weights["width"]), float(weights["sigma_max"]))
        drift.load_state_dict({k: torch.tensor(np.asarray(v), dtype=torch.float32)
                               for k, v in weights["drift"].items()})
        diffusion.load_state_dict({k: torch.tensor(np.asarray(v), dtype=torch.float32)
                                   for k, v in weights["diffusion"].items()})
        with torch.no_grad():
            head.weight.copy_(torch.tensor(np.asarray(weights["head_weight"]), dtype=torch.float32))
            head.bias.copy_(torch.tensor(np.asarray(weights["head_bias"]), dtype=torch.float32))
        drift.eval()
        diffusion.eval()
        head.eval()
        _CACHE["id"] = id(weights)
        _CACHE["drift"] = drift
        _CACHE["diffusion"] = diffusion
        _CACHE["head"] = head
    return _CACHE["drift"], _CACHE["diffusion"], _CACHE["head"]


def teacher_drift(weights: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate f(x, t). ``x`` is (n, 2), ``t`` is (n,) -> (n, 2)."""
    import torch

    drift, _, _ = _load_teacher(weights)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    with torch.no_grad():
        out = drift(torch.tensor(x, dtype=torch.float32),
                    torch.tensor(t, dtype=torch.float32))
    return out.numpy().astype(np.float64)


def teacher_g(weights: dict, x0: np.ndarray) -> np.ndarray:
    """Evaluate g(x0). ``x0`` is (n, 2) -> (n,)."""
    import torch

    _, diffusion, _ = _load_teacher(weights)
    x0 = np.atleast_2d(np.asarray(x0, dtype=np.float64))
    with torch.no_grad():
        out = diffusion(torch.tensor(x0, dtype=torch.float32))
    return out.numpy().astype(np.float64)


def teacher_logits(weights: dict, x: np.ndarray) -> np.ndarray:
    """Linear readout. ``x`` is (n, 2) -> logits (n, 2)."""
    w = np.asarray(weights["head_weight"], dtype=np.float64)
    b = np.asarray(weights["head_bias"], dtype=np.float64)
    return np.asarray(x, dtype=np.float64) @ w.T + b


# ---------------------------------------------------------------- queries

def rollout_states(weights: dict, x0: np.ndarray, dt: float, n_steps: int,
                   increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Record (state, time) at the start of each EM step.

    Returns ``states`` of shape ``(n, n_steps, 2)`` and ``times`` of shape
    ``(n, n_steps)``. Diffusion is g(x0), held constant.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    g0 = np.array(teacher_g(weights, x), dtype=np.float64, copy=True)
    states = np.empty((len(x), n_steps, 2), dtype=np.float64)
    times = np.empty((len(x), n_steps), dtype=np.float64)
    for k in range(n_steps):
        states[:, k] = x
        times[:, k] = k * dt
        f = teacher_drift(weights, x, times[:, k])
        x = x + f * dt + g0[:, None] * increments[:, k]
    return states, times


def drift_queries(weights: dict, x0: np.ndarray, dt: float, n_steps: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """On-path (x, y, t) rows and teacher drift rows, finite rows only."""
    increments = rng.standard_normal((len(x0), n_steps, 2)) * np.sqrt(dt)
    states, times = rollout_states(weights, x0, dt, n_steps, increments)
    flat_x = np.concatenate([states.reshape(-1, 2), times.reshape(-1, 1)], axis=1)
    # Drift at the recorded states, re-queried so the target is the field
    # and not a finite difference.
    flat_t = times.reshape(-1)
    target = teacher_drift(weights, states.reshape(-1, 2), flat_t)
    ok = np.isfinite(flat_x).all(axis=1) & np.isfinite(target).all(axis=1)
    return flat_x[ok], target[ok]


# ---------------------------------------------------------------- SINDy

def vif_max(x: np.ndarray, degree: int) -> float:
    """Max VIF of the degree-``degree`` library, constant column excluded."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    theta = evaluate_powers((x - mean) / scale, monomial_powers(x.shape[1], degree))
    powers = monomial_powers(x.shape[1], degree)
    keep = [j for j, p in enumerate(powers) if p.sum() > 0]
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


def select_field(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray,
                 y_va: np.ndarray, degrees: list[int], thresholds: list[float],
                 ridge: float) -> tuple[PolynomialField, dict[str, Any]]:
    """Pick degree and threshold by pooled validation NRMSE."""
    best: PolynomialField | None = None
    best_val = np.inf
    best_meta: dict[str, Any] = {}
    for degree in degrees:
        for thr in thresholds:
            field = fit_polynomial_field(x_tr, y_tr, degree, thr, ridge)
            val = nrmse(y_va, field.predict(x_va))
            if val < best_val:
                best_val = val
                best = field
                best_meta = {"degree": degree, "threshold": thr,
                             "validation_nrmse": val,
                             "active_terms": int(np.count_nonzero(field.coefficients))}
    assert best is not None
    return best, best_meta


def select_field_balanced(x_tr: np.ndarray, y_tr: np.ndarray,
                          x_va: np.ndarray, y_va: np.ndarray,
                          regime_va: np.ndarray, degrees: list[int],
                          thresholds: list[float], ridge: float
                          ) -> tuple[PolynomialField, dict[str, Any]]:
    """Pick degree and threshold by the mean of ID and OOD validation NRMSE.

    ``regime_va`` is 0 on ID rows and 1 on OOD rows.
    """
    best: PolynomialField | None = None
    best_val = np.inf
    best_meta: dict[str, Any] = {}
    id_rows = regime_va == 0
    ood_rows = regime_va == 1
    for degree in degrees:
        for thr in thresholds:
            field = fit_polynomial_field(x_tr, y_tr, degree, thr, ridge)
            pred = field.predict(x_va)
            id_score = nrmse(y_va[id_rows], pred[id_rows])
            ood_score = nrmse(y_va[ood_rows], pred[ood_rows])
            score = 0.5 * (id_score + ood_score)
            if score < best_val:
                best_val = score
                best = field
                best_meta = {"degree": degree, "threshold": thr,
                             "validation_mean_regime_nrmse": score,
                             "validation_id_nrmse": id_score,
                             "validation_ood_nrmse": ood_score,
                             "active_terms": int(np.count_nonzero(field.coefficients))}
    assert best is not None
    return best, best_meta


def fit_ols(x: np.ndarray, y: np.ndarray, degree: int) -> PolynomialField:
    """Dense degree-``degree`` least squares in standardized coordinates.

    The column standardization uses ``x`` only (the training queries).
    This is the library ceiling, not a selected model.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    theta = evaluate_powers((x - mean) / scale, monomial_powers(x.shape[1], degree))
    coef = np.zeros((y.shape[1], theta.shape[1]))
    for j in range(y.shape[1]):
        sol, *_ = np.linalg.lstsq(theta, y[:, j], rcond=None)
        coef[j] = sol
    return PolynomialField(monomial_powers(x.shape[1], degree), coef, mean, scale,
                           {"degree": degree, "fit": "ols", "library_size": int(theta.shape[1])})


def linear_x0_accuracy(x_tr: np.ndarray, y_tr: np.ndarray,
                       x_te: np.ndarray, y_te: np.ndarray) -> float:
    """Least-squares linear classifier fit on train x0, accuracy on test."""
    design = np.concatenate([x_tr, np.ones((len(x_tr), 1))], axis=1)
    target = y_tr.astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = np.concatenate([x_te, np.ones((len(x_te), 1))], axis=1) @ coef
    return float(np.mean((pred >= 0.5).astype(int) == y_te))


def retained_accuracy(logits: np.ndarray, labels: np.ndarray,
                      failed: np.ndarray) -> float:
    """Accuracy with failed rows counted incorrect (mass retained)."""
    pred = np.argmax(logits, axis=1)
    correct = (pred == labels) & ~failed
    return float(np.mean(correct))


def terminal_variance(drift_fn, diffusion_fn, x0: np.ndarray, dt: float,
                      increments: np.ndarray, failure_radius: float) -> dict[str, float]:
    """Across-path variance of the terminal state.

    ``increments`` has shape ``(n_paths, n, n_steps, 2)``. A sample counts
    only if every path stays finite and inside the failure radius; otherwise
    it is a failure. The variance is the mean over surviving samples and
    over the two coordinates of the across-path variance.
    """
    n_paths, n, _, _ = increments.shape
    finals = []
    failed_any = np.zeros(n, dtype=bool)
    for p in range(n_paths):
        sim = simulate_sdenet(drift_fn, diffusion_fn, x0, dt, increments[p],
                              failure_radius, failure_radius)
        finals.append(sim["final"])
        failed_any = failed_any | sim["failed"]
    stack = np.stack(finals, axis=0)  # (paths, n, 2)
    keep = ~failed_any
    if not keep.any():
        return {"failure_rate": 1.0, "variance": float("nan"), "n_kept": 0}
    var = np.var(stack[:, keep, :], axis=0).mean()
    return {"failure_rate": float(np.mean(failed_any)),
            "variance": float(var), "n_kept": int(keep.sum())}


def _pass(ok: bool) -> str:
    return "pass" if ok else "fail"


def rms_error(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean square of the residual, the companion of NRMSE when the truth is small."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


# ---------------------------------------------------------------- main

def main() -> dict[str, Any]:
    started = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    import torch
    print(f"torch {torch.__version__} numpy {np.__version__} device cpu")

    data = make_split(cfg)
    weights = train_teacher(cfg, data["train"]["x0"], data["train"]["y"])
    # Drop the optimizer history's tensors; history is JSON-ready.
    _load_teacher(weights)

    # Runtime tensor semantics. These shapes are measured, not assumed.
    probe_x = data["train"]["x0"][:8]
    probe_t = np.linspace(0.0, 1.0, 8)
    f_probe = teacher_drift(weights, probe_x, probe_t)
    g_probe = teacher_g(weights, probe_x)
    semantics = {
        "x0": list(probe_x.shape),
        "t": list(probe_t.shape),
        "drift_f": list(f_probe.shape),
        "diffusion_g": list(g_probe.shape),
        "g_min_probe": float(g_probe.min()),
        "g_max_probe": float(g_probe.max()),
        "sigma_max": float(weights["sigma_max"]),
        "head_weight": [2, 2],
        "interpretation": (
            "f(x, t) is R^2 -> R^2 with an explicit time input. "
            "g(x0) is a positive scalar held constant along the path; "
            "it is not re-evaluated at x_t. The linear head is a readout "
            "of x_T and is not part of the SDE."
        ),
    }
    print("semantics", json.dumps(semantics))
    if f_probe.shape != (8, 2) or g_probe.shape != (8,):
        raise RuntimeError(f"unexpected teacher shapes {semantics}")
    if not (g_probe.min() > 0.0 and g_probe.max() < float(weights["sigma_max"]) + 1e-6):
        raise RuntimeError("diffusion left the sigmoid range")

    dt_train = float(cfg["teacher"]["T"]) / int(cfg["teacher"]["n_steps_train"])
    n_em = int(cfg["teacher"]["n_steps_train"])
    qrng = np.random.default_rng(int(cfg["distillation"]["query_seed"]))
    var = float(cfg["data"]["ood_noise_variance"])
    degrees = [int(d) for d in cfg["distillation"]["degrees"]]
    thresholds = [float(t) for t in cfg["distillation"]["thresholds"]]
    ridge = float(cfg["distillation"]["ridge"])

    drift_q = {}
    g_q = {}
    for name in ("train", "validation", "test"):
        x0 = data[name]["x0"]
        dx, dy = drift_queries(weights, x0, dt_train, n_em, qrng)
        drift_q[name] = (dx, dy)
        ood = perturb(qrng, x0, var)
        gx = np.concatenate([x0, ood], axis=0)
        gy = teacher_g(weights, gx)[:, None]
        regime = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x0), dtype=int)])
        g_q[name] = (gx, gy, regime)

    x_tr, y_tr = drift_q["train"]
    vif_drift = vif_max(x_tr, 3)
    vif_g = vif_max(g_q["train"][0], 3)

    drift_field, drift_meta = select_field(
        x_tr, y_tr, drift_q["validation"][0], drift_q["validation"][1],
        degrees, thresholds, ridge)
    g_field, g_meta = select_field_balanced(
        g_q["train"][0], g_q["train"][1], g_q["validation"][0], g_q["validation"][1],
        g_q["validation"][2], degrees, thresholds, ridge)

    # Ceilings: OLS on train, scored on test. Not candidates.
    drift_ceiling = fit_ols(x_tr, y_tr, 3)
    g_ceiling = fit_ols(g_q["train"][0], g_q["train"][1], 3)

    x_te, y_te = drift_q["test"]
    g_te, gy_te, reg_te = g_q["test"]
    id_te = reg_te == 0
    ood_te = reg_te == 1
    drift_hat = drift_field.predict(x_te)
    g_hat = g_field.predict(g_te)
    g_ceil = g_ceiling.predict(g_te)
    results: dict[str, Any] = {
        "semantics": semantics,
        "training_tail": weights["history"][-5:],
        "linear_x0_test_accuracy": linear_x0_accuracy(
            data["train"]["x0"], data["train"]["y"],
            data["test"]["x0"], data["test"]["y"]),
        "vif_drift_degree3": vif_drift,
        "vif_g_degree3": vif_g,
        "drift_selection": drift_meta,
        "g_selection": g_meta,
        "n_drift_queries": {k: int(len(v[0])) for k, v in drift_q.items()},
        "drift_test_nrmse": nrmse(y_te, drift_hat),
        "drift_test_rms": rms_error(y_te, drift_hat),
        "drift_ceiling_test_nrmse": nrmse(y_te, drift_ceiling.predict(x_te)),
        "drift_ceiling_test_rms": rms_error(y_te, drift_ceiling.predict(x_te)),
        "g_id_test_nrmse": nrmse(gy_te[id_te], g_hat[id_te]),
        "g_id_test_rms": rms_error(gy_te[id_te], g_hat[id_te]),
        "g_id_teacher_rms": float(np.sqrt(np.mean(gy_te[id_te] ** 2))),
        "g_ood_test_nrmse": nrmse(gy_te[ood_te], g_hat[ood_te]),
        "g_ood_test_rms": rms_error(gy_te[ood_te], g_hat[ood_te]),
        "g_ood_teacher_rms": float(np.sqrt(np.mean(gy_te[ood_te] ** 2))),
        "g_id_ceiling_nrmse": nrmse(gy_te[id_te], g_ceil[id_te]),
        "g_id_ceiling_rms": rms_error(gy_te[id_te], g_ceil[id_te]),
        "g_ood_ceiling_nrmse": nrmse(gy_te[ood_te], g_ceil[ood_te]),
        "g_ood_ceiling_rms": rms_error(gy_te[ood_te], g_ceil[ood_te]),
        "sigma_id_test_nrmse": nrmse(gy_te[id_te] ** 2, g_hat[id_te] ** 2),
        "query_ranges": {
            "drift_x": [float(x_tr[:, 0].min()), float(x_tr[:, 0].max())],
            "drift_y": [float(x_tr[:, 1].min()), float(x_tr[:, 1].max())],
            "drift_t": [float(x_tr[:, 2].min()), float(x_tr[:, 2].max())],
            "g_x": [float(g_q["train"][0][:, 0].min()), float(g_q["train"][0][:, 0].max())],
            "g_y": [float(g_q["train"][0][:, 1].min()), float(g_q["train"][0][:, 1].max())],
        },
    }
    # Range clause on the test diffusion queries.
    sigma_max = float(weights["sigma_max"])
    outside = (g_hat[:, 0] <= 0.0) | (g_hat[:, 0] > sigma_max)
    results["g_outside_range_fraction"] = float(np.mean(outside))
    results["mean_g"] = {
        "teacher_id": float(gy_te[id_te].mean()),
        "teacher_ood": float(gy_te[ood_te].mean()),
        "symbolic_id": float(g_hat[id_te].mean()),
        "symbolic_ood": float(g_hat[ood_te].mean()),
    }
    results["mean_g"]["teacher_gap"] = (results["mean_g"]["teacher_ood"]
                                        - results["mean_g"]["teacher_id"])
    results["mean_g"]["symbolic_gap"] = (results["mean_g"]["symbolic_ood"]
                                         - results["mean_g"]["symbolic_id"])

    # Off-manifold grid, reporting only.
    grid_xy = np.array([[x, y] for x in np.linspace(-4, 4, 17)
                        for y in np.linspace(-4, 4, 17)])
    centers = np.asarray(cfg["data"]["centers"], dtype=np.float64)
    dist_center = np.min(np.linalg.norm(grid_xy[:, None, :] - centers[None, :, :], axis=2), axis=1)
    far = dist_center > 1.2
    g_grid_teacher = teacher_g(weights, grid_xy)[:, None]
    g_grid_sym = g_field.predict(grid_xy)
    g_grid_ceil = g_ceiling.predict(grid_xy)
    grid_res: dict[str, Any] = {
        "g_symbolic_vs_teacher": nrmse(g_grid_teacher, g_grid_sym),
        "g_symbolic_vs_teacher_rms": float(np.sqrt(np.mean((g_grid_teacher - g_grid_sym) ** 2))),
        "g_ceiling_vs_teacher": nrmse(g_grid_teacher, g_grid_ceil),
        "g_ceiling_vs_teacher_rms": float(np.sqrt(np.mean((g_grid_teacher - g_grid_ceil) ** 2))),
        "g_far_symbolic_vs_teacher": nrmse(g_grid_teacher[far], g_grid_sym[far]),
        "g_far_symbolic_rms": float(np.sqrt(np.mean((g_grid_teacher[far] - g_grid_sym[far]) ** 2))),
        "g_far_ceiling_vs_teacher": nrmse(g_grid_teacher[far], g_grid_ceil[far]),
        "g_far_ceiling_rms": float(np.sqrt(np.mean((g_grid_teacher[far] - g_grid_ceil[far]) ** 2))),
        "n_grid": int(len(grid_xy)), "n_far": int(far.sum()),
    }
    for t_value in (0.0, 0.5, 1.0):
        xt = np.concatenate([grid_xy, np.full((len(grid_xy), 1), t_value)], axis=1)
        teacher_f = teacher_drift(weights, grid_xy, np.full(len(grid_xy), t_value))
        sym_f = drift_field.predict(xt)
        ceil_f = drift_ceiling.predict(xt)
        grid_res[f"t_{t_value}_drift_symbolic"] = nrmse(teacher_f, sym_f)
        grid_res[f"t_{t_value}_drift_symbolic_rms"] = float(np.sqrt(np.mean((teacher_f - sym_f) ** 2)))
        grid_res[f"t_{t_value}_drift_ceiling"] = nrmse(teacher_f, ceil_f)
        grid_res[f"t_{t_value}_drift_far_symbolic"] = nrmse(teacher_f[far], sym_f[far])
        grid_res[f"t_{t_value}_drift_far_rms"] = float(
            np.sqrt(np.mean((teacher_f[far] - sym_f[far]) ** 2)))
    results["grid"] = grid_res

    # ---- helpers bound to the fitted fields and the teacher ----
    def drift_teacher(x, t):
        tt = np.full(len(x), t)
        return teacher_drift(weights, x, tt)

    def g_teacher(x0):
        return teacher_g(weights, x0)

    def drift_symbolic(x, t):
        xt = np.concatenate([x, np.full((len(x), 1), t)], axis=1)
        return drift_field.predict(xt)

    def g_symbolic(x0):
        return g_field.predict(x0)[:, 0]

    def g_zero(x0):
        return np.zeros(len(x0))

    # ---- H4: ID simulation ----
    sim_cfg = cfg["simulation"]
    rng = np.random.default_rng(int(sim_cfg["seed"]))
    n_id = int(sim_cfg["n_id"])
    dt = float(sim_cfg["dt"])
    n_steps = int(round(float(sim_cfg["T"]) / dt))
    x0_id = data["test"]["x0"][:n_id]
    y_id = data["test"]["y"][:n_id]
    dW = rng.standard_normal((n_id, n_steps, 2)) * np.sqrt(dt)
    radius = float(sim_cfg["failure_radius"])
    exc = float(sim_cfg["excursion_radius"])
    sim_teacher = simulate_sdenet(drift_teacher, g_teacher, x0_id, dt, dW, radius, exc)
    sim_symbolic = simulate_sdenet(drift_symbolic, g_symbolic, x0_id, dt, dW, radius, exc)
    both = ~sim_teacher["failed"] & ~sim_symbolic["failed"]
    if both.any():
        paired = float(np.sqrt(np.mean((sim_teacher["final"][both] - sim_symbolic["final"][both]) ** 2)))
    else:
        paired = float("nan")
    results["simulation_id"] = {
        "teacher_failure_rate": sim_teacher["failure_rate"],
        "symbolic_failure_rate": sim_symbolic["failure_rate"],
        "teacher_excursion_rate": sim_teacher["excursion_rate"],
        "symbolic_excursion_rate": sim_symbolic["excursion_rate"],
        "teacher_failure_class": {
            c: int(sum(v == c for v in sim_teacher["failure_class"]))
            for c in ("non_finite", "outside_radius")},
        "symbolic_failure_class": {
            c: int(sum(v == c for v in sim_symbolic["failure_class"]))
            for c in ("non_finite", "outside_radius")},
        "paired_terminal_rmse": paired,
        "teacher_accuracy": retained_accuracy(
            teacher_logits(weights, np.nan_to_num(sim_teacher["final"])), y_id, sim_teacher["failed"]),
        "symbolic_accuracy": retained_accuracy(
            teacher_logits(weights, np.nan_to_num(sim_symbolic["final"])), y_id, sim_symbolic["failed"]),
        "teacher_terminal_rms": float(np.sqrt(np.mean(sim_teacher["final"] ** 2))),
        "symbolic_terminal_rms": float(np.sqrt(np.mean(sim_symbolic["final"] ** 2))),
        "n": n_id, "dt": dt, "n_steps": n_steps,
    }

    # ---- H5: terminal variance, ID and OOD, K paths ----
    n_var = int(sim_cfg["variance_n"])
    k_paths = int(sim_cfg["variance_paths"])
    x0_var = data["test"]["x0"][:n_var]
    x0_ood = perturb(rng, x0_var, var)
    dW_k = rng.standard_normal((k_paths, n_var, n_steps, 2)) * np.sqrt(dt)
    var_teacher_id = terminal_variance(drift_teacher, g_teacher, x0_var, dt, dW_k, radius)
    var_sym_id = terminal_variance(drift_symbolic, g_symbolic, x0_var, dt, dW_k, radius)
    var_zero_id = terminal_variance(drift_teacher, g_zero, x0_var, dt, dW_k, radius)
    var_teacher_ood = terminal_variance(drift_teacher, g_teacher, x0_ood, dt, dW_k, radius)
    var_sym_ood = terminal_variance(drift_symbolic, g_symbolic, x0_ood, dt, dW_k, radius)
    var_zero_ood = terminal_variance(drift_teacher, g_zero, x0_ood, dt, dW_k, radius)

    def ratio(a: dict, b: dict) -> float:
        if not np.isfinite(a["variance"]) or not np.isfinite(b["variance"]):
            return float("nan")
        return float(a["variance"] / max(b["variance"], 1e-15))

    results["terminal_variance"] = {
        "teacher_id": var_teacher_id, "symbolic_id": var_sym_id, "nodiff_id": var_zero_id,
        "teacher_ood": var_teacher_ood, "symbolic_ood": var_sym_ood, "nodiff_ood": var_zero_ood,
        "teacher_ratio": ratio(var_teacher_ood, var_teacher_id),
        "symbolic_ratio": ratio(var_sym_ood, var_sym_id),
    }

    # ---- H6: coarse vs fine, common random numbers, ID test ----
    # Exact fractions: coarse step is the training step T/6, fine step is
    # T/30, so each coarse increment is the sum of 5 fine increments.
    fine_steps = 30
    block = fine_steps // n_em
    fine_dt = 1.0 / fine_steps
    coarse_dt = 1.0 / n_em
    if abs(fine_dt - float(cfg["hypotheses"]["H6_discretization"]["fine_dt"])) > 1e-12:
        raise RuntimeError("H6 fine_dt drifted from the frozen config")
    if fine_steps % n_em != 0:
        raise RuntimeError("H6 block aggregation is not exact")
    dW_fine = rng.standard_normal((n_id, fine_steps, 2)) * np.sqrt(fine_dt)
    dW_coarse = aggregate_increments(dW_fine, block)
    fine_sim = simulate_sdenet(drift_teacher, g_teacher, x0_id, fine_dt, dW_fine, radius, exc)
    coarse_sim = simulate_sdenet(drift_teacher, g_teacher, x0_id, coarse_dt, dW_coarse, radius, exc)
    both_h6 = ~fine_sim["failed"] & ~coarse_sim["failed"]
    if both_h6.any():
        h6_rmse = float(np.sqrt(np.mean((fine_sim["final"][both_h6]
                                         - coarse_sim["final"][both_h6]) ** 2)))
    else:
        h6_rmse = float("nan")
    results["h6"] = {
        "fine_dt": fine_dt, "coarse_dt": coarse_dt, "block": block,
        "fine_failure_rate": fine_sim["failure_rate"],
        "coarse_failure_rate": coarse_sim["failure_rate"],
        "paired_terminal_rmse": h6_rmse,
    }

    # Per-step SNR (diagnostic, not a verdict). ID test x0 at t=0 versus
    # the OOD test cloud at t=0. Drift contribution |f|dt against g sqrt(dt).
    snr_rows = {
        "id": np.concatenate([data["test"]["x0"], np.zeros((len(data["test"]["x0"]), 1))], axis=1),
        "ood": np.concatenate([g_te[ood_te], np.zeros((int(ood_te.sum()), 1))], axis=1),
    }
    snr_dt = 0.05
    snr: dict[str, float] = {}
    for name, rows in snr_rows.items():
        f_abs = np.linalg.norm(teacher_drift(weights, rows[:, :2], rows[:, 2]), axis=1)
        g_abs = np.abs(teacher_g(weights, rows[:, :2]))
        drift_step = float(np.mean(f_abs * snr_dt))
        diff_step = float(np.mean(g_abs * np.sqrt(snr_dt)))
        snr[f"{name}_drift_step"] = drift_step
        snr[f"{name}_diffusion_step"] = diff_step
        # Ratio of means, not the mean of ratios: g is near 0 on some ID
        # points, and the mean of |f|/g then diverges (denominator artifact).
        snr[f"{name}_snr_ratio_of_means"] = float(drift_step / max(diff_step, 1e-15))
    results["step_contributions"] = snr

    # ---- verdicts from the frozen config ----
    h = cfg["hypotheses"]
    sim = results["simulation_id"]
    mg = results["mean_g"]
    gate = float(h["H5_paper_axis_id_ood_gap"]["failure_gate"])
    tv = results["terminal_variance"]

    def gated(population_failure: float, ok: bool) -> str:
        if population_failure > gate:
            return "gated"
        return _pass(ok)

    verdicts = {
        "H1": _pass(results["drift_test_nrmse"]
                    <= h["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"]),
        "H2_id": _pass(results["g_id_test_nrmse"]
                       <= h["H2_diffusion_distillation"]["symbolic_g_id_test_nrmse_vs_teacher_max"]),
        "H2_ood": _pass(results["g_ood_test_nrmse"]
                        <= h["H2_diffusion_distillation"]["symbolic_g_ood_test_nrmse_vs_teacher_max"]),
        "H2_sigma": _pass(results["sigma_id_test_nrmse"]
                          <= h["H2_diffusion_distillation"]["symbolic_sigma_id_test_nrmse_vs_teacher_max"]),
        "H2c": _pass(results["g_outside_range_fraction"] <= h["H2c_range"]["max"]),
        "H3_drift_vif": ("indeterminate" if vif_drift > h["H3_collinearity_gate"]["vif_max"]
                         else "pass"),
        "H3_g_vif": ("indeterminate" if vif_g > h["H3_collinearity_gate"]["vif_max"]
                     else "pass"),
        "H4_failure": _pass(
            sim["symbolic_failure_rate"] <= h["H4_simulation"]["symbolic_failure_rate_max"]
            and sim["symbolic_failure_rate"] - sim["teacher_failure_rate"]
            <= h["H4_simulation"]["symbolic_failure_rate_minus_teacher_max"]),
        "H4_excursion": _pass(
            sim["symbolic_excursion_rate"] - sim["teacher_excursion_rate"]
            <= h["H4_simulation"]["excursion_rate_symbolic_minus_teacher_max"]),
        "H4_paired": _pass(sim["paired_terminal_rmse"]
                           <= h["H4_simulation"]["paired_terminal_rmse_max"]),
        "H4_teacher_accuracy": _pass(sim["teacher_accuracy"]
                                     >= h["H4_simulation"]["teacher_accuracy_min"]),
        "H4_symbolic_accuracy": _pass(sim["symbolic_accuracy"]
                                      >= h["H4_simulation"]["symbolic_accuracy_min"]),
        "H5_teacher_g_id": _pass(mg["teacher_id"] <= h["H5_paper_axis_id_ood_gap"]["teacher_mean_g_id_max"]),
        "H5_teacher_g_ood": _pass(mg["teacher_ood"] >= h["H5_paper_axis_id_ood_gap"]["teacher_mean_g_ood_min"]),
        "H5_symbolic_g_id": _pass(mg["symbolic_id"] <= h["H5_paper_axis_id_ood_gap"]["symbolic_mean_g_id_max"]),
        "H5_symbolic_g_ood": _pass(mg["symbolic_ood"] >= h["H5_paper_axis_id_ood_gap"]["symbolic_mean_g_ood_min"]),
        "H5_teacher_gap": _pass(mg["teacher_gap"] >= h["H5_paper_axis_id_ood_gap"]["teacher_gap_min"]),
        "H5_symbolic_gap": _pass(mg["symbolic_gap"] >= h["H5_paper_axis_id_ood_gap"]["symbolic_gap_min"]),
        "H5_teacher_var_ratio": gated(
            max(tv["teacher_id"]["failure_rate"], tv["teacher_ood"]["failure_rate"]),
            tv["teacher_ratio"] >= h["H5_paper_axis_id_ood_gap"]["teacher_terminal_var_ratio_min"]),
        "H5_symbolic_var_ratio": gated(
            max(tv["symbolic_id"]["failure_rate"], tv["symbolic_ood"]["failure_rate"]),
            tv["symbolic_ratio"] >= h["H5_paper_axis_id_ood_gap"]["symbolic_terminal_var_ratio_min"]),
        "H5_nodiff": _pass(tv["nodiff_ood"]["variance"]
                           <= h["H5_paper_axis_id_ood_gap"]["nodiff_ood_terminal_var_max"]),
        "H6": ("indeterminate" if max(results["h6"]["fine_failure_rate"],
                                      results["h6"]["coarse_failure_rate"])
               > h["H6_discretization"]["failure_gate"]
               else _pass(results["h6"]["paired_terminal_rmse"]
                          >= h["H6_discretization"]["paired_terminal_rmse_min"])),
    }
    results["verdicts"] = verdicts

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drift_field.save(OUT_DIR / "symbolic_f.json")
    g_field.save(OUT_DIR / "symbolic_g.json")
    weight_payload = {k: weights[k] for k in
                      ("drift", "diffusion", "head_weight", "head_bias", "width", "sigma_max")}
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(weight_payload))
    expanded_f = expand_to_original(drift_field)
    expanded_g = expand_to_original(g_field)
    tex = (
        "Symbolic SDE in the original coordinates. "
        "The diffusion is evaluated at the initial condition $(x_0, y_0)$, "
        "not at the current state.\n\n$$\n"
        + tex_equations(expanded_f, ("f_{1}(x, y, t)", "f_{2}(x, y, t)"),
                        var_names=("x", "y", "t"))
        + "\n$$\n\n$$\n"
        + tex_equations(expanded_g, ("g(x_0, y_0)",), var_names=("x", "y"))
        + "\n$$\n\n"
        "\\[\n"
        "\\begin{aligned}\n"
        "dx &= f_{1}(x, y, t)\\,dt + g(x_0, y_0)\\,dW^{1},\\\\\n"
        "dy &= f_{2}(x, y, t)\\,dt + g(x_0, y_0)\\,dW^{2}.\n"
        "\\end{aligned}\n"
        "\\]\n"
    )
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": [{",".join(map(str, exp)): coef for exp, coef in poly.items()}
                  for poly in expanded_f],
        "diffusion": [{",".join(map(str, exp)): coef for exp, coef in poly.items()}
                      for poly in expanded_g],
    }, indent=2))
    results["elapsed_seconds"] = float(time.time() - started)
    results["torch"] = torch.__version__
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"verdicts": verdicts,
                      "drift_test_nrmse": results["drift_test_nrmse"],
                      "g_id_test_nrmse": results["g_id_test_nrmse"],
                      "g_ood_test_nrmse": results["g_ood_test_nrmse"],
                      "mean_g": mg,
                      "simulation_id": sim,
                      "elapsed_seconds": results["elapsed_seconds"]}, indent=2))
    return results


if __name__ == "__main__":
    main()
