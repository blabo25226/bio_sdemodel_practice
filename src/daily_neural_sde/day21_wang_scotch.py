"""Day 21: Wang, Jennings & Gong (ICLR 2024) SCOTCH neural SDE.

Pipeline (settings frozen in ``configs/daily_neural_sde_day21.json``):

1. System: a 3D cubic chain in the paper's observational Ito class. The
   signature graph is sparse. Lorenz-96 and glycolysis are out of scope.
2. Teacher: graph-gated drift and graph-gated diagonal diffusion, equation
   (33) / Appendix B. The graph is fixed at the true signature. Training is
   the Euler-Maruyama Gaussian transition likelihood, not the latent ELBO.
3. Distillation: direct queries of those two fields. The library is powers
   of ``(x, y, z)`` only. Validation chooses degree and threshold. Test is
   scored once. A parent-restricted library is a second fit and does not
   choose the reported polynomial.

Run from the repository root:
``python -m src.daily_neural_sde.day21_wang_scotch``

Outputs land in ``outputs/daily_neural_sde/day21/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day21.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day21"
REPORT_DIR = ROOT / "reports" / "daily_neural_sde" / "day21_wang_scotch"
DTYPE = torch.float64
VAR_NAMES = ("x", "y", "z")
COMPONENT_NAMES = ("x", "y", "z")
FieldFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-21 config."""
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


def truth_graph(cfg: dict[str, Any]) -> np.ndarray:
    """Signature graph, shape ``(3, 3)``. Rows are sources, columns are targets."""
    graph = np.asarray(cfg["truth"]["graph"], dtype=np.float64)
    if graph.shape != (3, 3):
        raise ValueError("day 21 graph must be 3 by 3")
    return graph


def truth_drift(x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Cubic chain drift. ``x`` has shape ``(n, 3)`` and the result matches it."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    linear = np.asarray(cfg["truth"]["linear"], dtype=np.float64)
    cubic_x = float(cfg["truth"]["cubic_x"])
    cubic_z = float(cfg["truth"]["cubic_z"])
    out = x @ linear.T
    out[:, 0] = out[:, 0] + cubic_x * x[:, 0] ** 3
    out[:, 2] = out[:, 2] + cubic_z * x[:, 2] ** 3
    return out


def truth_diffusion(x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Constant diagonal diffusion, broadcast to ``x`` of shape ``(n, 3)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    sigma = np.asarray(cfg["truth"]["sigma"], dtype=np.float64)
    return np.broadcast_to(sigma, x.shape).copy()


class SoftplusMLP(nn.Module):
    """Two softplus hidden layers and a linear head.

    A residual skip is added on the second hidden layer, whose widths match.
    That is the ``--res_connection`` case used for the Lorenz run.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(..., in_dim)`` to ``(..., out_dim)``."""
        hidden = torch.nn.functional.softplus(self.fc1(x))
        second = torch.nn.functional.softplus(self.fc2(hidden))
        if hidden.shape[-1] == second.shape[-1]:
            second = second + hidden
        return self.fc3(second)


class GraphGatedField(nn.Module):
    """SCOTCH prior coefficient, Appendix B equation (33).

    ``f_d(z) = zeta(sum_i G_{i,d} l(z_i, e_i), e_d)``. ``z`` has shape
    ``(n, d)`` and the output has the same shape. A zero graph entry removes
    that source from the sum, so the partial in that source is identically zero.
    """

    def __init__(self, dimension: int, embed: int, hidden: int, graph: torch.Tensor,
                 sigmoid_head: bool) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.embed_dim = int(embed)
        self.sigmoid_head = bool(sigmoid_head)
        self.embed = nn.Parameter(torch.randn(dimension, embed) * 0.1)
        self.local = SoftplusMLP(1 + embed, hidden, embed)
        self.zeta = SoftplusMLP(embed + embed, hidden, 1)
        self.register_buffer("graph", graph)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the gated field. ``z`` is ``(n, d)``."""
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("expected states (n, d)")
        n = z.shape[0]
        embed = self.embed.unsqueeze(0).expand(n, -1, -1)
        local_in = torch.cat([z.unsqueeze(-1), embed], dim=-1)
        local = self.local(local_in)
        aggregated = torch.einsum("st,nse->nte", self.graph, local)
        zeta_in = torch.cat([aggregated, embed], dim=-1)
        out = self.zeta(zeta_in).squeeze(-1)
        if self.sigmoid_head:
            out = torch.sigmoid(out)
        return out


class ScotchTeacher(nn.Module):
    """Prior drift and prior diffusion at one fixed signature graph."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model = cfg["model"]
        graph = torch.as_tensor(truth_graph(cfg), dtype=DTYPE)
        embed = int(model["embed"])
        hidden = int(model["hidden"])
        self.drift = GraphGatedField(3, embed, hidden, graph, sigmoid_head=False)
        self.diffusion = GraphGatedField(3, embed, hidden, graph.clone(), sigmoid_head=True)
        if model.get("diffusion_bias_at_logit_sigma", False):
            sigma = float(cfg["truth"]["sigma"][0])
            bias = math.log(sigma / (1.0 - sigma))
            nn.init.constant_(self.diffusion.zeta.fc3.bias, bias)

    def drift_value(self, z: torch.Tensor) -> torch.Tensor:
        """Neural drift, shape ``(n, 3)``."""
        return self.drift(z)

    def diffusion_value(self, z: torch.Tensor) -> torch.Tensor:
        """Neural diagonal diffusion, shape ``(n, 3)``, values in ``(0, 1)``."""
        return self.diffusion(z)


def generate_truth(cfg: dict[str, Any], n_series: int, seed: int) -> np.ndarray:
    """Euler-Maruyama paths of the cubic chain.

    Returns states of shape ``(n_series, n_obs + 1, 3)`` on the observation grid.
    """
    data = cfg["data"]
    dt = float(data["dt_gen"])
    horizon = float(data["horizon"])
    stride = int(round(float(data["dt_obs"]) / dt))
    if abs(stride * dt - float(data["dt_obs"])) > 1e-12:
        raise ValueError("dt_obs must be an integer multiple of dt_gen")
    n_steps = int(round(horizon / dt))
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, float(data["x0_std"]), size=(n_series, 3))
    sigma = np.asarray(cfg["truth"]["sigma"], dtype=np.float64)
    recorded = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps):
        drift = truth_drift(x, cfg)
        shock = rng.normal(size=x.shape)
        x = x + drift * dt + sigma * sqrt_dt * shock
        if (step + 1) % stride == 0:
            recorded.append(x.copy())
    return np.stack(recorded, axis=1)


def transition_pairs(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten path increments. Both arrays have shape ``(n_series * n_steps, 3)``."""
    current = paths[:, :-1, :].reshape(-1, 3)
    nxt = paths[:, 1:, :].reshape(-1, 3)
    return current, nxt


def shifted_gaussian_nll(current: np.ndarray, nxt: np.ndarray, drift: np.ndarray,
                         diffusion: np.ndarray, dt: float) -> float:
    """One-step Gaussian NLL without the constant ``0.5 log(2 pi)``.

    ``current`` and ``nxt`` have shape ``(n, 3)``. The omitted constant does
    not change the minimizer. It must be added back before comparing with a
    textbook normal density.
    """
    var = (diffusion * math.sqrt(dt)) ** 2
    mean = current + drift * dt
    return float(0.5 * np.mean((nxt - mean) ** 2 / var + np.log(var)))


def train_teacher(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit the gated drift and diffusion by one-step Gaussian likelihood."""
    torch.manual_seed(int(cfg["seeds"]["network"]))
    model_cfg = cfg["model"]
    teacher = ScotchTeacher(cfg).to(dtype=DTYPE)
    paths = generate_truth(cfg, int(cfg["data"]["n_series"]), int(cfg["seeds"]["data"]))
    current_np, nxt_np = transition_pairs(paths)
    current = torch.as_tensor(current_np, dtype=DTYPE)
    nxt = torch.as_tensor(nxt_np, dtype=DTYPE)
    dt = float(cfg["data"]["dt_obs"])
    optimizer = torch.optim.Adam(teacher.parameters(), lr=float(model_cfg["lr"]))
    batch = int(model_cfg["batch_size"])
    losses: list[float] = []
    n = current.shape[0]
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seeds"]["network"]))
    for _ in range(int(model_cfg["epochs"])):
        order = torch.randperm(n, generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, n, batch):
            pick = order[start:start + batch]
            x = current[pick]
            y = nxt[pick]
            drift = teacher.drift_value(x)
            diffusion = teacher.diffusion_value(x)
            mean = x + drift * dt
            var = (diffusion * math.sqrt(dt)) ** 2
            loss = 0.5 * (((y - mean) ** 2) / var + torch.log(var)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), float(model_cfg["grad_clip"]))
            optimizer.step()
            total += float(loss.detach()) * len(pick)
            seen += len(pick)
        losses.append(total / max(seen, 1))
    return {"model": teacher, "loss": losses, "n_transitions": int(n),
            "path_abs_max": float(np.max(np.abs(paths)))}


def save_weights(teacher: ScotchTeacher, path: Path) -> None:
    """Save the state dict as JSON lists. No pickle."""
    payload = {key: value.detach().cpu().numpy().tolist()
               for key, value in teacher.state_dict().items()}
    path.write_text(json.dumps(payload))


def load_weights(teacher: ScotchTeacher, path: Path) -> ScotchTeacher:
    """Restore a teacher saved with :func:`save_weights`."""
    payload = json.loads(path.read_text())
    state = {key: torch.as_tensor(value, dtype=DTYPE) for key, value in payload.items()}
    teacher.load_state_dict(state)
    return teacher


def field_np(module: nn.Module, x: np.ndarray) -> np.ndarray:
    """Evaluate a torch field on ``x`` of shape ``(n, 3)``."""
    with torch.no_grad():
        tensor = torch.as_tensor(np.atleast_2d(x), dtype=DTYPE)
        return module(tensor).detach().cpu().numpy()


def simulate_em_diag(drift_fn: FieldFn, diffusion_fn: FieldFn, x0: np.ndarray, dt: float,
                     increments: np.ndarray, failure_radius: float) -> dict[str, Any]:
    """Paired Euler-Maruyama with diagonal diffusion and latched failures.

    ``increments`` has shape ``(n, n_steps, d)``. Classes are ``non_finite``,
    ``outside_radius``, and ``nonpositive_diffusion``. Failed rows stay NaN.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    n, _ = x.shape
    n_steps = increments.shape[1]
    if increments.shape[0] != n:
        raise ValueError("increments do not match the batch")
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    observed = [x.copy()]
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps):
        active_idx = np.flatnonzero(~failed)
        if len(active_idx):
            drift = np.asarray(drift_fn(x[active_idx]), dtype=np.float64)
            diff = np.asarray(diffusion_fn(x[active_idx]), dtype=np.float64)
            bad_g = (~np.isfinite(diff).all(axis=1)) | (np.min(diff, axis=1) <= 0.0)
            bad_f = ~np.isfinite(drift).all(axis=1)
            for local, idx in enumerate(active_idx):
                if bad_g[local] or bad_f[local]:
                    failed[idx] = True
                    failure_class[idx] = "nonpositive_diffusion" if bad_g[local] else "non_finite"
                    continue
                x[idx] = x[idx] + drift[local] * dt + diff[local] * sqrt_dt * increments[idx, step]
        bad_nf = ~np.isfinite(x).all(axis=1)
        bad_radius = np.linalg.norm(x, axis=1) > failure_radius
        newly = (~failed) & (bad_nf | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nf[idx] else "outside_radius"
        failed = failed | newly
        x[failed] = np.nan
        observed.append(x.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class, "failure_rate": float(np.mean(failed))}


def rollout_queries(teacher: ScotchTeacher, cfg: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    """Teacher paths used as distillation states. Failures are kept out of the split."""
    data = cfg["data"]
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, float(data["x0_std"]), size=(n, 3))
    steps = int(data["n_sim_steps"])
    increments = rng.normal(size=(n, steps, 3))
    sim = simulate_em_diag(lambda z: field_np(teacher.drift, z),
                           lambda z: field_np(teacher.diffusion, z),
                           x0, float(data["dt_sim"]), increments, float(data["failure_radius"]))
    finite = ~sim["failed"]
    states = sim["states"][finite]
    owners = []
    rows = []
    for owner, path in enumerate(states):
        keep = np.isfinite(path).all(axis=1)
        # Drop the terminal row's successor gap: every finite interior state is a query.
        used = path[keep]
        if len(used) == 0:
            continue
        rows.append(used)
        owners.append(np.full(len(used), owner, dtype=int))
    if not rows:
        raise RuntimeError("every teacher rollout failed")
    return {"states": np.concatenate(rows, axis=0),
            "owners": np.concatenate(owners),
            "n_finite": int(finite.sum()),
            "failure_rate": sim["failure_rate"],
            "failure_class": sim["failure_class"]}


def split_owners(owners: np.ndarray, n_train: int, n_val: int, n_test: int) -> dict[str, np.ndarray]:
    """Trajectory-disjoint masks. Owners are already in rollout order."""
    unique = []
    for owner in owners:
        if not unique or unique[-1] != int(owner):
            unique.append(int(owner))
    need = n_train + n_val + n_test
    if len(unique) < need:
        raise RuntimeError(f"need {need} finite trajectories, found {len(unique)}")
    chosen = unique[:need]
    groups = {"train": set(chosen[:n_train]),
              "val": set(chosen[n_train:n_train + n_val]),
              "test": set(chosen[n_train + n_val:need])}
    masks = {name: np.isin(owners, list(group)) for name, group in groups.items()}
    masks["n_finite_trajectories"] = len(unique)
    return masks


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


def parent_columns(graph: np.ndarray, component: int) -> np.ndarray:
    """Source indices that the signature graph allows into one component."""
    return np.flatnonzero(graph[:, component] > 0.5)


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


def constant_floor_from_train(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """NRMSE of predicting the training mean on the test targets."""
    return nrmse(y_test, np.full_like(y_test, float(np.mean(y_train))))


def stack_predict(fields: list[PolynomialField], x: np.ndarray) -> np.ndarray:
    """Stack per-component polynomial fields. Result has shape ``(n, k)``."""
    columns = [field.predict(x)[:, 0] for field in fields]
    return np.column_stack(columns)


def parent_predict(fields: list[PolynomialField], x: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Evaluate parent-restricted fields. Each field sees only its parent columns."""
    columns = []
    for component, field in enumerate(fields):
        cols = parent_columns(graph, component)
        columns.append(field.predict(x[:, cols])[:, 0])
    return np.column_stack(columns)


def partial_tensor(fn: FieldFn, x: np.ndarray, eps: float) -> np.ndarray:
    """Central partials. Result has shape ``(n, component, variable)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    grads = np.zeros((len(x), x.shape[1], x.shape[1]), dtype=np.float64)
    for variable in range(x.shape[1]):
        plus = x.copy()
        minus = x.copy()
        plus[:, variable] += eps
        minus[:, variable] -= eps
        grads[:, :, variable] = (fn(plus) - fn(minus)) / (2.0 * eps)
    return grads


def signature_partials(grads: np.ndarray, graph: np.ndarray) -> dict[str, float]:
    """Split a Jacobian sample into parent and non-parent entries."""
    parent = graph.T > 0.5
    # grads is (n, component, variable); graph is (source, target).
    parent_mask = np.broadcast_to(parent, grads.shape)
    return {"parent_rms": rms(grads[parent_mask]),
            "nonparent_rms": rms(grads[~parent_mask])}


def make_grid(cfg: dict[str, Any]) -> np.ndarray:
    """Product grid of the three coordinates. Shape ``(n, 3)``."""
    axis = np.asarray(cfg["data"]["grid_axis"], dtype=np.float64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def fresh_initial(cfg: dict[str, Any], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Initial states and standard normal increments for one simulation."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, float(cfg["data"]["x0_std"]), size=(n, 3))
    steps = int(cfg["data"]["n_sim_steps"])
    increments = rng.normal(size=(n, steps, 3))
    return x0, increments


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
    left_mean = mean_curve(left, left_failed)
    right_mean = mean_curve(right, right_failed)
    if left_mean is None or right_mean is None:
        return float("nan")
    return float(np.sqrt(np.mean((left_mean - right_mean) ** 2)))


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
        counts += np.abs(field.coefficients[0]) > 0
    return (counts / draws).tolist()


def component_record(prediction: np.ndarray, teacher: np.ndarray, y_train: np.ndarray,
                     ceiling: float) -> dict[str, float]:
    """One component of a field comparison on the test queries."""
    return {"nrmse": nrmse(teacher, prediction),
            "residual_rms": rms(prediction - teacher),
            "teacher_rms": rms(teacher),
            "floor": constant_floor_from_train(y_train, teacher),
            "ceiling": ceiling}


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, str]:
    """Apply the frozen gates. Thresholds are not recomputed here."""
    h1_cfg = cfg["hypotheses"]["H1_drift"]
    h2_cfg = cfg["hypotheses"]["H2_diffusion"]
    h3_cfg = cfg["hypotheses"]["H3_collinearity_gate"]
    h4_cfg = cfg["hypotheses"]["H4_paired_simulation"]
    h5_cfg = cfg["hypotheses"]["H5_law"]
    h6_cfg = cfg["hypotheses"]["H6_off_manifold"]
    h7_cfg = cfg["hypotheses"]["H7_signature"]
    h8_cfg = cfg["hypotheses"]["H8_parent_library"]
    h9_cfg = cfg["hypotheses"]["H9_truth"]
    h1 = all(row["nrmse"] <= h1_cfg["symbolic_vs_teacher_test_nrmse_max"]
             and row["nrmse"] < row["floor"] for row in metrics["drift"])
    h2 = (all(row["nrmse"] <= h2_cfg["symbolic_vs_teacher_test_nrmse_max"]
              for row in metrics["diffusion"])
          and metrics["diffusion_nonpositive_fraction"] <= h2_cfg["nonpositive_fraction_max"])
    h3 = "indeterminate" if metrics["vif"] > h3_cfg["vif_max"] else "pass"
    failure_ok = (metrics["fail_teacher"] <= h4_cfg["failure_rate_max"]
                  and metrics["fail_symbolic"] <= h4_cfg["failure_rate_max"]
                  and abs(metrics["fail_teacher"] - metrics["fail_symbolic"]) <= h4_cfg["failure_gap_max"])
    h4 = "indeterminate" if not failure_ok else _pass(metrics["path_rmse"] <= h4_cfg["path_rmse_max"])
    law_fail_ok = (metrics["law_fail_teacher"] <= h5_cfg["failure_rate_max"]
                   and metrics["law_fail_symbolic"] <= h5_cfg["failure_rate_max"]
                   and metrics["law_fail_floor"] <= h5_cfg["failure_rate_max"])
    if (not law_fail_ok) or metrics["law_floor"] > h5_cfg["floor_mean_path_rmse_max"]:
        h5 = "indeterminate"
    else:
        h5 = _pass(metrics["law_symbolic"] <= h5_cfg["symbolic_mean_path_rmse_max"])
    h6 = _pass(metrics["grid_nrmse"] <= h6_cfg["grid_nrmse_max"])
    teacher_gate = metrics["teacher_nonparent_rms"] <= h7_cfg["teacher_nonparent_partial_rms_max"]
    if metrics["teacher_parent_rms"] < h7_cfg["parent_rms_min_for_ratio"]:
        h7 = "indeterminate" if teacher_gate else "fail"
    else:
        ratio_ok = metrics["symbolic_relative_nonparent"] <= h7_cfg["symbolic_relative_nonparent_max"]
        h7 = _pass(teacher_gate and ratio_ok)
    h8 = _pass(metrics["parent_library_nrmse"] <= metrics["full_library_nrmse"] + h8_cfg["gap_max"])
    h9 = _pass(metrics["teacher_vs_truth_drift"] <= h9_cfg["drift_nrmse_max"]
               and metrics["symbolic_vs_truth_drift"] <= h9_cfg["drift_nrmse_max"]
               and metrics["teacher_vs_truth_diffusion"] <= h9_cfg["diffusion_nrmse_max"]
               and metrics["symbolic_vs_truth_diffusion"] <= h9_cfg["diffusion_nrmse_max"])
    return {"H1": h1 and "pass" or "fail", "H2": h2 and "pass" or "fail", "H3": h3,
            "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8, "H9": h9}


def evaluate(cfg: dict[str, Any], teacher: ScotchTeacher, states: np.ndarray,
             masks: dict[str, np.ndarray], drift_fields: list[PolynomialField],
             diffusion_fields: list[PolynomialField], parent_fields: list[PolynomialField],
             meta: dict[str, Any]) -> dict[str, Any]:
    """Score the frozen test split once, then the simulations and the grid."""
    graph = truth_graph(cfg)
    x_train, x_val, x_test = states[masks["train"]], states[masks["val"]], states[masks["test"]]
    del x_val
    drift_train = field_np(teacher.drift, x_train)
    drift_test = field_np(teacher.drift, x_test)
    diff_train = field_np(teacher.diffusion, x_train)
    diff_test = field_np(teacher.diffusion, x_test)
    sym_drift = stack_predict(drift_fields, x_test)
    sym_diff = stack_predict(diffusion_fields, x_test)
    parent_drift = parent_predict(parent_fields, x_test, graph)
    ridge = float(cfg["distill"]["ridge"])
    drift_rows = []
    diffusion_rows = []
    for j in range(3):
        ceiling_field = fit_polynomial_field(x_train, drift_train[:, j:j + 1], 3, 0.0, ridge)
        ceiling = nrmse(drift_test[:, j], ceiling_field.predict(x_test)[:, 0])
        drift_rows.append(component_record(sym_drift[:, j], drift_test[:, j], drift_train[:, j], ceiling))
        drift_rows[-1]["degree"] = meta[f"f{j + 1}"]["degree"]
        drift_rows[-1]["threshold"] = meta[f"f{j + 1}"]["threshold"]
        drift_rows[-1]["active"] = meta[f"f{j + 1}"]["active"]
        drift_rows[-1]["matches_ceiling"] = bool(meta[f"f{j + 1}"]["degree"] == 3
                                                 and float(meta[f"f{j + 1}"]["threshold"]) == 0.0)
        diff_ceiling_field = fit_polynomial_field(x_train, diff_train[:, j:j + 1], 3, 0.0, ridge)
        diff_ceiling = nrmse(diff_test[:, j], diff_ceiling_field.predict(x_test)[:, 0])
        diffusion_rows.append(component_record(sym_diff[:, j], diff_test[:, j], diff_train[:, j], diff_ceiling))
        diffusion_rows[-1]["degree"] = meta[f"g{j + 1}"]["degree"]
        diffusion_rows[-1]["threshold"] = meta[f"g{j + 1}"]["threshold"]
        diffusion_rows[-1]["active"] = meta[f"g{j + 1}"]["active"]
        diffusion_rows[-1]["matches_ceiling"] = bool(meta[f"g{j + 1}"]["degree"] == 3
                                                     and float(meta[f"g{j + 1}"]["threshold"]) == 0.0)
        drift_rows[-1]["parent_nrmse"] = nrmse(drift_test[:, j], parent_drift[:, j])
    data = cfg["data"]
    x0, increments = fresh_initial(cfg, int(data["n_paired"]), int(cfg["seeds"]["paired"]))
    teacher_sim = simulate_em_diag(lambda z: field_np(teacher.drift, z),
                                   lambda z: field_np(teacher.diffusion, z),
                                   x0, float(data["dt_sim"]), increments, float(data["failure_radius"]))
    symbolic_sim = simulate_em_diag(lambda z: stack_predict(drift_fields, z),
                                    lambda z: stack_predict(diffusion_fields, z),
                                    x0, float(data["dt_sim"]), increments, float(data["failure_radius"]))
    paired = path_rmse(teacher_sim["states"], symbolic_sim["states"],
                       teacher_sim["failed"], symbolic_sim["failed"])
    def _law(seed: int, drift_fn: FieldFn, diffusion_fn: FieldFn) -> dict[str, Any]:
        start, shocks = fresh_initial(cfg, int(data["n_law"]), seed)
        return simulate_em_diag(drift_fn, diffusion_fn, start, float(data["dt_sim"]), shocks,
                                float(data["failure_radius"]))
    law_teacher = _law(int(cfg["seeds"]["law_a"]),
                       lambda z: field_np(teacher.drift, z),
                       lambda z: field_np(teacher.diffusion, z))
    law_floor = _law(int(cfg["seeds"]["law_b"]),
                     lambda z: field_np(teacher.drift, z),
                     lambda z: field_np(teacher.diffusion, z))
    law_symbolic = _law(int(cfg["seeds"]["law_symbolic"]),
                        lambda z: stack_predict(drift_fields, z),
                        lambda z: stack_predict(diffusion_fields, z))
    grid = make_grid(cfg)
    grid_teacher = field_np(teacher.drift, grid)
    grid_symbolic = stack_predict(drift_fields, grid)
    eps = float(cfg["distill"]["partial_eps"])
    teacher_parts = signature_partials(
        partial_tensor(lambda z: field_np(teacher.drift, z), grid, eps), graph)
    symbolic_parts = signature_partials(
        partial_tensor(lambda z: stack_predict(drift_fields, z), grid, eps), graph)
    truth_on_test = truth_drift(x_test, cfg)
    linear = np.asarray(cfg["truth"]["linear"], dtype=np.float64)
    linear_on_test = x_test @ linear.T
    cubic_on_test = np.zeros_like(x_test)
    cubic_on_test[:, 0] = float(cfg["truth"]["cubic_x"]) * x_test[:, 0] ** 3
    cubic_on_test[:, 2] = float(cfg["truth"]["cubic_z"]) * x_test[:, 2] ** 3
    sigma = np.asarray(cfg["truth"]["sigma"], dtype=np.float64)
    truth_g = np.broadcast_to(sigma, diff_test.shape)
    step_dt = float(data["dt_sim"])
    step_ratio = float(rms(drift_test) * step_dt / max(rms(diff_test) * math.sqrt(step_dt), 1e-8))
    spectrum = library_spectrum(x_train, int(cfg["distill"]["vif_degree"]))
    metrics: dict[str, Any] = {
        "drift": drift_rows,
        "diffusion": diffusion_rows,
        "diffusion_nonpositive_fraction": float(np.mean(np.min(sym_diff, axis=1) <= 0.0)),
        "vif": vif_max(x_train, int(cfg["distill"]["vif_degree"])),
        "library_rank": spectrum["rank"],
        "library_condition": spectrum["condition"],
        "library_terms": spectrum["terms"],
        "fail_teacher": teacher_sim["failure_rate"],
        "fail_symbolic": symbolic_sim["failure_rate"],
        "path_rmse": paired["rmse"],
        "path_retained": paired["retained_fraction"],
        "fail_classes_teacher": class_counts(teacher_sim["failure_class"]),
        "fail_classes_symbolic": class_counts(symbolic_sim["failure_class"]),
        "law_floor": mean_path_rmse(law_teacher["states"], law_floor["states"],
                                    law_teacher["failed"], law_floor["failed"]),
        "law_symbolic": mean_path_rmse(law_teacher["states"], law_symbolic["states"],
                                       law_teacher["failed"], law_symbolic["failed"]),
        "law_fail_teacher": law_teacher["failure_rate"],
        "law_fail_symbolic": law_symbolic["failure_rate"],
        "law_fail_floor": law_floor["failure_rate"],
        "grid_nrmse": nrmse(grid_teacher, grid_symbolic),
        "grid_residual_rms": rms(grid_symbolic - grid_teacher),
        "path_residual_rms": rms(sym_drift - drift_test),
        "teacher_nonparent_rms": teacher_parts["nonparent_rms"],
        "teacher_parent_rms": teacher_parts["parent_rms"],
        "symbolic_nonparent_rms": symbolic_parts["nonparent_rms"],
        "symbolic_parent_rms": symbolic_parts["parent_rms"],
        "symbolic_relative_nonparent": float(symbolic_parts["nonparent_rms"]
                                             / max(symbolic_parts["parent_rms"], 1e-8)),
        "full_library_nrmse": float(np.mean([row["nrmse"] for row in drift_rows])),
        "parent_library_nrmse": float(np.mean([row["parent_nrmse"] for row in drift_rows])),
        "teacher_vs_truth_drift": nrmse(truth_on_test, drift_test),
        "symbolic_vs_truth_drift": nrmse(truth_on_test, sym_drift),
        "teacher_vs_truth_diffusion": nrmse(truth_g, diff_test),
        "symbolic_vs_truth_diffusion": nrmse(truth_g, sym_diff),
        "step_ratio": step_ratio,
        "test_drift_rms": rms(drift_test),
        "test_diffusion_rms": rms(diff_test),
        "truth_linear_rms": rms(linear_on_test),
        "truth_cubic_rms": rms(cubic_on_test),
        "parent_vif": [vif_max(x_train[:, parent_columns(graph, j)], 3) for j in range(3)],
    }
    metrics["amplified"] = bool(metrics["grid_residual_rms"]
                                > float(cfg["hypotheses"]["H6_off_manifold"]["amplification_factor"])
                                * metrics["path_residual_rms"])
    metrics["verdicts"] = decide(cfg, metrics)
    figure = {"teacher_states": teacher_sim["states"][:12],
              "symbolic_states": symbolic_sim["states"][:12],
              "grid": grid, "grid_teacher": grid_teacher, "grid_symbolic": grid_symbolic}
    expanded = {
        "f1": expand_to_original(drift_fields[0]),
        "f2": expand_to_original(drift_fields[1]),
        "f3": expand_to_original(drift_fields[2]),
        "g1": expand_to_original(diffusion_fields[0]),
        "g2": expand_to_original(diffusion_fields[1]),
        "g3": expand_to_original(diffusion_fields[2]),
    }
    return {"metrics": metrics, "figure": figure, "expanded": expanded}


def make_figure(figure: dict[str, Any], path: Path) -> None:
    """Paired paths and a z=0 slice of the teacher drift. Not used for selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    for idx in range(figure["teacher_states"].shape[0]):
        teacher = figure["teacher_states"][idx]
        symbolic = figure["symbolic_states"][idx]
        if np.isfinite(teacher).all():
            ax.plot(teacher[:, 0], teacher[:, 1], color="C0", alpha=0.45, lw=1.0)
        if np.isfinite(symbolic).all():
            ax.plot(symbolic[:, 0], symbolic[:, 1], color="C1", alpha=0.45, lw=1.0, ls="--")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("paired paths")
    grid = figure["grid"]
    mask = np.isclose(grid[:, 2], 0.0)
    sc = axes[1].scatter(grid[mask, 0], grid[mask, 1], c=figure["grid_teacher"][mask, 0],
                         cmap="coolwarm", s=42)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_title(r"teacher $f_1$ at $z=0$")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_tex(expanded: dict[str, list[dict[tuple[int, ...], float]]]) -> str:
    """TeX for the full-library symbolic drift and diffusion."""
    parts = [
        "% Day 21. Full power library in (x, y, z). Parent restrictions are not in this file.",
        tex_equations(expanded["f1"], (r"\hat f_1(x,y,z)",), VAR_NAMES),
        tex_equations(expanded["f2"], (r"\hat f_2(x,y,z)",), VAR_NAMES),
        tex_equations(expanded["f3"], (r"\hat f_3(x,y,z)",), VAR_NAMES),
        tex_equations(expanded["g1"], (r"\hat g_1(x,y,z)",), VAR_NAMES),
        tex_equations(expanded["g2"], (r"\hat g_2(x,y,z)",), VAR_NAMES),
        tex_equations(expanded["g3"], (r"\hat g_3(x,y,z)",), VAR_NAMES),
    ]
    return "\n".join(parts)


def main() -> None:
    """Train the gated fields, distill them, and write the one-shot test."""
    started = time.perf_counter()
    cfg = load_config()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    trained = train_teacher(cfg)
    teacher = trained["model"]
    print(f"train loss {trained['loss'][0]:.6g} -> {trained['loss'][-1]:.6g} "
          f"transitions={trained['n_transitions']} path_abs_max={trained['path_abs_max']:.4g}",
          flush=True)
    train_paths = generate_truth(cfg, int(cfg["data"]["n_series"]), int(cfg["seeds"]["data"]))
    current_np, nxt_np = transition_pairs(train_paths)
    dt_obs = float(cfg["data"]["dt_obs"])
    teacher_drift = field_np(teacher.drift, current_np)
    teacher_diff = field_np(teacher.diffusion, current_np)
    nll_teacher = shifted_gaussian_nll(current_np, nxt_np, teacher_drift, teacher_diff, dt_obs)
    nll_truth = shifted_gaussian_nll(current_np, nxt_np, truth_drift(current_np, cfg),
                                     truth_diffusion(current_np, cfg), dt_obs)
    nll_zero = shifted_gaussian_nll(current_np, nxt_np, np.zeros_like(current_np),
                                   truth_diffusion(current_np, cfg), dt_obs)
    data = cfg["data"]
    rolled = rollout_queries(teacher, cfg, int(data["n_rollout_pool"]), int(cfg["seeds"]["queries"]))
    print(f"query trajectories finite={rolled['n_finite']} failure_rate={rolled['failure_rate']:.4g}",
          flush=True)
    masks = split_owners(rolled["owners"], int(data["n_train"]), int(data["n_val"]), int(data["n_test"]))
    states = rolled["states"]
    x_train, x_val = states[masks["train"]], states[masks["val"]]
    drift_train = field_np(teacher.drift, x_train)
    drift_val = field_np(teacher.drift, x_val)
    diff_train = field_np(teacher.diffusion, x_train)
    diff_val = field_np(teacher.diffusion, x_val)
    graph = truth_graph(cfg)
    drift_fields: list[PolynomialField] = []
    diffusion_fields: list[PolynomialField] = []
    parent_fields: list[PolynomialField] = []
    meta: dict[str, Any] = {}
    for j, name in enumerate(("f1", "f2", "f3")):
        field, info = select_field(x_train, drift_train[:, j], x_val, drift_val[:, j], cfg, name)
        drift_fields.append(field)
        meta[name] = info
        cols = parent_columns(graph, j)
        parent, parent_info = select_field(x_train[:, cols], drift_train[:, j], x_val[:, cols],
                                           drift_val[:, j], cfg, f"parent_{name}")
        parent_fields.append(parent)
        meta[f"parent_{name}"] = parent_info
        print(f"selected {name} degree={info['degree']} threshold={info['threshold']} "
              f"val={info['score']:.4g} active={info['active']}", flush=True)
    for j, name in enumerate(("g1", "g2", "g3")):
        field, info = select_field(x_train, diff_train[:, j], x_val, diff_val[:, j], cfg, name)
        diffusion_fields.append(field)
        meta[name] = info
        print(f"selected {name} degree={info['degree']} threshold={info['threshold']} "
              f"val={info['score']:.4g} active={info['active']}", flush=True)
    frequencies = {}
    for j, name in enumerate(("f1", "f2", "f3", "g1", "g2", "g3")):
        target = drift_train[:, j] if name.startswith("f") else diff_train[:, j % 3]
        frequencies[name] = bootstrap_frequency(
            x_train, target, int(meta[name]["degree"]), float(meta[name]["threshold"]),
            float(cfg["distill"]["ridge"]), int(cfg["distill"]["bootstrap"]),
            int(cfg["seeds"]["bootstrap"]) + j)
    outcome = evaluate(cfg, teacher, states, masks, drift_fields, diffusion_fields, parent_fields, meta)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_weights(teacher, OUT_DIR / "teacher_scotch_weights.json")
    for name, field in zip(("f1", "f2", "f3"), drift_fields):
        field.save(OUT_DIR / f"symbolic_{name}.json")
    for name, field in zip(("g1", "g2", "g3"), diffusion_fields):
        field.save(OUT_DIR / f"symbolic_{name}.json")
    for name, field in zip(("f1", "f2", "f3"), parent_fields):
        field.save(OUT_DIR / f"symbolic_parent_{name}.json")
    tex = render_tex(outcome["expanded"])
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (REPORT_DIR / "symbolic_sde.tex").write_text(tex)
    expanded_dump = {name: [{str(key): value for key, value in poly.items()} for poly in polys]
                     for name, polys in outcome["expanded"].items()}
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
        "shifted_nll_teacher": nll_teacher,
        "shifted_nll_truth_euler": nll_truth,
        "shifted_nll_zero_drift": nll_zero,
        "shifted_nll_omits_half_log_2pi": True,
        "train_transitions": trained["n_transitions"],
        "train_path_abs_max": trained["path_abs_max"],
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
