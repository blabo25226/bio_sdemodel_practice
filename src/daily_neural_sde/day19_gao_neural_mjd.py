"""Day 19: Gao et al. (NeurIPS 2025) Neural MJD -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day19.json``):

1. Data: Appendix D.2 scalar Merton jump diffusion. One window per path,
   trajectory-disjoint. The teacher never sees validation or test paths.
2. Teacher: an MLP maps the nine context log-returns and the future-time
   coordinate to ``(mu, sigma, lambda, nu, gamma)``. Training is the
   paper's truncated jump likelihood plus a price-mean penalty (Algorithm 1),
   with ``kappa = 5``. A second MLP with ``lambda = 0`` is the Neural BS
   ablation and is not used to choose the polynomial.
3. Distillation: direct queries of those five neural rates. The library is
   powers of the context returns and future time only. Validation chooses
   degree and threshold. Test is scored once.

Run from the repository root:
``python -m src.daily_neural_sde.day19_gao_neural_mjd``

Outputs land in ``outputs/daily_neural_sde/day19/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day19.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day19"
REPORT_DIR = ROOT / "reports" / "daily_neural_sde" / "day19_gao_neural_mjd"
DTYPE = torch.float64
NAMES = ("mu", "sigma", "lam", "nu", "gamma")
FieldFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-19 config."""
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


def compensator(nu: np.ndarray | float, gamma: np.ndarray | float) -> np.ndarray | float:
    """``E[Y - 1] = exp(nu + gamma^2 / 2) - 1`` for lognormal jump ratios."""
    return np.exp(np.asarray(nu, dtype=np.float64) + 0.5 * np.asarray(gamma, dtype=np.float64) ** 2) - 1.0


def inv_softplus(value: float) -> float:
    """Inverse of ``log(1 + exp(x))`` for a positive target."""
    return float(math.log(math.expm1(value)))


def uniform_constant_floor(low: float, high: float) -> float:
    """NRMSE of predicting the mean of a uniform law. Shape-free."""
    mean = 0.5 * (low + high)
    var = (high - low) ** 2 / 12.0
    return float(math.sqrt(var / (mean ** 2 + var)))


def sample_parameters(n: int, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw latent MJD rates. Each array has shape ``(n,)``."""
    data = cfg["data"]

    def draw(key: str) -> np.ndarray:
        low, high = data[key]
        return rng.uniform(float(low), float(high), size=n)

    return {"mu": draw("mu_range"), "sigma": draw("sigma_range"), "lam": draw("lambda_range"),
            "nu": draw("nu_range"), "gamma": draw("gamma_range")}


def simulate_log_paths(params: dict[str, np.ndarray], n_steps: int, dt: float,
                       rng: np.random.Generator, jump_slots: int) -> dict[str, np.ndarray]:
    """Exact piecewise Euler-Poisson step of the log Merton SDE.

    Returns ``log_s`` of shape ``(n, n_steps + 1)`` starting at 0, plus the
    noise that generated it. Jump clocks are unit exponentials.
    """
    n = len(params["mu"])
    log_s = np.zeros((n, n_steps + 1), dtype=np.float64)
    z = rng.standard_normal((n, n_steps))
    exp_unit = rng.exponential(1.0, size=(n, n_steps, jump_slots))
    gauss = rng.standard_normal((n, n_steps, jump_slots))
    mu, sigma = params["mu"], params["sigma"]
    lam, nu, gamma = params["lam"], params["nu"], params["gamma"]
    k = compensator(nu, gamma)
    for step in range(n_steps):
        jump, _cap = jump_sum(exp_unit[:, step], gauss[:, step], lam, nu, gamma, dt)
        log_s[:, step + 1] = (log_s[:, step]
                              + (mu - lam * k - 0.5 * sigma ** 2) * dt
                              + sigma * math.sqrt(dt) * z[:, step]
                              + jump)
    return {"log_s": log_s, "z": z, "exp": exp_unit, "gauss": gauss}


def jump_sum(exp_unit: np.ndarray, gauss: np.ndarray, lam: np.ndarray, nu: np.ndarray,
             gamma: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Sum of jumps in an interval. ``exp_unit`` and ``gauss`` are ``(n, slots)``.

    The cap flag is true when every slot lands inside the interval, so the
    Poisson count may have been truncated.
    """
    lam = np.asarray(lam, dtype=np.float64)
    arrivals = np.cumsum(exp_unit, axis=1) / np.maximum(lam, 1e-15)[:, None]
    within = (arrivals <= dt) & (lam[:, None] > 0.0)
    sizes = nu[:, None] + gamma[:, None] * gauss
    return (sizes * within).sum(axis=1), within[:, -1].copy()


def generate_windows(cfg: dict[str, Any], rng: np.random.Generator, n: int) -> dict[str, Any]:
    """One context/future window per path. Paths that leave the log radius are redrawn.

    Context log-returns have shape ``(n, 9)`` when there are 10 context prices.
    Future log-prices are absolute, with the path started at log-price 0.
    """
    data = cfg["data"]
    n_context = int(data["context_prices"])
    n_future = int(data["future_steps"])
    n_steps = n_context - 1 + n_future
    dt = float(data["dt"])
    radius = float(data["log_radius"])
    slots = int(data["jump_slots"])
    kept_log: list[np.ndarray] = []
    kept_params: list[dict[str, float]] = []
    discarded = 0
    rounds = 0
    while len(kept_log) < n and rounds < int(data["max_resample_rounds"]) * n:
        rounds += 1
        params = sample_parameters(1, cfg, rng)
        out = simulate_log_paths({key: value for key, value in params.items()}, n_steps, dt, rng, slots)
        path = out["log_s"][0]
        if not np.isfinite(path).all() or float(np.max(np.abs(path))) > radius:
            discarded += 1
            continue
        kept_log.append(path)
        kept_params.append({key: float(value[0]) for key, value in params.items()})
    if len(kept_log) < n:
        raise RuntimeError(f"only kept {len(kept_log)} of {n} MJD windows")
    log_s = np.stack(kept_log, axis=0)
    stacked = {key: np.array([row[key] for row in kept_params], dtype=np.float64) for key in NAMES}
    returns = np.diff(log_s[:, :n_context], axis=1)
    present = log_s[:, n_context - 1]
    future = log_s[:, n_context:]
    return {"returns": returns, "present": present, "future": future, "log_s": log_s,
            "params": stacked, "discarded": discarded, "dt": dt}


def split_windows(windows: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Cut a generated pool into train, validation, and test without reshuffling."""
    data = cfg["data"]
    counts = {"train": int(data["n_train"]), "val": int(data["n_val"]), "test": int(data["n_test"])}
    frames: dict[str, dict[str, Any]] = {}
    start = 0
    for name, count in counts.items():
        stop = start + count
        frames[name] = {
            "returns": windows["returns"][start:stop],
            "present": windows["present"][start:stop],
            "future": windows["future"][start:stop],
            "params": {key: value[start:stop] for key, value in windows["params"].items()},
        }
        start = stop
    return frames


def future_times(n_steps: int) -> np.ndarray:
    """Normalized future coordinates ``1/T, ..., 1``. Shape ``(T,)``."""
    return (np.arange(1, n_steps + 1, dtype=np.float64)) / float(n_steps)


def design_matrix(returns: np.ndarray, n_steps: int) -> np.ndarray:
    """Repeat each context across future times. Shape ``(n * T, 10)``."""
    returns = np.asarray(returns, dtype=np.float64)
    times = future_times(n_steps)
    tiled_returns = np.repeat(returns, n_steps, axis=0)
    tiled_time = np.tile(times, len(returns))[:, None]
    return np.concatenate([tiled_returns, tiled_time], axis=1)


def repeat_targets(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Broadcast a per-path rate to every future bin. Shape ``(n * T,)``."""
    return np.repeat(np.asarray(values, dtype=np.float64), n_steps)


class RateMLP(nn.Module):
    """Map ``(returns, time)`` to MJD rates. Diffusion and intensity are positive.

    Input shape is ``(batch, 10)``. Each rate is ``(batch,)``.
    ``jumps=False`` is the Neural BS ablation: lambda, nu, and gamma are fixed.
    """

    def __init__(self, width: int, depth: int, jumps: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        incoming = 10
        for _ in range(depth):
            layers.append(nn.Linear(incoming, width))
            layers.append(nn.SiLU())
            incoming = width
        self.out_dim = 5 if jumps else 2
        layers.append(nn.Linear(incoming, self.out_dim))
        self.net = nn.Sequential(*layers)
        self.jumps = jumps
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("expected a linear head")
        nn.init.normal_(last.weight, std=0.01)
        with torch.no_grad():
            last.bias.zero_()
            last.bias[0] = 0.3
            last.bias[1] = inv_softplus(0.3)
            if jumps:
                last.bias[2] = inv_softplus(6.5)
                last.bias[3] = 0.0
                last.bias[4] = inv_softplus(0.75)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = self.net(features)
        mu = raw[:, 0]
        sigma = torch.nn.functional.softplus(raw[:, 1])
        if not self.jumps:
            zeros = torch.zeros_like(mu)
            ones = torch.ones_like(mu)
            return mu, sigma, zeros, zeros, ones
        lam = torch.nn.functional.softplus(raw[:, 2])
        nu = raw[:, 3]
        gamma = torch.nn.functional.softplus(raw[:, 4])
        return mu, sigma, lam, nu, gamma


def transition_log_prob(log_cond: torch.Tensor, log_next: torch.Tensor, mu: torch.Tensor,
                        sigma: torch.Tensor, lam: torch.Tensor, nu: torch.Tensor,
                        gamma: torch.Tensor, dt: float, kappa: int) -> torch.Tensor:
    """Truncated Merton transition log-density. Inputs share shape ``(batch,)``.

    The sum stops at ``kappa`` jumps, as in the paper's equation (12) with
    Theorem 4.1. The ``n = 0`` term does not evaluate ``log(lambda)``.
    """
    steps = torch.arange(0, kappa + 1, device=log_cond.device, dtype=log_cond.dtype)
    k = torch.exp(nu + 0.5 * gamma ** 2) - 1.0
    rate = lam * dt
    n = steps.view(*([1] * rate.ndim), -1)
    log_rate = torch.log(rate.clamp_min(1e-30))
    n_log = torch.where(n == 0, torch.zeros_like(log_rate)[..., None], n * log_rate[..., None])
    log_poisson = -rate[..., None] + n_log - torch.lgamma(n + 1.0)
    mean = (log_cond + (mu - lam * k - 0.5 * sigma ** 2) * dt)[..., None] + n * nu[..., None]
    var = (sigma ** 2 * dt)[..., None] + n * (gamma ** 2)[..., None]
    var = var.clamp_min(1e-12)
    log_normal = -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + (log_next[..., None] - mean) ** 2 / var)
    return torch.logsumexp(log_poisson + log_normal, dim=-1)


def batch_loss(model: RateMLP, returns: torch.Tensor, present: torch.Tensor, future: torch.Tensor,
               dt: float, omega: float, kappa: int, log_clip: float) -> torch.Tensor:
    """Algorithm 1 loss on one batch. ``future`` has shape ``(batch, T)``.

    The transition conditions on the conditional mean path, not on the
    observed price. The penalty compares price ratios, clipped in log space.
    """
    batch, horizon = future.shape
    times = torch.arange(1, horizon + 1, device=future.device, dtype=future.dtype) / horizon
    features = torch.cat([
        returns[:, None, :].expand(batch, horizon, returns.shape[1]),
        times.view(1, horizon, 1).expand(batch, horizon, 1),
    ], dim=-1)
    mu, sigma, lam, nu, gamma = model(features.reshape(batch * horizon, -1))
    mu = mu.view(batch, horizon)
    sigma = sigma.view(batch, horizon)
    lam = lam.view(batch, horizon)
    nu = nu.view(batch, horizon)
    gamma = gamma.view(batch, horizon)
    log_hat = present[:, None] + dt * torch.cumsum(mu, dim=1)
    log_cond = torch.cat([present[:, None], log_hat[:, :-1]], dim=1)
    nll = -transition_log_prob(log_cond.reshape(-1), future.reshape(-1), mu.reshape(-1),
                               sigma.reshape(-1), lam.reshape(-1), nu.reshape(-1),
                               gamma.reshape(-1), dt, kappa)
    observed = future.clamp(-log_clip, log_clip)
    predicted = log_hat.clamp(-log_clip, log_clip)
    penalty = (torch.exp(observed) - torch.exp(predicted)) ** 2
    return (nll + omega * penalty.reshape(-1)).mean()


def train_model(frame: dict[str, Any], cfg: dict[str, Any], jumps: bool, seed: int,
                label: str) -> dict[str, Any]:
    """Fit one MLP on the training window only. Epoch count is frozen."""
    torch.manual_seed(seed)
    model_cfg = cfg["model"]
    model = RateMLP(int(model_cfg["width"]), int(model_cfg["depth"]), jumps).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["lr"]))
    returns = torch.tensor(frame["returns"], dtype=DTYPE)
    present = torch.tensor(frame["present"], dtype=DTYPE)
    future = torch.tensor(frame["future"], dtype=DTYPE)
    dt = float(cfg["data"]["dt"])
    omega = float(model_cfg["omega"])
    kappa = int(model_cfg["kappa"])
    log_clip = float(model_cfg["regularizer_log_clip"])
    batch_size = int(model_cfg["batch_size"])
    history: list[float] = []
    generator = torch.Generator()
    generator.manual_seed(seed)
    for epoch in range(int(model_cfg["epochs"])):
        order = torch.randperm(len(returns), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(returns), batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, returns[index], present[index], future[index], dt, omega, kappa, log_clip)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{label} produced a non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg["grad_clip"]))
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        history.append(total / max(seen, 1))
        if epoch % 20 == 0 or epoch + 1 == int(model_cfg["epochs"]):
            print(f"{label} epoch={epoch} loss={history[-1]:.6g}", flush=True)
    model.eval()
    return {"model": model, "loss": history, "jumps": jumps}


def query_rates(model: RateMLP, features: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate the five rates. ``features`` is ``(n, 10)`` and each rate is ``(n,)``."""
    with torch.no_grad():
        tensor = torch.tensor(np.asarray(features, dtype=np.float64), dtype=DTYPE)
        values = model(tensor)
    return {name: value.detach().cpu().numpy().astype(np.float64) for name, value in zip(NAMES, values)}


def select_field(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                 cfg: dict[str, Any], name: str) -> tuple[PolynomialField, dict[str, Any]]:
    """Validation NRMSE, then smaller degree, then smaller threshold."""
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


def field_pack(prediction: np.ndarray, teacher: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """NRMSE and residual RMS for one scalar rate."""
    return {"sym_teacher_nrmse": nrmse(teacher, prediction),
            "teacher_truth_nrmse": nrmse(truth, teacher),
            "sym_truth_nrmse": nrmse(truth, prediction),
            "sym_teacher_rms": rms(prediction - teacher),
            "teacher_truth_rms": rms(teacher - truth),
            "sym_truth_rms": rms(prediction - truth),
            "teacher_rms": rms(teacher),
            "truth_rms": rms(truth)}


def relative_rms(reference: np.ndarray, changed: np.ndarray) -> float:
    """``||changed - reference|| / ||reference||``."""
    return float(rms(changed - reference) / max(rms(reference), 1e-8))


def shuffle_returns(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the nine return columns independently on each row. Time stays."""
    out = np.array(features, dtype=np.float64, copy=True)
    for row in out:
        rng.shuffle(row[:-1])
    return out


def reverse_time(features: np.ndarray) -> np.ndarray:
    """Send ``u`` to ``1.1 - u`` so the ten-bin grid reverses."""
    out = np.array(features, dtype=np.float64, copy=True)
    out[:, -1] = 1.1 - out[:, -1]
    return out


def cross_partial_rms(fn: FieldFn, features: np.ndarray, eps: float = 1e-3) -> float:
    """RMS of the mixed partial of a scalar field in the first two inputs."""
    def partial(points: np.ndarray) -> np.ndarray:
        plus, minus = np.array(points, copy=True), np.array(points, copy=True)
        plus[:, 0] += eps
        minus[:, 0] -= eps
        return (fn(plus) - fn(minus)) / (2.0 * eps)

    plus, minus = np.array(features, copy=True), np.array(features, copy=True)
    plus[:, 1] += eps
    minus[:, 1] -= eps
    return rms((partial(plus) - partial(minus)) / (2.0 * eps))


def gradient_rms(fn: FieldFn, features: np.ndarray, eps: float = 1e-3) -> float:
    """RMS of every first partial of a scalar field."""
    grads = []
    for axis in range(features.shape[1]):
        plus, minus = np.array(features, copy=True), np.array(features, copy=True)
        plus[:, axis] += eps
        minus[:, axis] -= eps
        grads.append((fn(plus) - fn(minus)) / (2.0 * eps))
    return rms(np.stack(grads, axis=1))


def make_noise(n: int, n_steps: int, slots: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shared Wiener increments and unit exponential clocks. Shapes ``(n, steps, ...)``."""
    return {"z": rng.standard_normal((n, n_steps)),
            "exp": rng.exponential(1.0, size=(n, n_steps, slots)),
            "gauss": rng.standard_normal((n, n_steps, slots))}


def simulate_forecast(rate_fn: Callable[[int], dict[str, np.ndarray]], present: np.ndarray,
                      noise: dict[str, np.ndarray], dt: float, radius: float) -> dict[str, Any]:
    """Coupled log-price forecast. Failed rows stay NaN and keep their class.

    ``rate_fn(step)`` returns per-path rates of shape ``(n,)``.
    """
    n, n_steps = noise["z"].shape
    log_s = np.full((n, n_steps + 1), np.nan, dtype=np.float64)
    log_s[:, 0] = np.asarray(present, dtype=np.float64)
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    sqrt_dt = math.sqrt(dt)
    for step in range(n_steps):
        active = ~failed
        if not active.any():
            break
        rates = rate_fn(step)
        mu = np.asarray(rates["mu"], dtype=np.float64)
        sigma = np.asarray(rates["sigma"], dtype=np.float64)
        lam = np.asarray(rates["lam"], dtype=np.float64)
        nu = np.asarray(rates["nu"], dtype=np.float64)
        gamma = np.asarray(rates["gamma"], dtype=np.float64)
        bad_coeff = active & ((sigma <= 0.0) | (lam < 0.0) | (gamma <= 0.0) | ~np.isfinite(mu))
        for index in np.flatnonzero(bad_coeff):
            failure_class[index] = "nonpositive_coefficient"
        failed = failed | bad_coeff
        active = ~failed
        if not active.any():
            break
        jump, cap = jump_sum(noise["exp"][active, step], noise["gauss"][active, step],
                             lam[active], nu[active], gamma[active], dt)
        for local, index in enumerate(np.flatnonzero(active)):
            if cap[local]:
                failure_class[index] = "jump_cap"
                failed[index] = True
        active = ~failed
        if not active.any():
            break
        jump, _cap = jump_sum(noise["exp"][active, step], noise["gauss"][active, step],
                              lam[active], nu[active], gamma[active], dt)
        k = compensator(nu[active], gamma[active])
        log_s[active, step + 1] = (log_s[active, step]
                                   + (mu[active] - lam[active] * k - 0.5 * sigma[active] ** 2) * dt
                                   + sigma[active] * sqrt_dt * noise["z"][active, step]
                                   + jump)
        bad_nonfinite = ~np.isfinite(log_s[:, step + 1])
        bad_radius = np.zeros(n, dtype=bool)
        finite = np.isfinite(log_s[:, step + 1])
        bad_radius[finite] = np.abs(log_s[finite, step + 1]) > radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for index in np.flatnonzero(newly):
            failure_class[index] = "non_finite" if bad_nonfinite[index] else "outside_radius"
        failed = failed | newly
        log_s[failed, step + 1] = np.nan
    return {"log_s": log_s, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed))}


def failure_summary(sim: dict[str, Any]) -> dict[str, int]:
    """Count failure classes. Survivors are omitted."""
    counts: dict[str, int] = {}
    for label in sim["failure_class"]:
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    return counts


def constant_rates(params: dict[str, np.ndarray]) -> Callable[[int], dict[str, np.ndarray]]:
    """Pathwise latent rates, constant across future bins."""
    def query(_step: int) -> dict[str, np.ndarray]:
        return params
    return query


def scheduled_rates(fields: dict[str, PolynomialField], returns: np.ndarray,
                    n_steps: int) -> Callable[[int], dict[str, np.ndarray]]:
    """Polynomial rates at the future coordinate of this bin."""
    def query(step: int) -> dict[str, np.ndarray]:
        time = np.full((len(returns), 1), (step + 1) / float(n_steps))
        features = np.concatenate([returns, time], axis=1)
        return {name: field.predict(features)[:, 0] for name, field in fields.items()}
    return query


def network_rates(model: RateMLP, returns: np.ndarray, n_steps: int) -> Callable[[int], dict[str, np.ndarray]]:
    """Neural rates at the future coordinate of this bin."""
    def query(step: int) -> dict[str, np.ndarray]:
        time = np.full((len(returns), 1), (step + 1) / float(n_steps))
        features = np.concatenate([returns, time], axis=1)
        return query_rates(model, features)
    return query


def path_rmse(left: np.ndarray, right: np.ndarray, failed: np.ndarray) -> float:
    """RMS of log-price paths on trajectories both models finished."""
    keep = ~failed
    if not keep.any():
        return float("nan")
    return rms(left[keep] - right[keep])


def mean_path_rmse(left: np.ndarray, right: np.ndarray, failed: np.ndarray) -> float:
    """RMS across time of the difference of mean log-price paths."""
    keep = ~failed
    if keep.sum() < 2:
        return float("nan")
    return rms(left[keep].mean(axis=0) - right[keep].mean(axis=0))


def terminal_var_relative(left: np.ndarray, right: np.ndarray, failed: np.ndarray) -> float:
    """Relative absolute error of the terminal log-price variance."""
    keep = ~failed
    if keep.sum() < 2:
        return float("nan")
    base = float(np.var(right[keep, -1]))
    return abs(float(np.var(left[keep, -1])) - base) / max(base, 1e-8)


def slice_grid(cfg: dict[str, Any], n_steps: int) -> np.ndarray:
    """``r_1`` by ``u`` slice. The other eight returns are zero. Not used for selection."""
    distill = cfg["distill"]
    r_values = np.linspace(float(distill["slice_r_low"]), float(distill["slice_r_high"]), int(distill["slice_size"]))
    times = future_times(n_steps)
    rows = []
    for time in times:
        for value in r_values:
            row = np.zeros(10, dtype=np.float64)
            row[0] = value
            row[-1] = time
            rows.append(row)
    return np.stack(rows, axis=0)


def step_ratio(mu: np.ndarray, sigma: np.ndarray, dt: float) -> float:
    """``|mu| dt / (sigma sqrt(dt))`` averaged over a sample."""
    return float(np.mean(np.abs(mu) * dt / (np.maximum(sigma, 1e-8) * math.sqrt(dt))))


def save_weights(model: RateMLP, path: Path) -> None:
    """JSON weights. No pickle."""
    payload = {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
    path.write_text(json.dumps({"jumps": model.jumps, "state": payload}))


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. Thresholds are read from ``cfg`` and not modified."""
    hyp = cfg["hypotheses"]
    h1 = hyp["H1_fields"]
    mu, sigma = metrics["mu_test"], metrics["sigma_test"]
    distill = (mu["sym_teacher_nrmse"] <= h1["symbolic_vs_teacher_test_nrmse_max"]
               and sigma["sym_teacher_nrmse"] <= h1["symbolic_vs_teacher_test_nrmse_max"])
    identify = (mu["teacher_truth_nrmse"] <= h1["teacher_vs_truth_test_nrmse_max"]
                and sigma["teacher_truth_nrmse"] <= h1["teacher_vs_truth_test_nrmse_max"])
    symbolic_truth = (mu["sym_truth_nrmse"] <= h1["symbolic_vs_truth_test_nrmse_max"]
                      and sigma["sym_truth_nrmse"] <= h1["symbolic_vs_truth_test_nrmse_max"])
    h2 = hyp["H2_diffusion"]
    verdicts: dict[str, Any] = {
        "H1_distill": _pass(distill),
        "H1_identify": _pass(identify),
        "H1_symbolic_truth": _pass(symbolic_truth),
        "H1": _pass(distill and identify and symbolic_truth),
        "H2_distill": _pass(metrics["sigma_test_nrmse"] <= h2["symbolic_vs_teacher_test_nrmse_max"]
                            and metrics["sigma_nonpositive_fraction"] <= h2["fraction_nonpositive_max"]),
        "H2_mean": _pass(metrics["sigma_mean_relative_error"] <= h2["mean_relative_error_max"]),
        "H2_local_shadow": bool(metrics["sigma_slice_nrmse"]
                                > h2["grid_over_ontraj_factor"] * max(metrics["sigma_test_nrmse"], 1e-8)),
        "H3": ("indeterminate" if metrics["vif_degree3"] > hyp["H3_collinearity_gate"]["vif_max"]
               else "pass"),
        "H3_closed_form": False,
    }
    verdicts["H4"] = _simulation_verdict(hyp["H4_simulation"], metrics)
    verdicts["H5"] = _law_verdict(hyp["H5_law"], metrics)
    h6 = hyp["H6_invariance"]
    verdicts["H6"] = _pass(
        metrics["shuffle_teacher"] <= h6["shuffle_relative_rms_max"]
        and metrics["shuffle_symbolic"] <= h6["shuffle_relative_rms_max"]
        and metrics["time_teacher"] <= h6["time_reversal_relative_rms_max"]
        and metrics["time_symbolic"] <= h6["time_reversal_relative_rms_max"]
        and metrics["slice_sym_teacher_nrmse"] <= h6["slice_sym_teacher_nrmse_max"])
    verdicts["H6_amplified"] = bool(
        metrics["slice_sym_teacher_rms"] > float(cfg["distill"]["amplification_factor"])
        * max(metrics["mu_test"]["sym_teacher_rms"], 1e-12))
    h7 = hyp["H7_discretization"]
    verdicts["H7"] = _pass(math.isfinite(metrics["disc_rmse"])
                           and metrics["disc_rmse"] <= h7["terminal_rmse_max"]
                           and metrics["disc_retained"] >= h7["min_retained_fraction"])
    h8 = hyp["H8_jump_versus_black_scholes"]
    verdicts["H8"] = _pass(metrics["bs_sigma_nrmse"] - sigma["teacher_truth_nrmse"]
                           >= h8["sigma_nrmse_improvement_min"]
                           and sigma["teacher_truth_nrmse"] <= h8["mjd_sigma_nrmse_max"])
    h9 = hyp["H9_cross_partial"]
    verdicts["H9"] = _pass(metrics["cross_ratio_teacher"] <= h9["cross_partial_ratio_max"])
    return verdicts


def _simulation_verdict(gate: dict[str, Any], metrics: dict[str, Any]) -> str:
    worst = max(metrics["fail_truth"], metrics["fail_teacher"], metrics["fail_symbolic"])
    if worst > gate["failure_rate_max"] or abs(metrics["fail_symbolic"] - metrics["fail_teacher"]) > gate["failure_gap_max"]:
        return "indeterminate"
    values = (metrics["paired_sym_teacher"], metrics["paired_teacher_truth"],
              metrics["var_sym_teacher"], metrics["var_teacher_truth"], metrics["field_teacher_truth"])
    if not all(math.isfinite(value) for value in values):
        return "indeterminate"
    ok = (metrics["paired_sym_teacher"] <= gate["path_rmse_symbolic_vs_teacher_max"]
          and metrics["paired_teacher_truth"] <= gate["path_rmse_teacher_vs_truth_max"]
          and metrics["var_sym_teacher"] <= gate["terminal_var_symbolic_vs_teacher_max"]
          and metrics["var_teacher_truth"] <= gate["terminal_var_teacher_vs_truth_max"]
          and metrics["field_teacher_truth"] <= gate["field_teacher_vs_truth_max"])
    return _pass(ok)


def _law_verdict(gate: dict[str, Any], metrics: dict[str, Any]) -> str:
    worst = max(metrics["law_fail_truth"], metrics["law_fail_teacher"], metrics["law_fail_symbolic"])
    if worst > gate["failure_rate_max"]:
        return "indeterminate"
    if (not math.isfinite(metrics["floor_mean"]) or metrics["floor_mean"] > gate["mean_path_rmse_teacher_max"]
            or not math.isfinite(metrics["floor_var"]) or metrics["floor_var"] > gate["terminal_var_teacher_max"]):
        return "indeterminate"
    ok = (metrics["law_mean_teacher"] <= gate["mean_path_rmse_teacher_max"]
          and metrics["law_mean_symbolic"] <= gate["mean_path_rmse_symbolic_max"]
          and metrics["law_var_teacher"] <= gate["terminal_var_teacher_max"]
          and metrics["law_var_symbolic"] <= gate["terminal_var_symbolic_max"])
    return _pass(ok)


def evaluate(cfg: dict[str, Any], frames: dict[str, dict[str, Any]], teacher: RateMLP,
             baseline: RateMLP, fields: dict[str, PolynomialField],
             meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score the frozen teacher once. This is the only test-set pass."""
    dt = float(cfg["data"]["dt"])
    n_steps = int(cfg["data"]["future_steps"])
    radius = float(cfg["data"]["log_radius"])
    slots = int(cfg["data"]["jump_slots"])
    test = frames["test"]
    x_test = design_matrix(test["returns"], n_steps)
    teacher_rates = query_rates(teacher, x_test)
    baseline_rates = query_rates(baseline, x_test)
    symbolic = {name: field.predict(x_test)[:, 0] for name, field in fields.items()}
    truth = {name: repeat_targets(test["params"][name], n_steps) for name in NAMES}
    packs = {name: field_pack(symbolic[name], teacher_rates[name], truth[name]) for name in ("mu", "sigma")}
    oracle = fit_polynomial_field(design_matrix(frames["train"]["returns"], n_steps),
                                  query_rates(teacher, design_matrix(frames["train"]["returns"], n_steps))["mu"][:, None],
                                  3, 0.0, 0.0)
    oracle_sigma = fit_polynomial_field(
        design_matrix(frames["train"]["returns"], n_steps),
        query_rates(teacher, design_matrix(frames["train"]["returns"], n_steps))["sigma"][:, None],
        3, 0.0, 0.0)
    train_mu = frames["train"]["params"]["mu"]
    train_sigma = frames["train"]["params"]["sigma"]
    constant_mu = np.full(len(truth["mu"]), float(train_mu.mean()))
    constant_sigma = np.full(len(truth["sigma"]), float(train_sigma.mean()))
    slice_x = slice_grid(cfg, n_steps)
    slice_teacher = query_rates(teacher, slice_x)
    slice_symbolic = {name: field.predict(slice_x)[:, 0] for name, field in fields.items()}
    rng_shuffle = np.random.default_rng(int(cfg["seeds"]["shuffle"]))
    shuffled = shuffle_returns(x_test, rng_shuffle)
    reversed_time = reverse_time(x_test)
    teacher_mu = teacher_rates["mu"]
    symbolic_mu = symbolic["mu"]
    mu_fn = lambda points, model=teacher: query_rates(model, points)["mu"]  # noqa: E731
    sym_fn = lambda points, field=fields["mu"]: field.predict(points)[:, 0]  # noqa: E731
    noise = make_noise(len(test["returns"]), n_steps, slots, np.random.default_rng(int(cfg["seeds"]["sim"])))
    sim_truth = simulate_forecast(constant_rates(test["params"]), test["present"], noise, dt, radius)
    sim_teacher = simulate_forecast(network_rates(teacher, test["returns"], n_steps), test["present"], noise, dt, radius)
    sim_symbolic = simulate_forecast(scheduled_rates(fields, test["returns"], n_steps), test["present"], noise, dt, radius)
    frozen = network_rates(teacher, test["returns"], n_steps)
    first = frozen(0)

    def frozen_query(_step: int, rates: dict[str, np.ndarray] = first) -> dict[str, np.ndarray]:
        return rates

    sim_coarse = simulate_forecast(frozen_query, test["present"], noise, dt, radius)
    both = ~(sim_teacher["failed"] | sim_coarse["failed"])
    disc = rms(sim_teacher["log_s"][both, -1] - sim_coarse["log_s"][both, -1]) if both.any() else float("nan")
    failed_pair = sim_truth["failed"] | sim_teacher["failed"] | sim_symbolic["failed"]
    law = _law_block(cfg, teacher, fields, dt, radius, slots)
    metrics: dict[str, Any] = {
        "mu_test": packs["mu"],
        "sigma_test": packs["sigma"],
        "sigma_test_nrmse": packs["sigma"]["sym_teacher_nrmse"],
        "sigma_nonpositive_fraction": float(np.mean(symbolic["sigma"] <= 0.0)),
        "sigma_mean_relative_error": abs(float(teacher_rates["sigma"].mean()) - 0.3) / 0.3,
        "teacher_mu_mean": float(teacher_rates["mu"].mean()),
        "teacher_sigma_mean": float(teacher_rates["sigma"].mean()),
        "sigma_slice_nrmse": nrmse(slice_teacher["sigma"], slice_symbolic["sigma"]),
        "vif_degree3": vif_max(design_matrix(frames["train"]["returns"], n_steps), 3),
        "library_degree3": library_spectrum(design_matrix(frames["train"]["returns"], n_steps), 3),
        "vif_degree1": vif_max(design_matrix(frames["train"]["returns"], n_steps), 1),
        "oracle_mu_nrmse": nrmse(teacher_mu, oracle.predict(x_test)[:, 0]),
        "oracle_sigma_nrmse": nrmse(teacher_rates["sigma"], oracle_sigma.predict(x_test)[:, 0]),
        "constant_mu_nrmse": nrmse(truth["mu"], constant_mu),
        "constant_sigma_nrmse": nrmse(truth["sigma"], constant_sigma),
        "shuffle_teacher": relative_rms(teacher_mu, mu_fn(shuffled)),
        "shuffle_symbolic": relative_rms(symbolic_mu, sym_fn(shuffled)),
        "shuffle_truth": 0.0,
        "time_teacher": relative_rms(teacher_mu, mu_fn(reversed_time)),
        "time_symbolic": relative_rms(symbolic_mu, sym_fn(reversed_time)),
        "slice_sym_teacher_nrmse": nrmse(slice_teacher["mu"], slice_symbolic["mu"]),
        "slice_sym_teacher_rms": rms(slice_symbolic["mu"] - slice_teacher["mu"]),
        "cross_teacher": cross_partial_rms(mu_fn, x_test),
        "cross_symbolic": cross_partial_rms(sym_fn, x_test),
        "grad_teacher": gradient_rms(mu_fn, x_test),
        "grad_symbolic": gradient_rms(sym_fn, x_test),
        "fail_truth": sim_truth["failure_rate"],
        "fail_teacher": sim_teacher["failure_rate"],
        "fail_symbolic": sim_symbolic["failure_rate"],
        "failure_classes": {"truth": failure_summary(sim_truth), "teacher": failure_summary(sim_teacher),
                            "symbolic": failure_summary(sim_symbolic)},
        "paired_sym_teacher": path_rmse(sim_symbolic["log_s"], sim_teacher["log_s"], failed_pair),
        "paired_teacher_truth": path_rmse(sim_teacher["log_s"], sim_truth["log_s"], failed_pair),
        "var_sym_teacher": terminal_var_relative(sim_symbolic["log_s"], sim_teacher["log_s"], failed_pair),
        "var_teacher_truth": terminal_var_relative(sim_teacher["log_s"], sim_truth["log_s"], failed_pair),
        "field_teacher_truth": max(packs["mu"]["teacher_truth_nrmse"], packs["sigma"]["teacher_truth_nrmse"]),
        "disc_rmse": disc,
        "disc_retained": float(both.mean()),
        "bs_sigma_nrmse": nrmse(truth["sigma"], baseline_rates["sigma"]),
        "bs_mu_nrmse": nrmse(truth["mu"], baseline_rates["mu"]),
        "step_ratio": step_ratio(test["params"]["mu"], test["params"]["sigma"], dt),
        "selection": {name: {key: value for key, value in row.items() if key != "candidates"}
                      for name, row in meta.items()},
        "active_terms": {name: int(np.count_nonzero(field.coefficients)) for name, field in fields.items()},
    }
    metrics.update(law)
    metrics["cross_ratio_teacher"] = metrics["cross_teacher"] / max(packs["mu"]["teacher_rms"], 1e-8)
    metrics["drift_step_ratio_blind"] = bool(metrics["step_ratio"] < float(cfg["distill"]["step_ratio_blind"]))
    metrics["verdicts"] = decide(cfg, metrics)
    return {"metrics": metrics, "slice_x": slice_x, "slice_teacher_mu": slice_teacher["mu"],
            "slice_symbolic_mu": slice_symbolic["mu"], "teacher_mu_path": teacher_rates["mu"].reshape(-1, n_steps).mean(axis=1),
            "truth_mu_path": test["params"]["mu"], "expanded": {name: expand_to_original(field) for name, field in fields.items()}}


def _law_block(cfg: dict[str, Any], teacher: RateMLP, fields: dict[str, PolynomialField],
               dt: float, radius: float, slots: int) -> dict[str, Any]:
    """Fresh paths, four coupled forecasts. The floor uses an independent truth draw."""
    n = int(cfg["data"]["n_law"])
    n_steps = int(cfg["data"]["future_steps"])
    seeds = cfg["seeds"]
    windows = generate_windows(cfg, np.random.default_rng(int(seeds["law_truth"])), n)
    present = windows["present"]
    returns = windows["returns"]
    params = windows["params"]
    # ``law_noise`` and ``law_floor`` are distinct. Adding one to ``law_truth``
    # collides with ``law_floor`` and makes the Monte Carlo floor identically zero.
    noise_a = make_noise(n, n_steps, slots, np.random.default_rng(int(seeds["law_noise"])))
    noise_b = make_noise(n, n_steps, slots, np.random.default_rng(int(seeds["law_floor"])))
    truth = simulate_forecast(constant_rates(params), present, noise_a, dt, radius)
    floor = simulate_forecast(constant_rates(params), present, noise_b, dt, radius)
    teacher_sim = simulate_forecast(network_rates(teacher, returns, n_steps), present, noise_a, dt, radius)
    symbolic_sim = simulate_forecast(scheduled_rates(fields, returns, n_steps), present, noise_a, dt, radius)
    failed_floor = truth["failed"] | floor["failed"]
    failed_teacher = truth["failed"] | teacher_sim["failed"]
    failed_symbolic = truth["failed"] | symbolic_sim["failed"]
    return {
        "law_fail_truth": truth["failure_rate"],
        "law_fail_teacher": teacher_sim["failure_rate"],
        "law_fail_symbolic": symbolic_sim["failure_rate"],
        "floor_mean": mean_path_rmse(truth["log_s"], floor["log_s"], failed_floor),
        "floor_var": terminal_var_relative(truth["log_s"], floor["log_s"], failed_floor),
        "law_mean_teacher": mean_path_rmse(teacher_sim["log_s"], truth["log_s"], failed_teacher),
        "law_mean_symbolic": mean_path_rmse(symbolic_sim["log_s"], truth["log_s"], failed_symbolic),
        "law_var_teacher": terminal_var_relative(teacher_sim["log_s"], truth["log_s"], failed_teacher),
        "law_var_symbolic": terminal_var_relative(symbolic_sim["log_s"], truth["log_s"], failed_symbolic),
        "law_discarded": windows["discarded"],
    }


def make_figure(outcome: dict[str, Any], path: Path) -> None:
    """Slice of the drift rate and the pathwise scatter. Not used for selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = outcome["slice_x"]
    r_values = np.unique(np.round(grid[:, 0], decimals=8))
    times = np.unique(np.round(grid[:, -1], decimals=8))
    teacher = outcome["slice_teacher_mu"].reshape(len(times), len(r_values))
    symbolic = outcome["slice_symbolic_mu"].reshape(len(times), len(r_values))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    extent = (r_values.min(), r_values.max(), times.min(), times.max())
    image = axes[0].imshow(teacher, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("teacher mu")
    axes[0].set_xlabel("r_1")
    axes[0].set_ylabel("u")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    diff = axes[1].imshow(symbolic - teacher, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    axes[1].set_title("symbolic - teacher")
    axes[1].set_xlabel("r_1")
    fig.colorbar(diff, ax=axes[1], fraction=0.046)
    axes[2].scatter(outcome["truth_mu_path"], outcome["teacher_mu_path"], s=16, alpha=0.8)
    axes[2].set_xlabel("latent mu")
    axes[2].set_ylabel("teacher mu, mean over u")
    axes[2].set_title("test paths")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def probe_scales(seed: int = 19099) -> dict[str, Any]:
    """Truth-only scales on the probe seed. Does not train and does not use seed 19."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    n = 128
    windows = generate_windows(cfg, rng, n)
    dt = float(cfg["data"]["dt"])
    n_steps = int(cfg["data"]["future_steps"])
    features = design_matrix(windows["returns"], n_steps)
    mu = windows["params"]["mu"]
    sigma = windows["params"]["sigma"]
    noise = make_noise(n, n_steps, int(cfg["data"]["jump_slots"]), np.random.default_rng(seed + 3))
    truth = simulate_forecast(constant_rates(windows["params"]), windows["present"], noise, dt, float(cfg["data"]["log_radius"]))
    constant_params = {key: np.full(n, float(windows["params"][key].mean())) for key in NAMES}
    constant_params["lam"] = windows["params"]["lam"]
    constant_params["nu"] = windows["params"]["nu"]
    constant_params["gamma"] = windows["params"]["gamma"]
    constant = simulate_forecast(constant_rates(constant_params), windows["present"], noise, dt,
                                 float(cfg["data"]["log_radius"]))
    failed = truth["failed"] | constant["failed"]
    noise_b = make_noise(n, n_steps, int(cfg["data"]["jump_slots"]), np.random.default_rng(seed + 4))
    floor = simulate_forecast(constant_rates(windows["params"]), windows["present"], noise_b, dt,
                              float(cfg["data"]["log_radius"]))
    scales = {
        "probe_seed": seed,
        "discarded": windows["discarded"],
        "constant_floor_formula": uniform_constant_floor(0.1, 0.5),
        "constant_floor_empirical_mu": nrmse(repeat_targets(mu, n_steps), np.full(n * n_steps, float(mu.mean()))),
        "step_ratio": step_ratio(mu, sigma, dt),
        "vif_degree3": vif_max(features, 3),
        "paired_constant_mu_sigma_rmse": path_rmse(constant["log_s"], truth["log_s"], failed),
        "floor_mean": mean_path_rmse(truth["log_s"], floor["log_s"], truth["failed"] | floor["failed"]),
        "floor_var": terminal_var_relative(truth["log_s"], floor["log_s"], truth["failed"] | floor["failed"]),
        "return_std": float(windows["returns"].std()),
        "return_max_abs": float(np.max(np.abs(windows["returns"]))),
    }
    print(json.dumps(scales, indent=2))
    return scales


def main() -> None:
    """Train the two teachers, distill the jump model, and write the one-shot test."""
    started = time.perf_counter()
    cfg = load_config()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    data_cfg = cfg["data"]
    total = int(data_cfg["n_train"]) + int(data_cfg["n_val"]) + int(data_cfg["n_test"])
    windows = generate_windows(cfg, np.random.default_rng(int(cfg["seeds"]["data"])), total)
    frames = split_windows(windows, cfg)
    print(f"windows discarded={windows['discarded']} return_std={windows['returns'].std():.4g}", flush=True)
    trained = train_model(frames["train"], cfg, True, int(cfg["seeds"]["network"]), "neural-mjd")
    baseline = train_model(frames["train"], cfg, False, int(cfg["seeds"]["network_bs"]), "neural-bs")
    n_steps = int(data_cfg["future_steps"])
    x_train = design_matrix(frames["train"]["returns"], n_steps)
    x_val = design_matrix(frames["val"]["returns"], n_steps)
    teacher_train = query_rates(trained["model"], x_train)
    teacher_val = query_rates(trained["model"], x_val)
    fields: dict[str, PolynomialField] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name in NAMES:
        fields[name], meta[name] = select_field(x_train, teacher_train[name], x_val, teacher_val[name], cfg, name)
        print(f"selected {name} degree={meta[name]['degree']} threshold={meta[name]['threshold']} "
              f"val={meta[name]['score']:.4g} active={meta[name]['active']}", flush=True)
    outcome = evaluate(cfg, frames, trained["model"], baseline["model"], fields, meta)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_weights(trained["model"], OUT_DIR / "teacher_mjd_weights.json")
    save_weights(baseline["model"], OUT_DIR / "teacher_bs_weights.json")
    var_names = tuple(f"r_{{{i}}}" for i in range(1, 10)) + ("u",)
    lhs = {
        "mu": r"\mu(r,u)",
        "sigma": r"\sigma(r,u)",
        "lam": r"\lambda(r,u)",
        "nu": r"\nu(r,u)",
        "gamma": r"\gamma(r,u)",
    }
    tex_parts = ["% Day 19 symbolic MJD rates. These approximate the neural teacher, not the latent rates."]
    expanded_dump: dict[str, list[dict[str, float]]] = {}
    for name, field in fields.items():
        field.save(OUT_DIR / f"symbolic_{name}.json")
        expanded = outcome["expanded"][name]
        tex_parts.append(tex_equations(expanded, (lhs[name],), var_names))
        expanded_dump[name] = [{str(key): value for key, value in poly.items()} for poly in expanded]
    tex = "\n".join(tex_parts)
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (REPORT_DIR / "symbolic_sde.tex").write_text(tex)
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps(expanded_dump, indent=2))
    make_figure(outcome, REPORT_DIR / "drift_grid.png")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "git_head": head,
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "discarded": windows["discarded"],
        "return_std": float(windows["returns"].std()),
        "return_max_abs": float(np.max(np.abs(windows["returns"]))),
        "param_means": {key: float(value.mean()) for key, value in windows["params"].items()},
        "mjd_loss": trained["loss"],
        "bs_loss": baseline["loss"],
        "selection_candidates": {name: row["candidates"] for name, row in meta.items()},
        "metrics": jsonify(outcome["metrics"]),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    print(json.dumps(jsonify(outcome["metrics"]["verdicts"]), indent=2))
    print(f"wrote {OUT_DIR / 'results.json'} in {payload['seconds']:.1f}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe_scales()
    else:
        main()
