"""Day 13: Bartosh, Vetrov & Naesseth (ICML 2025) SDE Matching -> polynomial SINDy.

Pipeline (settings frozen in ``configs/daily_neural_sde_day13.json``):

1. Data: stochastic Lotka-Volterra in observable coordinates (x, y).
   The closed form is the data law. It is not a gene-regulatory equation.
2. Teacher: neural drift and diagonal neural diffusion trained by the
   simulation-free variational bound (posterior marginal F_phi, shared g).
   A time-only diffusion with the same budget is the H6 baseline.
3. Distillation: direct teacher queries, trajectory-disjoint splits,
   power library in (x, y, t), validation-only selection, test once.
4. Simulation keeps failed-trajectory mass and classifies non-positive
   populations separately from explosions.

Run from the repository root:
``python -m src.daily_neural_sde.day13_bartosh_sde_matching``

Outputs land in ``outputs/daily_neural_sde/day13/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day13.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day13"

FieldFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen day-13 config."""
    return json.loads(path.read_text())


def n_observation_steps(cfg: dict[str, Any]) -> int:
    """Number of Euler intervals on the stored grid. ``T`` must be a multiple of ``dt``."""
    t_horizon = float(cfg["data"]["T"])
    dt = float(cfg["data"]["dt"])
    steps = int(round(t_horizon / dt))
    if abs(steps * dt - t_horizon) > 1e-9:
        raise ValueError("T is not a multiple of dt")
    return steps


def observation_times(cfg: dict[str, Any]) -> np.ndarray:
    """Stored grid, shape (n_steps + 1,), from 0 through T inclusive."""
    steps = n_observation_steps(cfg)
    return np.linspace(0.0, float(cfg["data"]["T"]), steps + 1)


# ---------------------------------------------------------------- truth


def true_drift(x: np.ndarray, alpha: float, beta: float, delta: float,
               gamma: float) -> np.ndarray:
    """Lotka-Volterra drift. ``x`` has shape (n, 2) and the result matches it.

    dx = (alpha x - beta x y) dt, dy = (delta x y - gamma y) dt.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    prey, predator = x[:, 0], x[:, 1]
    return np.stack([
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator,
    ], axis=1)


def true_diffusion(x: np.ndarray, sigma: float) -> np.ndarray:
    """Diagonal multiplicative diffusion. ``x`` has shape (n, 2)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return sigma * x


def sample_x0(rng: np.random.Generator, n: int, cfg: dict[str, Any]) -> np.ndarray:
    """Lognormal initial conditions clipped to the frozen box. Shape (n, 2)."""
    data = cfg["data"]
    log_mean = np.asarray(data["x0_log_mean"], dtype=np.float64)
    draw = rng.normal(log_mean, float(data["x0_log_std"]), size=(n, 2))
    x0 = np.exp(draw)
    return np.clip(x0, np.asarray(data["x0_min"], dtype=np.float64),
                   np.asarray(data["x0_max"], dtype=np.float64))


def make_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Trajectory-disjoint truth paths. One x0 and one Wiener path per row.

    Stored states have shape (n, n_obs, 2). Generation uses several Euler
    steps per stored interval. A non-positive or non-finite state aborts:
    that event is a generator failure, not a silent drop.
    """
    data = cfg["data"]
    alpha = float(data["alpha"])
    beta = float(data["beta"])
    delta = float(data["delta"])
    gamma = float(data["gamma"])
    sigma = float(data["sigma"])
    dt = float(data["dt"])
    sub = int(data["generation_substeps"])
    dt_gen = dt / sub
    steps = n_observation_steps(cfg)
    times = observation_times(cfg)
    split = cfg["split"]
    rng = np.random.default_rng(int(split["seed"]))
    out: dict[str, Any] = {"times": times, "dt": dt, "n_steps": steps,
                           "generation_dt": dt_gen}
    sqrt_dt = math.sqrt(dt_gen)
    for name in ("train", "validation", "test"):
        n = int(split[name])
        x0 = sample_x0(rng, n, cfg)
        x = x0.copy()
        stored = [x.copy()]
        for _ in range(steps):
            for _sub in range(sub):
                increment = rng.standard_normal((n, 2))
                drift = true_drift(x, alpha, beta, delta, gamma)
                diffusion = true_diffusion(x, sigma)
                x = x + drift * dt_gen + diffusion * sqrt_dt * increment
                if not np.isfinite(x).all() or np.any(x <= 0.0):
                    raise RuntimeError(f"truth path left the positive orthant during {name}")
            stored.append(x.copy())
        out[name] = {"paths": np.stack(stored, axis=1), "x0": x0}
    return out


def normalization_from_train(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate mean and std of training states. Shape (2,)."""
    flat = paths.reshape(-1, paths.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


# ---------------------------------------------------------------- posterior algebra


def softplus_inverse(value: float) -> float:
    """Inverse of softplus for a positive ``value``."""
    if value <= 0.0:
        raise ValueError("softplus inverse is defined for positive values")
    return float(math.log(math.expm1(value)))


def gaussian_kl_diag(mu_q: torch.Tensor, sigma_q: torch.Tensor,
                     mu_p: torch.Tensor, sigma_p: torch.Tensor) -> torch.Tensor:
    """KL of diagonal Gaussians, summed over the state. Shape (batch,).

    ``mu_*`` and ``sigma_*`` have shape (batch, d). ``sigma`` is the std.
    """
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    terms = (torch.log(var_p) - torch.log(var_q) - 1.0
             + var_q / var_p + (mu_q - mu_p) ** 2 / var_p)
    return 0.5 * terms.sum(dim=-1)


def flow_velocity(mu: torch.Tensor, sigma: torch.Tensor, eps: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
    """Time derivative of mu + sigma * eps at fixed noise. Shape (batch, d).

    ``t`` has shape (batch, 1) and must be connected to ``mu`` and ``sigma``.
    """
    columns = []
    for index in range(mu.shape[1]):
        dmu = torch.autograd.grad(mu[:, index].sum(), t, create_graph=True,
                                  retain_graph=True, allow_unused=True)[0]
        dsigma = torch.autograd.grad(sigma[:, index].sum(), t, create_graph=True,
                                     retain_graph=True, allow_unused=True)[0]
        if dmu is None:
            dmu = torch.zeros_like(t)
        if dsigma is None:
            dsigma = torch.zeros_like(t)
        columns.append(dmu + dsigma * eps[:, index:index + 1])
    return torch.cat(columns, dim=1)


def diagonal_gg_divergence(g2: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Divergence of diag(g^2). Component i is d(g_i^2) / d(z_i). Shape (batch, d).

    ``g2`` must depend on ``z`` if the diffusion is state-dependent. Unused
    components contribute 0, which is the state-independent case.
    """
    if not g2.requires_grad:
        return torch.zeros_like(g2)
    columns = []
    for index in range(g2.shape[1]):
        grad = torch.autograd.grad(g2[:, index].sum(), z, create_graph=True,
                                   retain_graph=True, allow_unused=True)[0]
        if grad is None:
            columns.append(torch.zeros(g2.shape[0], 1, dtype=g2.dtype))
        else:
            columns.append(grad[:, index:index + 1])
    return torch.cat(columns, dim=1)


def assemble_posterior_drift(f_bar: torch.Tensor, g: torch.Tensor,
                             score: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    """Posterior drift from the probability-flow identity. Shape (batch, d).

    f = f_bar + (1/2) g^2 score + (1/2) div(g g^T), with diagonal g.
    This is equation (20) of Bartosh et al. for a diagonal diffusion.
    """
    return f_bar + 0.5 * (g ** 2) * score + 0.5 * div


# ---------------------------------------------------------------- model


def _mlp(in_dim: int, width: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, out_dim),
    )


def _init_mlp(module: nn.Module, last_bias: float | None) -> None:
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for layer in linears:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    if last_bias is not None and linears:
        nn.init.constant_(linears[-1].bias, last_bias)


class MatchingModel(nn.Module):
    """Prior Neural SDE plus a simulation-free Gaussian posterior.

    Training consumes normalized states. ``mean`` and ``std`` are the map
    back to population coordinates and are not used inside the loss.
    """

    def __init__(self, cfg: dict[str, Any], state_dependent: bool) -> None:
        super().__init__()
        teacher = cfg["teacher"]
        width = int(teacher["width"])
        hidden = int(teacher["gru_hidden"])
        self.state_dependent = bool(state_dependent)
        self.T = float(cfg["data"]["T"])
        self.g_floor = float(teacher["g_floor"])
        self.sigma_floor = float(teacher["sigma_floor"])
        self.rho_floor = float(teacher["rho_floor"])
        self.prior_std_floor = float(teacher["prior_std_floor"])
        self.drift_net = _mlp(3, width, 2)
        diffusion_in = 3 if self.state_dependent else 1
        self.diffusion_net = _mlp(diffusion_in, width, 2)
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, num_layers=1,
                          batch_first=True)
        self.mu_net = _mlp(hidden + 1, width, 2)
        self.sigma_net = _mlp(hidden + 1, width, 2)
        self.prior_mean = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self._prior_std_raw = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self._rho_raw = nn.Parameter(torch.zeros((), dtype=torch.float64))
        times = torch.tensor(observation_times(cfg), dtype=torch.float64)
        self.register_buffer("times", times)
        self.register_buffer("coord_mean", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("coord_std", torch.ones(2, dtype=torch.float64))
        self.double()
        _init_mlp(self.drift_net, None)
        _init_mlp(self.diffusion_net, softplus_inverse(float(teacher["g_init"]) - self.g_floor))
        _init_mlp(self.mu_net, None)
        _init_mlp(self.sigma_net, softplus_inverse(float(teacher["sigma_init"]) - self.sigma_floor))
        nn.init.zeros_(self.prior_mean)
        nn.init.constant_(self._prior_std_raw,
                          softplus_inverse(float(teacher["prior_std_init"]) - self.prior_std_floor))
        nn.init.constant_(self._rho_raw,
                          softplus_inverse(float(teacher["rho_init"]) - self.rho_floor))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store the train-only map. Shape (2,)."""
        with torch.no_grad():
            self.coord_mean.copy_(torch.tensor(mean, dtype=torch.float64))
            self.coord_std.copy_(torch.tensor(std, dtype=torch.float64))

    def rho(self) -> torch.Tensor:
        """Observation std in normalized coordinates. Scalar tensor."""
        return self.rho_floor + torch.nn.functional.softplus(self._rho_raw)

    def prior_std(self) -> torch.Tensor:
        """Diagonal prior std at t = 0. Shape (2,)."""
        return self.prior_std_floor + torch.nn.functional.softplus(self._prior_std_raw)

    def encode(self, paths: torch.Tensor) -> torch.Tensor:
        """Reverse-time GRU context. ``paths`` is (batch, n_obs, 2) -> (batch, hidden)."""
        batch, n_obs, _ = paths.shape
        u = (self.times / self.T).view(1, n_obs, 1).expand(batch, n_obs, 1)
        rev = torch.flip(torch.cat([paths, u], dim=-1), dims=[1])
        _output, hidden = self.gru(rev)
        return hidden[-1]

    def factors(self, context: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and std at time ``t``. Both have shape (batch, 2)."""
        features = torch.cat([context, t / self.T], dim=-1)
        mu = self.mu_net(features)
        sigma = self.sigma_floor + torch.nn.functional.softplus(self.sigma_net(features))
        return mu, sigma

    def prior_drift(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift in normalized coordinates. ``z`` is (batch, 2), ``t`` is (batch, 1)."""
        return self.drift_net(torch.cat([z, t / self.T], dim=-1))

    def prior_diffusion(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Positive diagonal diffusion in normalized coordinates. Shape (batch, 2)."""
        if self.state_dependent:
            features = torch.cat([z, t / self.T], dim=-1)
        else:
            features = t / self.T
        raw = self.diffusion_net(features)
        return self.g_floor + torch.nn.functional.softplus(raw)

    def elbo_terms(self, paths: torch.Tensor) -> dict[str, torch.Tensor]:
        """One-sample Monte Carlo estimate of the variational bound.

        ``paths`` has shape (batch, n_obs, 2) in normalized coordinates.
        Returns the batch-mean prior KL, diffusion integral, and reconstruction
        term. The diffusion term includes the factor T. Reconstruction includes
        the factor n_obs, matching Algorithm 1.
        """
        batch, n_obs, dimension = paths.shape
        context = self.encode(paths)
        t0 = torch.zeros(batch, 1, dtype=paths.dtype)
        mu0, sigma0 = self.factors(context, t0)
        prior_mean = self.prior_mean.view(1, 2).expand(batch, 2)
        prior_std = self.prior_std().view(1, 2).expand(batch, 2)
        kl = gaussian_kl_diag(mu0, sigma0, prior_mean, prior_std).mean()

        t = torch.rand(batch, 1, dtype=paths.dtype) * self.T
        t = t.detach().requires_grad_(True)
        eps = torch.randn(batch, dimension, dtype=paths.dtype)
        mu, sigma = self.factors(context, t)
        z = mu + sigma * eps
        f_bar = flow_velocity(mu, sigma, eps, t)
        g = self.prior_diffusion(z, t)
        # z is an input of g. Keep it in the graph for the divergence.
        div = diagonal_gg_divergence(g ** 2, z)
        score = -eps / sigma
        posterior = assemble_posterior_drift(f_bar, g, score, div)
        prior = self.prior_drift(z, t)
        residual = (prior - posterior) / g
        diffusion = self.T * 0.5 * (residual ** 2).sum(dim=-1).mean()

        index = torch.randint(0, n_obs, (batch,))
        t_obs = self.times[index].view(batch, 1)
        eps_obs = torch.randn(batch, dimension, dtype=paths.dtype)
        mu_obs, sigma_obs = self.factors(context, t_obs)
        z_obs = mu_obs + sigma_obs * eps_obs
        observed = paths[torch.arange(batch), index]
        rho = self.rho()
        quad = ((observed - z_obs) ** 2).sum(dim=-1) / (rho ** 2)
        nll = 0.5 * quad + 0.5 * dimension * torch.log(2.0 * math.pi * rho ** 2)
        reconstruction = float(n_obs) * nll.mean()
        return {"prior": kl, "diffusion": diffusion, "reconstruction": reconstruction,
                "total": kl + diffusion + reconstruction, "rho": rho.detach()}


def train_model(state_dependent: bool, paths_z: np.ndarray, cfg: dict[str, Any]
                ) -> tuple[MatchingModel, dict[str, Any]]:
    """Adam on the variational bound. Fixed step count. No validation peek."""
    teacher = cfg["teacher"]
    torch.manual_seed(int(cfg["seed"]))
    model = MatchingModel(cfg, state_dependent=state_dependent)
    torch.manual_seed(int(cfg["seed"]) + 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(teacher["lr"]),
                                 weight_decay=float(teacher["weight_decay"]))
    tensor = torch.tensor(paths_z, dtype=torch.float64)
    n = len(tensor)
    batch_size = int(teacher["batch_size"])
    steps = int(teacher["n_opt_steps"])
    clip = float(teacher["grad_clip"])
    rng = np.random.default_rng(int(cfg["seed"]) + 17)
    history: list[dict[str, float]] = []
    skipped = 0
    model.train()
    for step in range(steps):
        pick = rng.integers(0, n, size=batch_size)
        batch = tensor[pick]
        optimizer.zero_grad(set_to_none=True)
        terms = model.elbo_terms(batch)
        total = terms["total"]
        if not torch.isfinite(total):
            skipped += 1
            continue
        total.backward()
        grads_ok = True
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                grads_ok = False
                break
        if not grads_ok:
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({
                "step": step,
                "total": float(total.detach()),
                "prior": float(terms["prior"].detach()),
                "diffusion": float(terms["diffusion"].detach()),
                "reconstruction": float(terms["reconstruction"].detach()),
                "rho": float(terms["rho"]),
            })
            print(f"  {'state' if state_dependent else 'time'}-diffusion "
                  f"step {step}: total {history[-1]['total']:.4f} "
                  f"diff {history[-1]['diffusion']:.4f} "
                  f"rec {history[-1]['reconstruction']:.4f}", flush=True)
    model.eval()
    return model, {"skipped_nonfinite_steps": skipped, "history": history,
                   "state_dependent": state_dependent, "n_opt_steps": steps}


# ---------------------------------------------------------------- numpy replay


def stable_softplus(value: np.ndarray) -> np.ndarray:
    """softplus with a clip that matches torch to many digits."""
    return np.where(value > 20.0, value, np.log1p(np.exp(np.clip(value, -40.0, 20.0))))


def mlp_numpy(features: np.ndarray, layers: list[dict[str, Any]]) -> np.ndarray:
    """Tanh MLP. ``features`` is (n, in). Hidden layers use tanh; the last is linear."""
    hidden = np.asarray(features, dtype=np.float64)
    for index, layer in enumerate(layers):
        weight = np.asarray(layer["weight"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        hidden = hidden @ weight.T + bias
        if index < len(layers) - 1:
            hidden = np.tanh(hidden)
    return hidden


def export_layers(module: nn.Module) -> list[dict[str, list[float]]]:
    """Linear layers of an MLP, in order, as JSON-ready lists."""
    layers = []
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            layers.append({"weight": layer.weight.detach().cpu().tolist(),
                           "bias": layer.bias.detach().cpu().tolist()})
    return layers


def pack_weights(model: MatchingModel) -> dict[str, Any]:
    """Prior field only. The posterior is not part of the distilled SDE."""
    return {
        "drift": export_layers(model.drift_net),
        "diffusion": export_layers(model.diffusion_net),
        "mean": model.coord_mean.detach().cpu().tolist(),
        "std": model.coord_std.detach().cpu().tolist(),
        "T": model.T,
        "g_floor": model.g_floor,
        "state_dependent": model.state_dependent,
    }


def teacher_fields(weights: dict[str, Any], x: np.ndarray, t: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Physical drift and diffusion. ``x`` is (n, 2), ``t`` is (n,).

    The network runs in the normalized coordinate and the outputs are
    multiplied by the training std. Time enters as t / T.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if len(t) == 1 and len(x) > 1:
        t = np.full(len(x), float(t[0]))
    mean = np.asarray(weights["mean"], dtype=np.float64)
    std = np.asarray(weights["std"], dtype=np.float64)
    z = (x - mean) / std
    u = (t / float(weights["T"])).reshape(-1, 1)
    drift_z = mlp_numpy(np.concatenate([z, u], axis=1), weights["drift"])
    if weights["state_dependent"]:
        features = np.concatenate([z, u], axis=1)
    else:
        features = u
    diffusion_z = float(weights["g_floor"]) + stable_softplus(
        mlp_numpy(features, weights["diffusion"]))
    return drift_z * std, diffusion_z * std


def parity_error(model: MatchingModel, weights: dict[str, Any], seed: int) -> float:
    """Max absolute gap between the torch prior and the NumPy replay."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 2))
    t = rng.uniform(0.0, model.T, size=8)
    mean = model.coord_mean.detach().cpu().numpy()
    std = model.coord_std.detach().cpu().numpy()
    z = torch.tensor((x - mean) / std, dtype=torch.float64)
    tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float64)
    with torch.no_grad():
        drift_t = model.prior_drift(z, tt).numpy() * std
        diffusion_t = model.prior_diffusion(z, tt).numpy() * std
    drift_n, diffusion_n = teacher_fields(weights, x, t)
    return float(max(np.max(np.abs(drift_t - drift_n)),
                     np.max(np.abs(diffusion_t - diffusion_n))))


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
    """Stack (x, y, t) and direct field queries. A row is kept only if all are finite."""
    n_traj = paths.shape[0]
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


# ---------------------------------------------------------------- simulation


def _latch(x: np.ndarray, failed: np.ndarray, failure_class: list[str | None],
           excursion: np.ndarray, failure_radius: float,
           excursion_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mark new failures. Failed rows of ``x`` become NaN.

    Classes, in order: ``non_finite``, ``negative_state``, ``outside_radius``.
    An excursion is a finite positive state whose norm exceeds
    ``excursion_radius`` before failure.
    """
    finite = np.isfinite(x).all(axis=1)
    negative = finite & np.any(x <= 0.0, axis=1)
    safe = np.where(finite[:, None], x, 0.0)
    norm = np.linalg.norm(safe, axis=1)
    outside = finite & ~negative & (norm > failure_radius)
    excursion = excursion | (~failed & finite & ~negative & (norm > excursion_radius))
    newly = (~failed) & (~finite | negative | outside)
    for idx in np.flatnonzero(newly):
        if not finite[idx]:
            failure_class[idx] = "non_finite"
        elif negative[idx]:
            failure_class[idx] = "negative_state"
        else:
            failure_class[idx] = "outside_radius"
    failed = failed | newly
    x = np.array(x, dtype=np.float64, copy=True)
    x[failed] = np.nan
    return x, failed, excursion


def simulate_population(drift_fn: Callable[[np.ndarray, float], np.ndarray],
                        diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
                        x0: np.ndarray, dt: float, increments: np.ndarray,
                        failure_radius: float, excursion_radius: float) -> dict[str, Any]:
    """Euler-Maruyama for a time-dependent diagonal population SDE.

    ``drift_fn(x, t)`` and ``diffusion_fn(x, t)`` map active states (n_active, 2)
    and a scalar time to (n_active, 2). ``increments`` has shape (n, n_steps, 2)
    of standard normals. The update adds ``diffusion * sqrt(dt) * increment``.
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
    for step in range(increments.shape[1]):
        active = ~failed
        if active.any():
            drift = np.asarray(drift_fn(x[active], step * dt), dtype=np.float64)
            diffusion = np.asarray(diffusion_fn(x[active], step * dt), dtype=np.float64)
            if drift.shape != (int(active.sum()), 2) or diffusion.shape != (int(active.sum()), 2):
                raise ValueError("field functions must return shape (n_active, 2)")
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
            "n_negative_state": int(sum(c == "negative_state" for c in classes)),
            "n_outside_radius": int(sum(c == "outside_radius" for c in classes))}


def coarse_increments(fine: np.ndarray, block: int) -> np.ndarray:
    """N(0, 1) draws for a block sum of Wiener increments. Shape (n, n_coarse, d)."""
    if fine.shape[1] % block != 0:
        raise ValueError("fine step count is not divisible by block")
    n, n_fine, d = fine.shape
    grouped = fine.reshape(n, n_fine // block, block, d)
    return grouped.sum(axis=2) / np.sqrt(block)


def mean_path(states: np.ndarray) -> np.ndarray:
    """Cross-trajectory mean. Shape (time, 2). All-NaN times stay NaN."""
    with np.errstate(all="ignore"):
        return np.nanmean(states, axis=0)


def mean_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square of the Euclidean gap between two mean paths."""
    gap = a - b
    if not np.isfinite(gap).all():
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(gap ** 2, axis=1))))


def terminal_cov(states: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Covariance of finite terminal states, and how many were retained."""
    final = states[:, -1, :]
    ok = np.isfinite(final).all(axis=1)
    if int(ok.sum()) < 3:
        return None, int(ok.sum())
    return np.cov(final[ok], rowvar=False), int(ok.sum())


def rel_frob(left: np.ndarray, right: np.ndarray) -> float:
    """||left - right||_F / ||right||_F."""
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-8))


def polynomial_call(field: PolynomialField) -> Callable[[np.ndarray, float], np.ndarray]:
    """Wrap a polynomial in (x, y, t) as a simulator callback."""

    def _call(xy: np.ndarray, t: float) -> np.ndarray:
        features = np.concatenate([xy, np.full((len(xy), 1), t)], axis=1)
        return field.predict(features)

    return _call


def fresh_paths(drift_fn: Callable[[np.ndarray, float], np.ndarray],
                diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
                n: int, dt: float, n_steps: int, seed: int, cfg: dict[str, Any],
                failure_radius: float, excursion_radius: float) -> dict[str, Any]:
    """Paths from a fresh draw of the frozen initial law."""
    rng = np.random.default_rng(seed)
    x0 = sample_x0(rng, n, cfg)
    increments = rng.standard_normal((n, n_steps, 2))
    sim = simulate_population(drift_fn, diffusion_fn, x0, dt, increments,
                              failure_radius, excursion_radius)
    sim["x0"] = x0
    return sim


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
            "n": int(len(xy)),
            "fraction_nonpositive_state": float(np.mean(np.any(xy <= 0.0, axis=1))),
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


def monomial_coefficient(poly: dict[tuple[int, ...], float], exp: tuple[int, ...]) -> float:
    """Coefficient of one exponent, or 0 when the term was dropped."""
    return float(poly.get(exp, 0.0))


# ---------------------------------------------------------------- verdicts


def _pass(flag: bool) -> str:
    return "pass" if flag else "fail"


def decide(cfg: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates. ``metrics`` holds numbers already computed on test.

    Distribution clauses become ``indeterminate`` when their pre-registered
    failure or Monte Carlo-floor gate shuts. Thresholds are read from ``cfg``
    and are not modified here.
    """
    hyp = cfg["hypotheses"]
    h1, h2 = hyp["H1_drift"], hyp["H2_diffusion"]
    h4, h5 = hyp["H4_simulation"], hyp["H5_law"]
    h6, h7 = hyp["H6_state_independent_baseline"], hyp["H7_discretization_gate"]
    verdicts = {
        "H1": _pass(metrics["drift_sym_teacher"] <= h1["symbolic_vs_teacher_test_nrmse_max"]
                    and metrics["drift_teacher_truth"] <= h1["teacher_vs_truth_test_nrmse_max"]
                    and metrics["drift_sym_truth"] <= h1["symbolic_vs_truth_test_nrmse_max"]),
        "H2": _pass(metrics["g_sym_teacher"] <= h2["symbolic_vs_teacher_test_nrmse_max"]
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
    exc_ok = abs(metrics["exc_symbolic"] - metrics["exc_teacher"]) <= h4["excursion_rate_gap_max"]
    h7_ok = (math.isfinite(metrics["disc_rmse"])
             and metrics["disc_rmse"] <= h7["teacher_coarse_vs_fine_terminal_rmse_max"]
             and metrics["disc_retained"] >= h7["min_retained_fraction"])
    cov = metrics["cov_sym_teacher"]
    dist_ok = (math.isfinite(metrics["mean_sym_teacher"])
               and math.isfinite(metrics["mean_teacher_truth"])
               and metrics["mean_sym_teacher"] <= h4["paired_mean_rmse_symbolic_vs_teacher_max"]
               and metrics["mean_teacher_truth"] <= h4["paired_mean_rmse_teacher_vs_truth_max"]
               and cov is not None
               and cov <= h4["terminal_cov_rel_frob_symbolic_vs_teacher_max"])
    gate = (max(metrics["fail_truth"], metrics["fail_teacher"], metrics["fail_symbolic"])
            <= h4["failure_gate"] and h7_ok)
    verdicts["H4_failure"] = _pass(fail_ok)
    verdicts["H4_excursion_gap"] = _pass(exc_ok)
    verdicts["H4"] = (_pass(fail_ok and exc_ok and dist_ok) if gate else "indeterminate")
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
    improvement = metrics["g_time_truth"] - metrics["g_state_truth"]
    shared = (metrics["g_state_truth"] >= h6["shared_floor_nrmse"]
              and metrics["g_time_truth"] >= h6["shared_floor_nrmse"])
    verdicts["H6"] = _pass(improvement >= h6["diffusion_nrmse_improvement_min"]
                           and metrics["g_state_truth"] <= h6["state_dependent_nrmse_max"])
    verdicts["H6_shared_floor"] = bool(shared)
    return verdicts


# ---------------------------------------------------------------- reporting helpers


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


def posterior_scale_probe(model: MatchingModel, paths_z: np.ndarray, n: int = 64
                          ) -> dict[str, float]:
    """Mean posterior std on training series. Compared later with g sqrt(dt)."""
    model.eval()
    tensor = torch.tensor(paths_z[:n], dtype=torch.float64)
    with torch.no_grad():
        context = model.encode(tensor)
        t = torch.rand(len(tensor), 1, dtype=torch.float64) * model.T
        _mu, sigma = model.factors(context, t)
    sigma_np = sigma.numpy()
    return {"mean_sigma": float(sigma_np.mean()),
            "min_sigma": float(sigma_np.min()),
            "max_sigma": float(sigma_np.max()),
            "rho": float(model.rho().detach())}


def main() -> dict[str, Any]:
    """Train, distill, validate once, and write the day-13 outputs."""
    started = time.time()
    cfg = load_config()
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    data_cfg = cfg["data"]
    alpha = float(data_cfg["alpha"])
    beta = float(data_cfg["beta"])
    delta = float(data_cfg["delta"])
    gamma = float(data_cfg["gamma"])
    sigma = float(data_cfg["sigma"])
    print("generating Lotka-Volterra paths", flush=True)
    dataset = make_dataset(cfg)
    mean, std = normalization_from_train(dataset["train"]["paths"])
    train_z = (dataset["train"]["paths"] - mean) / std

    print("training state-dependent teacher", flush=True)
    state_model, state_log = train_model(True, train_z, cfg)
    state_model.set_normalization(mean, std)
    print("training time-only diffusion baseline", flush=True)
    time_model, time_log = train_model(False, train_z, cfg)
    time_model.set_normalization(mean, std)

    state_weights = pack_weights(state_model)
    time_weights = pack_weights(time_model)
    parity = parity_error(state_model, state_weights, int(cfg["seed"]) + 7)
    if parity > 1e-6:
        raise RuntimeError(f"numpy/torch field mismatch {parity}")
    probe_z = torch.zeros(8, 2, dtype=torch.float64)
    probe_t = torch.linspace(0.0, float(data_cfg["T"]), 8, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        probe_f = state_model.prior_drift(probe_z, probe_t)
        probe_g = state_model.prior_diffusion(probe_z, probe_t)
    probe = {"z": list(probe_z.shape), "t": list(probe_t.shape),
             "drift": list(probe_f.shape), "diffusion": list(probe_g.shape),
             "numpy_torch_max_abs": parity}
    sigma_probe = posterior_scale_probe(state_model, train_z)

    def state_f(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_fields(state_weights, xy, t)[0]

    def state_g(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_fields(state_weights, xy, t)[1]

    def time_f(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_fields(time_weights, xy, t)[0]

    def time_g(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return teacher_fields(time_weights, xy, t)[1]

    def truth_f(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return true_drift(xy, alpha, beta, delta, gamma)

    def truth_g(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        return true_diffusion(xy, sigma)

    times = dataset["times"]
    queries = {name: query_bundle(dataset[name]["paths"], times, {
        "drift": state_f, "diffusion": state_g, "drift_truth": truth_f,
        "diffusion_truth": truth_g, "drift_time": time_f, "diffusion_time": time_g,
    }) for name in ("train", "validation", "test")}

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
            "time_drift_vs_truth": field_block(queries[name]["drift_truth"],
                                               queries[name]["drift_time"]),
            "time_diffusion_vs_truth": field_block(queries[name]["diffusion_truth"],
                                                   queries[name]["diffusion_time"]),
            "fraction_symbolic_g_nonpositive": float(np.mean(np.any(g_hat <= 0.0, axis=1))),
        }

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
    oracle_cut = float(dist["oracle_exact_nrmse_max"])
    ceilings["oracle_exact"] = bool(
        ceilings["drift_oracle_vs_truth"] <= oracle_cut
        and ceilings["diffusion_oracle_vs_truth"] <= oracle_cut)
    vif = vif_max(queries["train"]["features"], 3)

    dt = float(dataset["dt"])
    f_norm = np.linalg.norm(queries["test"]["drift"], axis=1)
    g_norm = np.linalg.norm(queries["test"]["diffusion"], axis=1)
    g_truth_norm = np.linalg.norm(queries["test"]["diffusion_truth"], axis=1)
    snr = {"mean_abs_f_dt": float(np.mean(f_norm) * dt),
           "mean_g_sqrt_dt": float(np.mean(g_norm) * math.sqrt(dt)),
           "mean_truth_g_sqrt_dt": float(np.mean(g_truth_norm) * math.sqrt(dt)),
           "posterior_mean_sigma": sigma_probe["mean_sigma"]}
    snr["ratio_of_means_f_over_g"] = snr["mean_abs_f_dt"] / max(snr["mean_g_sqrt_dt"], 1e-15)
    snr["posterior_sigma_over_g_sqrt_dt"] = (
        sigma_probe["mean_sigma"] / max(snr["mean_g_sqrt_dt"], 1e-15))

    radius = float(cfg["simulation"]["failure_radius"])
    excursion_radius = float(cfg["simulation"]["excursion_radius"])
    test = dataset["test"]
    inc_rng = np.random.default_rng(int(cfg["simulation"]["discretization_seed"]) + 11)
    paired_inc = inc_rng.standard_normal((len(test["x0"]), dataset["n_steps"], 2))

    def run_paired(drift_fn: Callable, diffusion_fn: Callable) -> dict[str, Any]:
        return simulate_population(drift_fn, diffusion_fn, test["x0"], dt, paired_inc,
                                   radius, excursion_radius)

    sym_f = polynomial_call(drift_field)
    sym_g = polynomial_call(g_field)
    sim_truth = run_paired(truth_f, truth_g)
    sim_teacher = run_paired(state_f, state_g)
    sim_symbolic = run_paired(sym_f, sym_g)

    def paired_metrics(sim: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
        cov_s, n_s = terminal_cov(sim["states"])
        cov_r, n_r = terminal_cov(reference["states"])
        cov_gap = None if cov_s is None or cov_r is None else rel_frob(cov_s, cov_r)
        return {"counts": failure_counts(sim),
                "mean_rmse": mean_rmse(mean_path(sim["states"]), mean_path(reference["states"])),
                "terminal_cov_rel_frob": cov_gap,
                "terminal_finite_n": [n_s, n_r]}

    paired = {"teacher_vs_truth": paired_metrics(sim_teacher, sim_truth),
              "symbolic_vs_teacher": paired_metrics(sim_symbolic, sim_teacher),
              "symbolic_vs_truth": paired_metrics(sim_symbolic, sim_truth),
              "truth_counts": failure_counts(sim_truth)}

    fine_factor = int(cfg["simulation"]["fine_factor"])
    fine_dt = dt / fine_factor
    n_fine = dataset["n_steps"] * fine_factor
    disc_rng = np.random.default_rng(int(cfg["simulation"]["discretization_seed"]))
    fine_inc = disc_rng.standard_normal((len(test["x0"]), n_fine, 2))
    coarse_inc = coarse_increments(fine_inc, fine_factor)
    sim_fine = simulate_population(state_f, state_g, test["x0"], fine_dt, fine_inc,
                                   radius, excursion_radius)
    sim_coarse = simulate_population(state_f, state_g, test["x0"], dt, coarse_inc,
                                     radius, excursion_radius)
    both = (np.isfinite(sim_fine["final"]).all(axis=1)
            & np.isfinite(sim_coarse["final"]).all(axis=1))
    if both.any():
        terminal_gap = sim_coarse["final"][both] - sim_fine["final"][both]
        disc_rmse = float(np.sqrt(np.mean(np.sum(terminal_gap ** 2, axis=1))))
    else:
        disc_rmse = float("nan")
    discretization = {"terminal_rmse": disc_rmse,
                      "retained_fraction": float(np.mean(both)),
                      "fine_failure_rate": sim_fine["failure_rate"],
                      "coarse_failure_rate": sim_coarse["failure_rate"]}

    law_seed = int(cfg["simulation"]["law_seed"])
    n_law = len(test["x0"])
    law_truth = fresh_paths(truth_f, truth_g, n_law, dt, dataset["n_steps"],
                            law_seed, cfg, radius, excursion_radius)
    law_teacher = fresh_paths(state_f, state_g, n_law, dt, dataset["n_steps"],
                              law_seed, cfg, radius, excursion_radius)
    law_symbolic = fresh_paths(sym_f, sym_g, n_law, dt, dataset["n_steps"],
                               law_seed, cfg, radius, excursion_radius)
    observed = test["paths"]

    def law_cov(sim: dict[str, Any]) -> float | None:
        cov_s, _n_s = terminal_cov(sim["states"])
        cov_o, _n_o = terminal_cov(observed)
        if cov_s is None or cov_o is None:
            return None
        return rel_frob(cov_s, cov_o)

    law = {
        "truth_counts": failure_counts(law_truth),
        "teacher_counts": failure_counts(law_teacher),
        "symbolic_counts": failure_counts(law_symbolic),
        "mean_rmse_teacher_vs_observed": mean_rmse(mean_path(law_teacher["states"]),
                                                   mean_path(observed)),
        "mean_rmse_symbolic_vs_observed": mean_rmse(mean_path(law_symbolic["states"]),
                                                    mean_path(observed)),
        "mean_rmse_truth_vs_observed": mean_rmse(mean_path(law_truth["states"]),
                                                 mean_path(observed)),
        "terminal_cov_teacher_vs_observed": law_cov(law_teacher),
        "terminal_cov_symbolic_vs_observed": law_cov(law_symbolic),
        "terminal_cov_truth_vs_observed": law_cov(law_truth),
    }

    grid = diagnostic_grid(dataset["train"]["paths"].reshape(-1, 2),
                           [0.0, float(data_cfg["T"]) / 2.0, float(data_cfg["T"])],
                           state_f, state_g, drift_field, g_field, truth_f, truth_g)
    expanded_f = expand_to_original(drift_field)
    expanded_g = expand_to_original(g_field)
    coefficient_diagnostic = {
        "note": "Not a hypothesis. Closed-form reading is disallowed. Terms absent from the polynomial are stored as 0.",
        "truth_drift": {"x": alpha, "xy": -beta, "y": -gamma, "prey_xy": delta},
        "symbolic_drift_f1": {
            "const": monomial_coefficient(expanded_f[0], (0, 0, 0)),
            "x": monomial_coefficient(expanded_f[0], (1, 0, 0)),
            "y": monomial_coefficient(expanded_f[0], (0, 1, 0)),
            "xy": monomial_coefficient(expanded_f[0], (1, 1, 0)),
            "t": monomial_coefficient(expanded_f[0], (0, 0, 1)),
        },
        "symbolic_drift_f2": {
            "const": monomial_coefficient(expanded_f[1], (0, 0, 0)),
            "x": monomial_coefficient(expanded_f[1], (1, 0, 0)),
            "y": monomial_coefficient(expanded_f[1], (0, 1, 0)),
            "xy": monomial_coefficient(expanded_f[1], (1, 1, 0)),
            "t": monomial_coefficient(expanded_f[1], (0, 0, 1)),
        },
        "truth_diffusion": {"g1_x": sigma, "g2_y": sigma},
        "symbolic_diffusion_g1_x": monomial_coefficient(expanded_g[0], (1, 0, 0)),
        "symbolic_diffusion_g2_y": monomial_coefficient(expanded_g[1], (0, 1, 0)),
    }
    points = np.array([[1.0, 0.5], [0.9, 0.2], [1.5, 0.8]])
    point_probes = []
    for row in points:
        tt = np.array([0.0])
        feat = np.array([[row[0], row[1], 0.0]])
        point_probes.append({
            "x": row.tolist(), "t": 0.0,
            "truth_f": truth_f(row.reshape(1, 2), tt)[0].tolist(),
            "teacher_f": state_f(row.reshape(1, 2), tt)[0].tolist(),
            "symbolic_f": drift_field.predict(feat)[0].tolist(),
            "truth_g": truth_g(row.reshape(1, 2), tt)[0].tolist(),
            "teacher_g": state_g(row.reshape(1, 2), tt)[0].tolist(),
            "symbolic_g": g_field.predict(feat)[0].tolist(),
            "timeonly_g": time_g(row.reshape(1, 2), tt)[0].tolist(),
        })

    test_m = heldout["test"]
    metrics = {
        "drift_sym_teacher": test_m["drift_symbolic_vs_teacher"]["nrmse"],
        "drift_teacher_truth": test_m["drift_teacher_vs_truth"]["nrmse"],
        "drift_sym_truth": test_m["drift_symbolic_vs_truth"]["nrmse"],
        "g_sym_teacher": test_m["diffusion_symbolic_vs_teacher"]["nrmse"],
        "g_teacher_truth": test_m["diffusion_teacher_vs_truth"]["nrmse"],
        "g_sym_truth": test_m["diffusion_symbolic_vs_truth"]["nrmse"],
        "sigma_sym_teacher": test_m["sigma_symbolic_vs_teacher"]["nrmse"],
        "fraction_g_nonpositive": test_m["fraction_symbolic_g_nonpositive"],
        "vif": vif,
        "fail_truth": sim_truth["failure_rate"],
        "fail_teacher": sim_teacher["failure_rate"],
        "fail_symbolic": sim_symbolic["failure_rate"],
        "exc_teacher": sim_teacher["excursion_rate"],
        "exc_symbolic": sim_symbolic["excursion_rate"],
        "mean_sym_teacher": paired["symbolic_vs_teacher"]["mean_rmse"],
        "mean_teacher_truth": paired["teacher_vs_truth"]["mean_rmse"],
        "cov_sym_teacher": paired["symbolic_vs_teacher"]["terminal_cov_rel_frob"],
        "disc_rmse": disc_rmse,
        "disc_retained": discretization["retained_fraction"],
        "law_fail_truth": law_truth["failure_rate"],
        "law_fail_teacher": law_teacher["failure_rate"],
        "law_fail_symbolic": law_symbolic["failure_rate"],
        "law_mean_teacher": law["mean_rmse_teacher_vs_observed"],
        "law_mean_symbolic": law["mean_rmse_symbolic_vs_observed"],
        "law_cov_teacher": law["terminal_cov_teacher_vs_observed"],
        "law_cov_symbolic": law["terminal_cov_symbolic_vs_observed"],
        "floor_mean": law["mean_rmse_truth_vs_observed"],
        "floor_cov": law["terminal_cov_truth_vs_observed"],
        "g_state_truth": test_m["diffusion_teacher_vs_truth"]["nrmse"],
        "g_time_truth": test_m["time_diffusion_vs_truth"]["nrmse"],
    }
    verdicts = decide(cfg, metrics)
    train_norm = float(np.max(np.linalg.norm(dataset["train"]["paths"], axis=-1)))
    results: dict[str, Any] = {
        "day": 13,
        "date": cfg["date"],
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
        "train_max_norm": train_norm,
        "probe": probe,
        "posterior_probe": sigma_probe,
        "state_training": state_log,
        "time_training": time_log,
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
        "coefficient_diagnostic": coefficient_diagnostic,
        "query_dropped": {name: queries[name]["dropped"] for name in ("train", "validation", "test")},
        "verdicts": verdicts,
        "elapsed_seconds": float(time.time() - started),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drift_field.save(OUT_DIR / "symbolic_f.json")
    g_field.save(OUT_DIR / "symbolic_g.json")
    (OUT_DIR / "teacher_state_weights.json").write_text(json.dumps(state_weights))
    (OUT_DIR / "teacher_time_weights.json").write_text(json.dumps(time_weights))
    tex = (
        "Symbolic SDE in the original Lotka-Volterra coordinates (x, y, t).\n"
        "This is the polynomial fit to the state-dependent SDE Matching teacher.\n"
        "It is not a claim that the closed-form coefficients were recovered.\n"
        "t is a coordinate because the teacher networks take time.\n\n$$\n"
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
        "test": {key: test_m[key]["nrmse"] if isinstance(test_m[key], dict) and "nrmse" in test_m[key]
                 else test_m[key]
                 for key in ("drift_symbolic_vs_teacher", "drift_teacher_vs_truth",
                             "drift_symbolic_vs_truth", "diffusion_symbolic_vs_teacher",
                             "diffusion_teacher_vs_truth", "diffusion_symbolic_vs_truth")},
        "ceilings": ceilings,
        "vif": vif,
        "snr": snr,
        "failures": {"truth": sim_truth["failure_rate"],
                     "teacher": sim_teacher["failure_rate"],
                     "symbolic": sim_symbolic["failure_rate"]},
        "elapsed_seconds": results["elapsed_seconds"],
    }), indent=2), flush=True)
    return results


if __name__ == "__main__":
    main()
