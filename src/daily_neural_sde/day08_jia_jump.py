"""Day 8: Jia & Benson (2019) Neural Jump SDEs -> polynomial SINDy.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day08.json``):

1. Ground truth: an identifiable observable 1D jump-diffusion in the
   paper's jump form (state-dependent intensity lambda(x), jump size
   function w), with a Brownian diffusion term added (declared
   deviation: the paper's continuous part is a deterministic ODE).
   dX = -0.60 X dt + 0.35 dW + J dN(lambda(X)),
   lambda(x) = 0.25 + 0.08 x^2, J ~ N(-0.80, 0.50^2).
   Trajectories are simulated by Euler-Maruyama at dt_fine = 0.01
   (per fine step, a jump occurs with probability lambda(x) dt_fine)
   and observed every 0.05 on [0, 10]; initial states are Normal(0, 1)
   (Day 5 reflection: many initial conditions, never one).
   The paper's latent point-process pipeline, adjoint training through
   event times, and marked point-process benchmarks are cut and
   declared in the config: the teacher is reconstructed in
   identifiable coordinates.
2. Teachers (all transition likelihood on train transitions only):
   a jump-mixture EM teacher (two-component Gaussian mixture density:
   no-jump vs one-jump, jump probability 1 - exp(-lambda dt)) and a
   diffusion-only Gaussian EM teacher (the paper-axis baseline).
   A jump-mixture EM pair at dt = 0.5 vs transition-count-matched
   dt = 0.05 is the Day 1 discretization diagnostic (H6), regime-split
   by |x| with per-transition SNR quantified.
3. Distillation of the jump teacher into polynomial SINDy: drift
   candidates include free and origin-pinned libraries (Day 7
   structural-constraint reflection, implemented as mean-0
   standardization with no constant term, so the expanded polynomial
   genuinely vanishes at x = 0); lambda is fitted directly so
   its negativity stays visible; g, mu_J, sigma_J each have their own
   library. Direct function queries on trajectory-disjoint splits,
   validation-only selection, test once. A collinearity gate at the
   maximum selectable degree (3) precedes fitting (Day 1 / Day 7):
   because it forbids a coefficient verdict on the selected model,
   H3c uses a pre-frozen degree-2 no-intercept diagnostic fit.
   A wide grid x in [-4, 4] is the Day 2 off-manifold diagnostic,
   reported only, never used for selection (Day 5).
4. Validation by paired jump-diffusion simulation from test initial
   states with shared Brownian increments, shared uniform jump draws
   and shared jump-size normals, retaining failed-trajectory mass
   and counting |X| excursions (> 8) and symbolic-lambda clipping
   (negative lambda clipped at 0) as domain measures separate from
   failure (Day 4). Paper axis: H5 compares the jump teacher against
   the diffusion-only teacher on absolute test-transition NLL levels
   and on simulated vs observed large-increment rates (Day 7:
   absolute levels pre-decided as the main clause).

Run from the repository root:
``python -m src.daily_neural_sde.day08_jia_jump``
Outputs land in ``outputs/daily_neural_sde/day08/``.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, evaluate_powers, expand_to_original,
                     monomial_powers, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day08.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day08"

DT_FINE = 0.01
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


# ---------------------------------------------------------------- truth

def true_drift(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return -0.60 * x


def true_g(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.full((len(x), 1), 0.35)


def true_lambda(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return 0.25 + 0.08 * x[:, [0]] ** 2


def true_mu(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.full((len(x), 1), -0.80)


def true_sigma_j(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.full((len(x), 1), 0.50)


def generate_data(cfg: dict) -> dict[str, Any]:
    """Simulate jump-diffusion trajectories at dt_fine, observe every dt_obs."""
    rng = np.random.default_rng(int(cfg["seed"]))
    n_paths = int(cfg["data"]["n_trajectories"])
    t_end = float(cfg["data"]["t_end"])
    dt_obs = float(cfg["data"]["dt_obs"])
    n_fine = int(round(t_end / DT_FINE))
    obs_every = int(round(dt_obs / DT_FINE))
    n_obs = int(round(t_end / dt_obs)) + 1
    x0 = rng.standard_normal((n_paths, 1))
    buf = np.zeros((n_paths, n_obs, 1))
    buf[:, 0, :] = x0
    cur = x0.copy()
    obs_idx = 1
    for k in range(1, n_fine + 1):
        lam = true_lambda(cur)[:, 0]
        jump = rng.random(n_paths) < lam * DT_FINE
        sizes = rng.standard_normal(n_paths) * 0.50 - 0.80
        dW = rng.standard_normal(n_paths) * np.sqrt(DT_FINE)
        cur = cur + true_drift(cur) * DT_FINE + 0.35 * dW[:, None]
        cur[jump, 0] += sizes[jump]
        if k % obs_every == 0:
            buf[:, obs_idx, :] = cur
            obs_idx += 1
    assert obs_idx == n_obs
    return {"obs": buf, "x0": x0}


def build_queries(obs: np.ndarray, dt_obs: float) -> dict[str, np.ndarray]:
    n_paths, n_obs, _ = obs.shape
    times = np.arange(n_obs) * dt_obs
    cur = obs.transpose(1, 0, 2).reshape(-1, 1)
    t = np.repeat(times, n_paths)
    return {"cur": cur, "t": t}


def build_transitions(obs: np.ndarray, dt_obs: float) -> dict[str, np.ndarray]:
    n_paths, n_obs, _ = obs.shape
    cur = obs[:, :-1, :].transpose(1, 0, 2).reshape(-1, 1)
    nxt = obs[:, 1:, :].transpose(1, 0, 2).reshape(-1, 1)
    t = np.repeat(np.arange(n_obs - 1) * dt_obs, n_paths)
    dt = np.full(len(cur), dt_obs)
    return {"cur": cur, "nxt": nxt, "t": t, "dt": dt}


def subsample_transitions(tr: dict, n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tr["cur"]), size=n, replace=False)
    return {k: v[idx] for k, v in tr.items()}


def coarse_transitions(obs: np.ndarray, dt_obs: float, step: int) -> dict[str, np.ndarray]:
    """Transitions joining every ``step``-th observation (dt = step * dt_obs)."""
    sub = obs[:, ::step, :]
    n_paths, n_obs, _ = sub.shape
    cur = sub[:, :-1, :].transpose(1, 0, 2).reshape(-1, 1)
    nxt = sub[:, 1:, :].transpose(1, 0, 2).reshape(-1, 1)
    t = np.repeat(np.arange(n_obs - 1) * dt_obs * step, n_paths)
    dt = np.full(len(cur), dt_obs * step)
    return {"cur": cur, "nxt": nxt, "t": t, "dt": dt}


# ---------------------------------------------------------------- teachers

def _torch():
    import torch  # local import: tests run without torch
    return torch


class MLP:
    """Small tanh MLP with input standardization (train statistics)."""

    def __init__(self, in_dim: int, out_dim: int, hidden=(32, 32)):
        torch = _torch()
        self.torch = torch
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(torch.nn.Linear(d, h))
            layers.append(torch.nn.Tanh())
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)
        self.mean = np.zeros(in_dim)
        self.scale = np.ones(in_dim)

    def fit_standardizer(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0

    def _tensor(self, x: np.ndarray):
        torch = self.torch
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return torch.tensor(z, dtype=torch.float32)

    def forward_raw(self, x: np.ndarray):
        return self.net(self._tensor(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return self.net(self._tensor(x)).numpy().astype(np.float64)

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "weights": [p.detach().numpy().tolist()
                            for p in self.net.parameters()]}

    def load_state(self, state: dict[str, Any]) -> None:
        torch = self.torch
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.scale = np.asarray(state["scale"], dtype=np.float64)
        with torch.no_grad():
            for p, w in zip(self.net.parameters(), state["weights"]):
                p.copy_(torch.tensor(np.asarray(w, dtype=np.float32)))


def _softplus(raw):
    torch = _torch()
    return torch.nn.functional.softplus(raw)


def build_jump_teacher(tcfg: dict) -> dict:
    return {"f": MLP(1, 1, tuple(tcfg["hidden"])),
            "g": MLP(1, 1, (16,)),
            "lam": MLP(1, 1, (16,)),
            "mu": MLP(1, 1, (16,)),
            "sig": MLP(1, 1, (16,))}


def build_diffusion_teacher(tcfg: dict) -> dict:
    return {"f": MLP(1, 1, tuple(tcfg["hidden"])),
            "g": MLP(1, 1, (16,))}


def _train_jump(tr: dict, epochs: int, batch: int, lr: float, seed: int) -> dict:
    """Two-component Gaussian-mixture transition NLL for a jump teacher."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_jump_teacher({"hidden": [32, 32]})
    for net in model.values():
        net.fit_standardizer(tr["cur"])
    opt = torch.optim.Adam(list(itertools.chain.from_iterable(
        net.net.parameters() for net in model.values())), lr=lr)
    n = len(tr["cur"])
    last: dict[str, float] = {}
    for _ in range(epochs):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            cur = torch.tensor(tr["cur"][idx], dtype=torch.float32)
            nxt = torch.tensor(tr["nxt"][idx], dtype=torch.float32)
            dt = torch.tensor(tr["dt"][idx], dtype=torch.float32)[:, None]
            f = model["f"].forward_raw(tr["cur"][idx])
            g = _softplus(model["g"].forward_raw(tr["cur"][idx])) + 0.01
            lam = _softplus(model["lam"].forward_raw(tr["cur"][idx])) + 0.01
            mu = model["mu"].forward_raw(tr["cur"][idx])
            sig = _softplus(model["sig"].forward_raw(tr["cur"][idx])) + 0.05
            p_jump = 1.0 - torch.exp(-lam * dt)
            mean0 = cur + f * dt
            var0 = g ** 2 * dt + 1e-8
            mean1 = mean0 + mu
            var1 = var0 + sig ** 2
            log0 = -LOG_SQRT_2PI - 0.5 * torch.log(var0) \
                - 0.5 * (nxt - mean0) ** 2 / var0 + torch.log(1.0 - p_jump)
            log1 = -LOG_SQRT_2PI - 0.5 * torch.log(var1) \
                - 0.5 * (nxt - mean1) ** 2 / var1 + torch.log(p_jump)
            loss = -torch.mean(torch.logsumexp(
                torch.cat([log0, log1], dim=1), dim=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        last["loss"] = tot / max(cnt, 1)
    return {"model": model, "loss": last["loss"]}


def _train_diffusion(tr: dict, epochs: int, batch: int, lr: float, seed: int) -> dict:
    """Gaussian EM NLL for a diffusion-only teacher."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_diffusion_teacher({"hidden": [32, 32]})
    for net in model.values():
        net.fit_standardizer(tr["cur"])
    opt = torch.optim.Adam(list(itertools.chain.from_iterable(
        net.net.parameters() for net in model.values())), lr=lr)
    n = len(tr["cur"])
    last: dict[str, float] = {}
    for _ in range(epochs):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            cur = torch.tensor(tr["cur"][idx], dtype=torch.float32)
            nxt = torch.tensor(tr["nxt"][idx], dtype=torch.float32)
            dt = torch.tensor(tr["dt"][idx], dtype=torch.float32)[:, None]
            f = model["f"].forward_raw(tr["cur"][idx])
            g = _softplus(model["g"].forward_raw(tr["cur"][idx])) + 0.01
            var = g ** 2 * dt + 1e-8
            resid = nxt - cur - f * dt
            loss = torch.mean(LOG_SQRT_2PI + 0.5 * torch.log(var)
                              + 0.5 * resid ** 2 / var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        last["loss"] = tot / max(cnt, 1)
    return {"model": model, "loss": last["loss"]}


def jump_predict(model: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return {"f": model["f"].predict(x),
            "g": np.logaddexp(0.0, model["g"].predict(x)) + 0.01,
            "lam": np.logaddexp(0.0, model["lam"].predict(x)) + 0.01,
            "mu": model["mu"].predict(x),
            "sig": np.logaddexp(0.0, model["sig"].predict(x)) + 0.05}


def diffusion_predict(model: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return {"f": model["f"].predict(x),
            "g": np.logaddexp(0.0, model["g"].predict(x)) + 0.01}


def jump_nll_per_transition(model: dict, tr: dict) -> np.ndarray:
    """Mixture NLL per test transition (numpy, teacher nets in eval mode)."""
    pred = jump_predict(model, tr["cur"])
    dt = tr["dt"][:, None]
    p_jump = 1.0 - np.exp(-pred["lam"] * dt)
    mean0 = tr["cur"] + pred["f"] * dt
    var0 = pred["g"] ** 2 * dt + 1e-8
    mean1 = mean0 + pred["mu"]
    var1 = var0 + pred["sig"] ** 2
    log0 = -LOG_SQRT_2PI - 0.5 * np.log(var0) \
        - 0.5 * (tr["nxt"] - mean0) ** 2 / var0 + np.log(1.0 - p_jump)
    log1 = -LOG_SQRT_2PI - 0.5 * np.log(var1) \
        - 0.5 * (tr["nxt"] - mean1) ** 2 / var1 + np.log(p_jump)
    stacked = np.concatenate([log0, log1], axis=1)
    m = stacked.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(stacked - m).sum(axis=1)))


def diffusion_nll_per_transition(model: dict, tr: dict) -> np.ndarray:
    pred = diffusion_predict(model, tr["cur"])
    dt = tr["dt"][:, None]
    var = pred["g"] ** 2 * dt + 1e-8
    resid = tr["nxt"] - tr["cur"] - pred["f"] * dt
    return (LOG_SQRT_2PI + 0.5 * np.log(var)
            + 0.5 * resid ** 2 / var)[:, 0]


# ---------------------------------------------------------------- SINDy

def _stlsq(theta_n: np.ndarray, target_n: np.ndarray, threshold: float,
           ridge: float, max_iter: int = 20) -> np.ndarray:
    """STLSQ for one output in RMS-normalized units (mirrors common.py)."""
    n_terms = theta_n.shape[1]
    active = np.ones(n_terms, dtype=bool)
    coef = np.zeros(n_terms)
    for _ in range(max_iter):
        if not active.any():
            return np.zeros(n_terms)
        design = theta_n[:, active]
        gram = design.T @ design / len(design) + ridge * np.eye(active.sum())
        coef_active = np.linalg.solve(gram, design.T @ target_n / len(design))
        small = np.abs(coef_active) < threshold
        if not small.any():
            coef = np.zeros(n_terms)
            coef[active] = coef_active
            break
        idx = np.flatnonzero(active)
        active[idx[small]] = False
        coef = np.zeros(n_terms)
        coef[active] = coef_active[~small]
    if active.any():
        design = theta_n[:, active]
        solution, *_ = np.linalg.lstsq(design, target_n, rcond=None)
        coef = np.zeros(n_terms)
        coef[active] = solution
    return coef


def fit_field_custom(x: np.ndarray, y: np.ndarray, degree: int,
                     threshold: float, ridge: float,
                     intercept: bool, origin: bool = False) -> PolynomialField:
    """Polynomial STLSQ with optional intercept / origin libraries.

    ``intercept=False`` drops the constant term in *standardized*
    coordinates (which pins the fit at the train mean, not at the
    origin). ``origin=True`` is the structural constraint for a field
    vanishing at the origin: standardization uses mean 0 (scale only)
    and the constant term is dropped, so the expanded original-coordinate
    polynomial has no intercept either.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if origin:
        mean = np.zeros(x.shape[1])
    else:
        mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    powers = monomial_powers(x.shape[1], degree)
    if not intercept or origin:
        powers = powers[powers.sum(axis=1) > 0]
    theta = evaluate_powers(z, powers)
    feature_rms = np.sqrt(np.mean(theta ** 2, axis=0))
    feature_rms[feature_rms < 1e-12] = 1.0
    target_rms = np.sqrt(np.mean(y ** 2, axis=0))
    target_rms[target_rms < 1e-12] = 1.0
    theta_n = theta / feature_rms[None, :]
    coefficients = np.zeros((y.shape[1], len(powers)))
    for j in range(y.shape[1]):
        coef_n = _stlsq(theta_n, y[:, j] / target_rms[j], threshold, ridge)
        coefficients[j] = coef_n * target_rms[j] / feature_rms
    metadata = {"degree": degree, "threshold": threshold, "ridge": ridge,
                "intercept": intercept, "origin": origin,
                "library_size": int(len(powers)),
                "active_terms": int(np.count_nonzero(coefficients)),
                "optimizer": "numpy STLSQ (RMS-normalized, unbias refit)",
                "feature_rms": feature_rms.tolist(),
                "target_rms": target_rms.tolist()}
    return PolynomialField(powers, coefficients, mean, scale, metadata)
    theta = evaluate_powers(z, powers)
    feature_rms = np.sqrt(np.mean(theta ** 2, axis=0))
    feature_rms[feature_rms < 1e-12] = 1.0
    target_rms = np.sqrt(np.mean(y ** 2, axis=0))
    target_rms[target_rms < 1e-12] = 1.0
    theta_n = theta / feature_rms[None, :]
    coefficients = np.zeros((y.shape[1], len(powers)))
    for j in range(y.shape[1]):
        coef_n = _stlsq(theta_n, y[:, j] / target_rms[j], threshold, ridge)
        coefficients[j] = coef_n * target_rms[j] / feature_rms
    metadata = {"degree": degree, "threshold": threshold, "ridge": ridge,
                "intercept": intercept, "library_size": int(len(powers)),
                "active_terms": int(np.count_nonzero(coefficients)),
                "optimizer": "numpy STLSQ (RMS-normalized, unbias refit)",
                "feature_rms": feature_rms.tolist(),
                "target_rms": target_rms.tolist()}
    return PolynomialField(powers, coefficients, mean, scale, metadata)


def fit_field(x: np.ndarray, y: np.ndarray, x_val: np.ndarray,
              y_val: np.ndarray, degrees, thresholds, ridge: float,
              intercept_options=(True,), origin_options=(False,)) -> tuple:
    """Validation-selected polynomial field; returns (field, val, meta)."""
    best = None
    for degree in degrees:
        for intercept in intercept_options:
            for origin in origin_options:
                for thr in thresholds:
                    field = fit_field_custom(x, y, degree, thr, ridge,
                                             intercept, origin)
                    val = nrmse(y_val, field.predict(x_val))
                    if best is None or val < best[1]:
                        best = (field, val, {"degree": degree, "threshold": thr,
                                             "intercept": intercept,
                                             "origin": origin})
    assert best is not None
    return best


def vif_scores(x: np.ndarray) -> np.ndarray:
    """Variance inflation factors for columns of x."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    out = np.zeros(d)
    for j in range(d):
        others = np.delete(x, j, axis=1)
        design = np.concatenate([others, np.ones((len(x), 1))], axis=1)
        coef, *_ = np.linalg.lstsq(design, x[:, j], rcond=None)
        resid = x[:, j] - design @ coef
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((x[:, j] - x[:, j].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
        out[j] = 1.0 / max(1.0 - r2, 1e-9)
    return out


# ---------------------------------------------------------------- simulation

def simulate_jump(drift_fn, diff_fn, lam_fn, mu_fn, sig_fn, x0: np.ndarray,
                  dt: float, n_steps: int, dW: np.ndarray,
                  uniforms: np.ndarray, jump_normals: np.ndarray,
                  failure_radius: float, excursion_radius: float) -> dict[str, Any]:
    """Paired jump-diffusion Euler simulation with jump clipping counted.

    Each model decides jumps from its own lambda(x) (uniform < 1 -
    exp(-lambda dt)); lambda is clipped at 0 and sigma_J at 0.01, and
    clipped steps are counted as a domain measure separate from
    failure. Failures latch (non-finite or |X| > failure_radius).
    """
    x = np.array(x0, dtype=np.float64, copy=True).reshape(-1, 1)
    n = len(x)
    sim = np.zeros((n, n_steps + 1, 1))
    sim[:, 0, :] = x
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    ever_excursion = np.zeros(n, dtype=bool)
    clipped_steps = 0
    total_steps = 0
    jump_steps = 0
    large_steps = 0
    prev = x.copy()
    for k in range(1, n_steps + 1):
        active = ~failed
        if active.any():
            xa = x[active]
            f = drift_fn(xa)
            g = diff_fn(xa)
            lam_raw = lam_fn(xa)[:, 0]
            sig_raw = sig_fn(xa)[:, 0]
            clipped_steps += int(np.sum(lam_raw < 0) + np.sum(sig_raw < 0.01))
            total_steps += int(active.sum())
            lam = np.maximum(lam_raw, 0.0)
            sig = np.maximum(sig_raw, 0.01)
            mu = mu_fn(xa)[:, 0]
            is_jump = uniforms[active, k - 1] < 1.0 - np.exp(-lam * dt)
            jump_steps += int(np.sum(is_jump))
            step = f[:, 0] * dt + g[:, 0] * np.sqrt(dt) * dW[active, k - 1]
            step[is_jump] += mu[is_jump] + sig[is_jump] * jump_normals[active, k - 1][is_jump]
            new = xa[:, 0] + step
            large_steps += int(np.sum(np.abs(step) > 0.5))
            x[active, 0] = new
        radii = np.abs(x[:, 0])
        ever_excursion = ever_excursion | ((radii > excursion_radius) & active)
        bad_nonfinite = ~np.isfinite(x[:, 0])
        bad_radius = radii > failure_radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        x[failed] = np.nan
        sim[:, k, :] = x
        prev = x.copy()
    return {"states": sim, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion_rate": float(np.mean(ever_excursion)),
            "jump_step_rate": jump_steps / max(n * n_steps, 1),
            "large_step_rate": large_steps / max(n * n_steps, 1),
            "clip_rate": clipped_steps / max(total_steps, 1)}


# ---------------------------------------------------------------- main

def _split_indices(n: int, cfg: dict, rng: np.random.Generator):
    perm = rng.permutation(n)
    n_train = int(cfg["split"]["train"])
    n_val = int(cfg["split"]["validation"])
    return (perm[:n_train], perm[n_train:n_train + n_val],
            perm[n_train + n_val:n_train + n_val + int(cfg["split"]["test"])])


def _field_table(pred: np.ndarray, ref: np.ndarray) -> dict:
    return {"nrmse": nrmse(ref, pred),
            "rms_error": float(np.sqrt(np.mean((pred - ref) ** 2))),
            "ref_rms": float(np.sqrt(np.mean(ref ** 2)))}


def _paired_se(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    return float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0


def main() -> dict[str, Any]:
    t_start = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))
    data = generate_data(cfg)
    obs = data["obs"]
    dt_obs = float(cfg["data"]["dt_obs"])
    idx_tr, idx_va, idx_te = _split_indices(len(obs), cfg, rng)

    queries = {name: build_queries(obs[idx], dt_obs)
               for name, idx in (("train", idx_tr), ("validation", idx_va),
                                 ("test", idx_te))}
    tr_all = build_transitions(obs[idx_tr], dt_obs)
    tr_test = build_transitions(obs[idx_te], dt_obs)
    tr_coarse = coarse_transitions(obs[idx_tr], dt_obs, 10)
    tr_fine_sub = subsample_transitions(tr_all, len(tr_coarse["cur"]), seed=7)

    em = cfg["teacher"]["em_main"]
    jump = _train_jump(tr_all, int(em["epochs"]), int(em["batch"]),
                       float(em["lr"]), seed=0)
    diff_teacher = _train_diffusion(tr_all, int(em["epochs"]), int(em["batch"]),
                                    float(em["lr"]), seed=1)
    em_c = cfg["teacher"]["em_coarse"]
    coarse = _train_jump(tr_coarse, int(em_c["epochs"]), int(em_c["batch"]),
                         float(em_c["lr"]), seed=2)
    em_f = cfg["teacher"]["em_fine_sub"]
    fine_sub = _train_jump(tr_fine_sub, int(em_f["epochs"]), int(em_f["batch"]),
                           float(em_f["lr"]), seed=3)
    jump_model = jump["model"]

    # Teacher queries on each split.
    teacher_q = {}
    for name, q in queries.items():
        teacher_q[name] = jump_predict(jump_model, q["cur"])
    truth_q = {}
    for name, q in queries.items():
        truth_q[name] = {"f": true_drift(q["cur"]), "g": true_g(q["cur"]),
                         "lam": true_lambda(q["cur"]),
                         "mu": true_mu(q["cur"]), "sig": true_sigma_j(q["cur"])}

    # Collinearity gate at the maximum selectable degree (3), train queries.
    # The constant (intercept) column is excluded: its VIF is meaningless
    # (a constant is perfectly explained by the regression intercept).
    z_tr = (queries["train"]["cur"] - queries["train"]["cur"].mean(axis=0)) \
        / queries["train"]["cur"].std(axis=0)
    powers3 = monomial_powers(1, 3)
    theta3 = evaluate_powers(z_tr, powers3)[:, 1:]
    vif3 = vif_scores(theta3).tolist()
    powers2 = monomial_powers(1, 2)
    vif2 = vif_scores(evaluate_powers(z_tr, powers2)[:, 1:]).tolist()

    dist = cfg["distillation"]
    degrees = dist["degrees"]
    thresholds = dist["thresholds"]
    ridge = float(dist["ridge"])
    sym = {}
    sym_meta = {}
    for field_name, cfg_degrees, intercepts, origins in (
            ("f", degrees["drift_free"], (True,), (False, True)),
            ("g", degrees["diffusion"], (True,), (False,)),
            ("lam", degrees["lambda"], (True,), (False,)),
            ("mu", degrees["jump_mean"], (True,), (False,)),
            ("sig", degrees["jump_size_sd"], (True,), (False,))):
        field, val, meta = fit_field(
            queries["train"]["cur"], teacher_q["train"][field_name],
            queries["validation"]["cur"], teacher_q["validation"][field_name],
            cfg_degrees, thresholds, ridge, intercepts, origins)
        sym[field_name] = field
        sym_meta[field_name] = {"validation_nrmse_vs_teacher": val, **meta,
                                "active_terms": int(np.count_nonzero(field.coefficients))}
        field.save(OUT_DIR / f"symbolic_{field_name}.json")

    # Pre-frozen diagnostic fits for the coefficient verdicts.
    # The drift diagnostic is origin-pinned (mean 0, no intercept), so its
    # expanded original-coordinate polynomial genuinely vanishes at x = 0.
    diag_drift = fit_field_custom(queries["train"]["cur"], teacher_q["train"]["f"],
                                  2, 0.05, ridge, intercept=False, origin=True)
    diag_lam = fit_field_custom(queries["train"]["cur"], teacher_q["train"]["lam"],
                                2, 0.05, ridge, intercept=True)
    # Log-lambda diagnostic fit (never selected).
    log_lam_tr = np.log(np.maximum(teacher_q["train"]["lam"], 1e-6))
    log_lam_va = np.log(np.maximum(teacher_q["validation"]["lam"], 1e-6))
    log_lam_field, log_lam_val, log_lam_meta = fit_field(
        queries["train"]["cur"], log_lam_tr,
        queries["validation"]["cur"], log_lam_va,
        [2], thresholds, ridge, (True,))

    # Test evaluation (once, after selection).
    results: dict[str, Any] = {
        "config": "configs/daily_neural_sde_day08.json",
        "teacher_train_loss": {"jump_main": jump["loss"],
                               "diffusion_only": diff_teacher["loss"],
                               "jump_coarse": coarse["loss"],
                               "jump_fine_sub": fine_sub["loss"]},
        "vif_degree3_train": vif3, "vif_degree2_train": vif2,
        "symbolic_meta": sym_meta,
    }
    for field_name in ("f", "g", "lam", "mu", "sig"):
        pred_te = sym[field_name].predict(queries["test"]["cur"])
        results[f"symbolic_vs_teacher_test_{field_name}"] = _field_table(
            pred_te, teacher_q["test"][field_name])
        results[f"symbolic_vs_truth_test_{field_name}"] = _field_table(
            pred_te, truth_q["test"][field_name])
        pred_tr_teacher = jump_predict(jump_model, queries["test"]["cur"])[field_name]
        results[f"teacher_vs_truth_test_{field_name}"] = _field_table(
            pred_tr_teacher, truth_q["test"][field_name])
    # Implied Sigma (g^2) in 1D.
    g_sym_te = sym["g"].predict(queries["test"]["cur"])
    results["symbolic_sigma_vs_teacher_test"] = _field_table(
        g_sym_te ** 2, teacher_q["test"]["g"] ** 2)
    results["symbolic_sigma_vs_truth_test"] = _field_table(
        g_sym_te ** 2, truth_q["test"]["g"] ** 2)
    results["teacher_sigma_vs_truth_test"] = _field_table(
        teacher_q["test"]["g"] ** 2, truth_q["test"]["g"] ** 2)

    # Oracle diagnostic: same procedure fitted to truth values.
    oracle = {}
    for field_name, cfg_degrees, intercepts in (
            ("f", degrees["drift_free"], (True, False)),
            ("g", degrees["diffusion"], (True,)),
            ("lam", degrees["lambda"], (True,))):
        field, val, meta = fit_field(
            queries["train"]["cur"], truth_q["train"][field_name],
            queries["validation"]["cur"], truth_q["validation"][field_name],
            cfg_degrees, thresholds, ridge, intercepts)
        oracle[field_name] = {"validation_nrmse_vs_truth": val, **meta}
        pred_te = field.predict(queries["test"]["cur"])
        oracle[field_name]["test_nrmse_vs_truth"] = _field_table(
            pred_te, truth_q["test"][field_name])["nrmse"]
    results["oracle"] = oracle

    # Coefficient verdicts (pre-frozen diagnostic fits).
    exp_drift = expand_to_original(diag_drift)[0]
    exp_lam = expand_to_original(diag_lam)[0]
    results["diag_drift_expanded"] = {str(k): v for k, v in exp_drift.items()}
    results["diag_lambda_expanded"] = {str(k): v for k, v in exp_lam.items()}
    results["diag_drift_test_nrmse_vs_teacher"] = _field_table(
        diag_drift.predict(queries["test"]["cur"]), teacher_q["test"]["f"])
    results["log_lambda_diagnostic"] = {
        "validation_nrmse": log_lam_val, **log_lam_meta,
        "test_nrmse_vs_log_teacher": _field_table(
            log_lam_field.predict(queries["test"]["cur"]),
            np.log(np.maximum(teacher_q["test"]["lam"], 1e-6)))["nrmse"]}

    # Off-manifold grid x in [-4, 4].
    grid = np.linspace(-4.0, 4.0, 81)[:, None]
    grid_teacher = jump_predict(jump_model, grid)
    grid_sym = {k: sym[k].predict(grid) for k in ("f", "g", "lam")}
    grid_truth = {"f": true_drift(grid), "g": true_g(grid),
                  "lam": true_lambda(grid)}
    grid_out: dict[str, Any] = {}
    for k in ("f", "g", "lam"):
        grid_out[f"symbolic_vs_teacher_{k}"] = _field_table(grid_sym[k], grid_teacher[k])
        grid_out[f"symbolic_vs_truth_{k}"] = _field_table(grid_sym[k], grid_truth[k])
        grid_out[f"teacher_vs_truth_{k}"] = _field_table(grid_teacher[k], grid_truth[k])
    grid_out["symbolic_lambda_negative_fraction"] = float(
        np.mean(grid_sym["lam"] < 0))
    grid_out["symbolic_lambda_negative_count"] = int(np.sum(grid_sym["lam"] < 0))
    results["grid"] = grid_out

    # Discretization diagnostic H6: coarse vs matched-fine jump teachers.
    h6: dict[str, Any] = {}
    for label, model in (("coarse", coarse["model"]), ("fine_sub", fine_sub["model"])):
        pred = jump_predict(model, queries["test"]["cur"])
        h6[label] = {
            "drift_vs_truth": _field_table(pred["f"], truth_q["test"]["f"]),
            "g_vs_truth": _field_table(pred["g"], truth_q["test"]["g"]),
            "sigma_vs_truth": _field_table(pred["g"] ** 2, truth_q["test"]["g"] ** 2),
            "lambda_vs_truth": _field_table(pred["lam"], truth_q["test"]["lam"]),
        }
    small = np.abs(queries["test"]["cur"][:, 0]) < 0.5
    large = np.abs(queries["test"]["cur"][:, 0]) > 1.5
    for regime, mask in (("small", small), ("large", large)):
        for label, model in (("coarse", coarse["model"]), ("fine_sub", fine_sub["model"])):
            pred = jump_predict(model, queries["test"]["cur"][mask])
            h6[f"{label}_drift_vs_truth_{regime}"] = _field_table(
                pred["f"], truth_q["test"]["f"][mask])
    # Per-transition SNR by regime (drift signal |f| dt vs diffusion sd g sqrt(dt)).
    snr: dict[str, Any] = {}
    for regime, mask in (("small", small), ("large", large)):
        f_abs = np.abs(truth_q["test"]["f"][mask])
        snr[regime] = {
            "n": int(mask.sum()),
            "mean_abs_f": float(np.mean(f_abs)),
            "per_transition_drift_snr_dt005": float(
                np.mean(f_abs * 0.05 / (0.35 * np.sqrt(0.05)))),
            "per_transition_drift_snr_dt05": float(
                np.mean(f_abs * 0.5 / (0.35 * np.sqrt(0.5)))),
            "mean_lambda": float(np.mean(truth_q["test"]["lam"][mask])),
        }
    h6["snr"] = snr
    h6["drift_degradation"] = (h6["coarse"]["drift_vs_truth"]["nrmse"]
                               - h6["fine_sub"]["drift_vs_truth"]["nrmse"])
    h6["drift_degradation_small"] = (h6["coarse_drift_vs_truth_small"]["nrmse"]
                                     - h6["fine_sub_drift_vs_truth_small"]["nrmse"])
    h6["drift_degradation_large"] = (h6["coarse_drift_vs_truth_large"]["nrmse"]
                                     - h6["fine_sub_drift_vs_truth_large"]["nrmse"])
    h6["regime_gap"] = h6["drift_degradation_large"] - h6["drift_degradation_small"]
    results["H6_discretization"] = h6

    # Jump-contamination diagnostic for g (Day 1 analogue).
    diff_pred_te = diffusion_predict(diff_teacher["model"], queries["test"]["cur"])
    results["g_contamination"] = {
        "diffusion_only_teacher_g_vs_truth": _field_table(
            diff_pred_te["g"], truth_q["test"]["g"]),
        "jump_teacher_g_vs_truth": results["teacher_vs_truth_test_g"],
    }
    screened = np.abs(tr_test["nxt"][:, 0] - tr_test["cur"][:, 0]) <= 0.4
    # Two-stage screened g: RMS of screened increments / sqrt(dt), per split-free
    # (test transitions only, diagnostic -- no selection uses it).
    screened_resid = (tr_test["nxt"][screened, 0] - tr_test["cur"][screened, 0]
                      - true_drift(tr_test["cur"][screened])[:, 0] * dt_obs)
    results["g_contamination"]["screened_two_stage_g_estimate"] = float(
        np.sqrt(np.mean(screened_resid ** 2) / dt_obs))
    results["g_contamination"]["screened_fraction"] = float(np.mean(screened))
    results["g_contamination"]["truth_g"] = 0.35

    # Paper axis H5: NLL levels and large-increment rates.
    nll_jump = jump_nll_per_transition(jump_model, tr_test)
    nll_diff = diffusion_nll_per_transition(diff_teacher["model"], tr_test)
    nll_diff_arr = nll_diff - nll_jump
    results["H5_nll"] = {
        "jump_teacher_mean_nll": float(np.mean(nll_jump)),
        "diffusion_only_mean_nll": float(np.mean(nll_diff)),
        "difference_mean": float(np.mean(nll_diff_arr)),
        "difference_paired_se": _paired_se(nll_diff_arr),
    }
    obs_large_rate = float(np.mean(
        np.abs(tr_test["nxt"][:, 0] - tr_test["cur"][:, 0]) > 0.5))
    results["H5_observed_large_increment_rate"] = obs_large_rate

    # Paired simulation from test initial states.
    sim_cfg = cfg["simulation"]
    dt_sim = float(sim_cfg["dt"])
    n_steps = int(round(float(sim_cfg["t_end"]) / dt_sim))
    rng_sim = np.random.default_rng(int(sim_cfg["seed"]))
    n_te = len(idx_te)
    dW = rng_sim.standard_normal((n_te, n_steps))
    uniforms = rng_sim.random((n_te, n_steps))
    jump_normals = rng_sim.standard_normal((n_te, n_steps))
    x0_te = data["x0"][idx_te]

    def _teacher_fns(model):
        return (lambda x: jump_predict(model, x)["f"],
                lambda x: jump_predict(model, x)["g"],
                lambda x: jump_predict(model, x)["lam"],
                lambda x: jump_predict(model, x)["mu"],
                lambda x: jump_predict(model, x)["sig"])

    def _sym_fns():
        return (lambda x: sym["f"].predict(x),
                lambda x: sym["g"].predict(x),
                lambda x: sym["lam"].predict(x),
                lambda x: sym["mu"].predict(x),
                lambda x: sym["sig"].predict(x))

    def _truth_fns():
        return (true_drift, true_g, true_lambda, true_mu, true_sigma_j)

    def _diff_fns(model):
        # Diffusion-only model simulated as a jump model with lambda = 0.
        return (lambda x: diffusion_predict(model, x)["f"],
                lambda x: diffusion_predict(model, x)["g"],
                lambda x: np.zeros((len(np.atleast_2d(x)), 1)),
                lambda x: np.zeros((len(np.atleast_2d(x)), 1)),
                lambda x: np.ones((len(np.atleast_2d(x)), 1)))

    sims = {}
    for label, fns in (("teacher", _teacher_fns(jump_model)),
                       ("symbolic", _sym_fns()),
                       ("truth", _truth_fns()),
                       ("diffusion_only", _diff_fns(diff_teacher["model"]))):
        sims[label] = simulate_jump(
            fns[0], fns[1], fns[2], fns[3], fns[4], x0_te, dt_sim, n_steps,
            dW, uniforms, jump_normals, float(sim_cfg["failure_radius"]),
            float(sim_cfg["excursion_radius"]))
    paired = sims["symbolic"]["states"][:, :, 0] - sims["teacher"]["states"][:, :, 0]
    ok = np.isfinite(paired)
    per_traj_rmse = np.sqrt(np.nansum(np.where(ok, paired ** 2, 0.0), axis=1)
                            / np.maximum(ok.sum(axis=1), 1))
    valid_traj = ok.any(axis=1)
    results["simulation"] = {
        label: {k: v for k, v in sim.items()
                if k in ("failure_rate", "excursion_rate", "jump_step_rate",
                         "large_step_rate", "clip_rate")}
        for label, sim in sims.items()}
    results["simulation"]["paired_rmse_symbolic_vs_teacher_median"] = float(
        np.median(per_traj_rmse[valid_traj])) if valid_traj.any() else float("nan")
    results["simulation"]["paired_rmse_symbolic_vs_teacher_mean"] = float(
        np.mean(per_traj_rmse[valid_traj])) if valid_traj.any() else float("nan")
    results["H5_simulated_large_step_rates"] = {
        label: sims[label]["large_step_rate"] for label in sims}
    results["H5_simulated_jump_step_rates"] = {
        label: sims[label]["jump_step_rate"] for label in sims}

    # Downstream H7: one-step mean-prediction MSE on test transitions.
    def _onestep_mse(f, lam, mu):
        pred_mean = tr_test["cur"] + f * dt_obs + mu * lam * dt_obs
        return ((tr_test["nxt"] - pred_mean) ** 2)[:, 0]
    mse_teacher = _onestep_mse(teacher_q["test"]["f"] if False else
                               jump_predict(jump_model, tr_test["cur"])["f"],
                               jump_predict(jump_model, tr_test["cur"])["lam"],
                               jump_predict(jump_model, tr_test["cur"])["mu"])
    sym_on_test = {k: sym[k].predict(tr_test["cur"]) for k in ("f", "lam", "mu")}
    mse_sym = _onestep_mse(sym_on_test["f"], sym_on_test["lam"], sym_on_test["mu"])
    mse_diff_arr = mse_sym - mse_teacher
    results["H7_one_step"] = {
        "teacher_mse": float(np.mean(mse_teacher)),
        "symbolic_mse": float(np.mean(mse_sym)),
        "difference_mean": float(np.mean(mse_diff_arr)),
        "difference_paired_se": _paired_se(mse_diff_arr),
    }

    # Teacher weights (small MLPs, JSON-serializable).
    weights = {name: net.state() for name, net in jump_model.items()}
    weights_diff = {name: net.state() for name, net in diff_teacher["model"].items()}
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(weights, indent=2))
    (OUT_DIR / "teacher_weights_diffusion_only.json").write_text(
        json.dumps(weights_diff, indent=2))

    # TeX: symbolic jump-diffusion in original coordinates.
    expanded = {}
    for k in ("f", "g", "lam", "mu", "sig"):
        expanded[k] = expand_to_original(sym[k])[0]
    tex = (
        "%% Day 8 symbolic jump-diffusion (Jia & Benson 2019 Neural Jump SDEs; "
        "distilled jump teacher; x = X)\n"
        + tex_equations([expanded["f"]], ("\\hat f",), ("x",)) + "\n"
        + tex_equations([expanded["g"]], ("\\hat g",), ("x",)) + "\n"
        + tex_equations([expanded["lam"]], ("\\hat \\lambda",), ("x",)) + "\n"
        + tex_equations([expanded["mu"]], ("\\hat \\mu_J",), ("x",)) + "\n"
        + tex_equations([expanded["sig"]], ("\\hat \\sigma_J",), ("x",)) + "\n"
        + "%% Jump term: dX = f dt + g dW + J dN(lambda), "
        "J ~ N(mu_J, sigma_J^2).\n"
    )
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    results["symbolic_expanded"] = {
        k: {str(exp): v for exp, v in expanded[k].items()} for k in expanded}
    results["symbolic_tex"] = tex
    results["elapsed_seconds"] = time.time() - t_start
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in results
                      if k not in ("symbolic_expanded", "symbolic_tex")},
                     indent=2)[:4000])
    return results


if __name__ == "__main__":
    main()
