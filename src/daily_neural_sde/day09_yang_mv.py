"""Day 9: Yang et al. (2024) Neural McKean-Vlasov Processes -> polynomial SINDy.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day09.json``):

1. Ground truth: an identifiable observable finite-particle McKean-Vlasov
   SDE in the paper's form (2): an Ito double-well drift plus a mean-field
   attraction term E[phi(x, y)] with phi(x, y) = -(x - y) (Figure 1b),
   dX_t = [(X_t - X_t^3) + 0.9 * (m_t - X_t)] dt + 0.6 dW_t,
   m_t = E[X_t], realized with 64 particles per ensemble and the empirical
   ensemble mean as m_t (the paper's empirical-measure architecture
   analogue). The paper's diffusion is a known constant sigma in its
   analysis (Section 2.1). Each sample is an ensemble, never a single
   particle: splits are ensemble-disjoint, because particles in one
   ensemble share m_t (splitting particles within an ensemble would
   leak the mean-field input across splits).
2. Teachers (all Gaussian transition likelihood on train transitions
   only): an MV teacher with input (x, m) and an Ito teacher with input
   x only (the paper-axis baseline: distributional dependence dropped).
   An MV EM pair at dt = 0.5 (m frozen at the step start, the
   mean-field-specific approximation structure) vs a
   transition-count-matched dt = 0.05 pair is the Day 1 discretization
   diagnostic (H6), regime-split by |m| with per-transition SNR
   quantified.
3. Distillation of the MV teacher into polynomial SINDy in (x, m):
   drift candidates include full-degree libraries and factorized
   libraries (x-polynomials plus m and x*m), including a
   factorized-origin library that genuinely vanishes at (x, m) = (0, 0)
   (Day 7 structural-constraint reflection, mean-0 standardization).
   Direct function queries on ensemble-disjoint splits,
   validation-only selection, test once. A collinearity gate at the
   maximum selectable degree (3) precedes fitting (Day 1 / Day 7), with
   the constant column excluded (Day 8 lesson: its VIF is spurious);
   H3c uses a pre-frozen factorized diagnostic fit. A fixed (x, m)
   grid is the Day 2 off-manifold diagnostic, reported only, never
   used for selection (Day 5).
4. Validation by paired ensemble simulation from test initial states
   with shared Brownian increments, each model computing its own
   ensemble mean from its own simulated particles at every step
   (self-consistent mean field), retaining failed-particle mass and
   counting |X| excursions (> 8) and ensemble-mean excursions
   (|m_t| > 5) as domain measures separate from failure (Day 4).
   Paper axis: H5 compares the MV teacher against the Ito teacher on
   absolute test-transition NLL levels and on phase persistence
   (mean |m_T| on test ensembles with |m0| >= 0.8), with the Ito
   separation judged by a relative ratio (Day 8 lesson).

Run from the repository root:
``python -m src.daily_neural_sde.day09_yang_mv``

Outputs land in ``outputs/daily_neural_sde/day09/``.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, evaluate_powers, expand_to_original,
                     monomial_powers, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day09.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day09"

DT_FINE = 0.01
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
THETA = 0.9
SIGMA = 0.6


# ---------------------------------------------------------------- truth

def true_drift(xm: np.ndarray) -> np.ndarray:
    xm = np.atleast_2d(np.asarray(xm, dtype=np.float64))
    x, m = xm[:, [0]], xm[:, [1]]
    return (x - x ** 3) + THETA * (m - x)


def true_g(xm: np.ndarray) -> np.ndarray:
    xm = np.atleast_2d(np.asarray(xm, dtype=np.float64))
    return np.full((len(xm), 1), SIGMA)


def generate_data(cfg: dict) -> dict[str, Any]:
    """Simulate interacting-particle ensembles at dt_fine, observe every dt_obs."""
    rng = np.random.default_rng(int(cfg["seed"]))
    n_ens = int(cfg["data"]["n_ensembles"])
    n_p = int(cfg["data"]["n_particles"])
    t_end = float(cfg["data"]["t_end"])
    dt_obs = float(cfg["data"]["dt_obs"])
    n_fine = int(round(t_end / DT_FINE))
    obs_every = int(round(dt_obs / DT_FINE))
    n_obs = int(round(t_end / dt_obs)) + 1
    m0 = rng.uniform(-1.2, 1.2, size=n_ens)
    cur = m0[:, None] + 0.6 * rng.standard_normal((n_ens, n_p))
    buf = np.zeros((n_ens, n_obs, n_p))
    buf[:, 0, :] = cur
    obs_idx = 1
    for k in range(1, n_fine + 1):
        m = cur.mean(axis=1, keepdims=True)
        drift = (cur - cur ** 3) + THETA * (m - cur)
        cur = cur + drift * DT_FINE + SIGMA * np.sqrt(DT_FINE) * rng.standard_normal(cur.shape)
        if k % obs_every == 0:
            buf[:, obs_idx, :] = cur
            obs_idx += 1
    assert obs_idx == n_obs
    return {"obs": buf, "x0": buf[:, 0, :].copy(), "m0": m0}


def _pairs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_flat, m_flat, obs) with (ensemble, time, particle) order."""
    n_ens, n_obs, n_p = obs.shape
    means = obs.mean(axis=2)  # (E, T)
    x_flat = obs.reshape(-1)
    m_flat = np.repeat(means.reshape(-1), n_p)
    return x_flat, m_flat, obs


def build_transitions(obs: np.ndarray, dt_obs: float) -> dict[str, np.ndarray]:
    sub = obs[:, :-1, :]
    x_flat, m_flat, _ = _pairs(sub)
    nxt = obs[:, 1:, :].reshape(-1)
    dt = np.full(len(x_flat), dt_obs)
    return {"cur": np.stack([x_flat, m_flat], axis=1),
            "nxt": nxt[:, None], "dt": dt}


def build_queries(obs: np.ndarray) -> np.ndarray:
    x_flat, m_flat, _ = _pairs(obs)
    return np.stack([x_flat, m_flat], axis=1)


def subsample_transitions(tr: dict, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tr["cur"]), size=n, replace=False)
    return {k: v[idx] for k, v in tr.items()}


def coarse_transitions(obs: np.ndarray, dt_obs: float, step: int) -> dict:
    """Transitions joining every ``step``-th observation (m frozen at step start)."""
    return build_transitions(obs[:, ::step, :], dt_obs * step)


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


def _train_gaussian(tr: dict, in_dim: int, epochs: int, batch: int,
                    lr: float, seed: int) -> dict:
    """Gaussian EM NLL for a teacher with input dimension ``in_dim``."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    f_net = MLP(in_dim, 1, (32, 32))
    g_net = MLP(in_dim, 1, (16,))
    for net in (f_net, g_net):
        net.fit_standardizer(tr["cur"])
    opt = torch.optim.Adam(list(itertools.chain.from_iterable(
        net.net.parameters() for net in (f_net, g_net))), lr=lr)
    n = len(tr["cur"])
    last = 0.0
    for _ in range(epochs):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            cur = torch.tensor(tr["cur"][idx], dtype=torch.float32)
            nxt = torch.tensor(tr["nxt"][idx], dtype=torch.float32)
            dt = torch.tensor(tr["dt"][idx], dtype=torch.float32)[:, None]
            f = f_net.forward_raw(tr["cur"][idx])
            g = torch.nn.functional.softplus(g_net.forward_raw(tr["cur"][idx])) + 0.01
            var = g ** 2 * dt + 1e-8
            resid = nxt - cur[:, [0]] - f * dt
            loss = torch.mean(LOG_SQRT_2PI + 0.5 * torch.log(var)
                              + 0.5 * resid ** 2 / var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        last = tot / max(cnt, 1)
    return {"model": {"f": f_net, "g": g_net}, "loss": last}


def mv_predict(model: dict, xm: np.ndarray) -> dict[str, np.ndarray]:
    xm = np.atleast_2d(np.asarray(xm, dtype=np.float64))
    return {"f": model["f"].predict(xm),
            "g": np.logaddexp(0.0, model["g"].predict(xm)) + 0.01}


def ito_predict(model: dict, xm: np.ndarray) -> dict[str, np.ndarray]:
    """Ito teacher: same nets, but only the state column is fed in."""
    xm = np.atleast_2d(np.asarray(xm, dtype=np.float64))
    return {"f": model["f"].predict(xm[:, [0]]),
            "g": np.logaddexp(0.0, model["g"].predict(xm[:, [0]])) + 0.01}


def _model_state(model: dict) -> dict:
    return {k: net.state() for k, net in model.items()}


def gaussian_nll_per_transition(predict_fn, model: dict, tr: dict) -> np.ndarray:
    pred = predict_fn(model, tr["cur"])
    dt = tr["dt"][:, None]
    var = pred["g"] ** 2 * dt + 1e-8
    resid = tr["nxt"] - tr["cur"][:, [0]] - pred["f"] * dt
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


def fit_field_powers(x: np.ndarray, y: np.ndarray, powers: np.ndarray,
                     threshold: float, ridge: float,
                     mean0: bool = False) -> PolynomialField:
    """Polynomial STLSQ on an explicit monomial-powers library.

    ``mean0=True`` standardizes with mean 0 (scale only), so dropping the
    constant power genuinely pins the expanded polynomial at the origin
    of the original coordinates (Day 7 / Day 8 lesson).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = np.zeros(x.shape[1]) if mean0 else x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
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
    metadata = {"threshold": threshold, "ridge": ridge, "mean0": mean0,
                "library_size": int(len(powers)),
                "active_terms": int(np.count_nonzero(coefficients)),
                "optimizer": "numpy STLSQ (RMS-normalized, unbias refit)",
                "feature_rms": feature_rms.tolist(),
                "target_rms": target_rms.tolist()}
    return PolynomialField(powers, coefficients, mean, scale, metadata)


FACT_FREE = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)])
FACT_ORIGIN = np.array([(1, 0), (2, 0), (3, 0), (0, 1), (1, 1)])


def fit_field(x: np.ndarray, y: np.ndarray, x_val: np.ndarray,
              y_val: np.ndarray, degrees, thresholds, ridge: float,
              kinds=("full", "fact_free", "fact_origin")) -> tuple:
    """Validation-selected polynomial field over full/factorized libraries."""
    from .common import fit_polynomial_field
    best = None
    for kind in kinds:
        for thr in thresholds:
            if kind == "full":
                for degree in degrees:
                    field = fit_polynomial_field(x, y, degree, thr, ridge)
                    val = nrmse(y_val, field.predict(x_val))
                    meta = {"kind": kind, "degree": degree, "threshold": thr}
                    if best is None or val < best[1]:
                        best = (field, val, meta)
            elif kind == "fact_free":
                field = fit_field_powers(x, y, FACT_FREE, thr, ridge)
                val = nrmse(y_val, field.predict(x_val))
                meta = {"kind": kind, "degree": None, "threshold": thr}
                if best is None or val < best[1]:
                    best = (field, val, meta)
            elif kind == "fact_origin":
                field = fit_field_powers(x, y, FACT_ORIGIN, thr, ridge, mean0=True)
                val = nrmse(y_val, field.predict(x_val))
                meta = {"kind": kind, "degree": None, "threshold": thr}
                if best is None or val < best[1]:
                    best = (field, val, meta)
    assert best is not None
    return best


def vif_scores(x: np.ndarray) -> np.ndarray:
    """Variance inflation factors for columns of x (intercept included in each regression)."""
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

def simulate_mv(drift_fn, diff_fn, x0: np.ndarray, dt: float, n_steps: int,
                dW: np.ndarray, failure_radius: float,
                excursion_radius: float, mean_excursion_radius: float) -> dict[str, Any]:
    """Paired interacting-particle Euler simulation with self-consistent mean.

    Each model computes its own ensemble mean from its own simulated
    (non-failed) particles at every step. Failures latch per particle
    (non-finite or |X| > failure_radius); failed particles keep their
    mass in the counts and are NaN after failure.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    n_ens, n_p = x.shape
    if dW.shape != (n_ens, n_p, n_steps):
        raise ValueError("increments shape mismatch")
    failed = np.zeros((n_ens, n_p), dtype=bool)
    states = np.zeros((n_ens, n_steps + 1, n_p))
    states[:, 0, :] = x
    ever_exc = np.zeros((n_ens, n_p), dtype=bool)
    ever_mean_exc = np.zeros(n_ens, dtype=bool)
    sqrt_dt = np.sqrt(dt)
    for k in range(1, n_steps + 1):
        failed_before = failed.copy()
        counts = (~failed).sum(axis=1)
        sums = np.where(failed, 0.0, x).sum(axis=1)
        m = sums / np.maximum(counts, 1)
        xm = np.stack([x.reshape(-1), np.repeat(m, n_p)], axis=1)
        f = drift_fn(xm)[:, 0].reshape(n_ens, n_p)
        g = diff_fn(xm)[:, 0].reshape(n_ens, n_p)
        new = x + f * dt + g * sqrt_dt * dW[:, :, k - 1]
        new = np.where(failed_before, np.nan, new)
        ever_exc = ever_exc | ((np.abs(new) > excursion_radius) & ~failed_before)
        bad = (~failed_before) & (~np.isfinite(new) | (np.abs(new) > failure_radius))
        failed = failed_before | bad
        x = np.where(failed, np.nan, new)
        counts2 = (~failed).sum(axis=1)
        sums2 = np.where(failed, 0.0, x).sum(axis=1)
        m2 = sums2 / np.maximum(counts2, 1)
        ever_mean_exc = ever_mean_exc | ((counts2 > 0) & (np.abs(m2) > mean_excursion_radius))
        states[:, k, :] = x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_means = np.nanmean(states[:, -1, :], axis=1)
    return {"states": states, "failed": failed,
            "failure_rate": float(np.mean(failed)),
            "excursion_rate": float(np.mean(ever_exc)),
            "mean_excursion_rate": float(np.mean(ever_mean_exc)),
            "final_means": final_means}


# ---------------------------------------------------------------- main

def _split_indices(n: int, cfg: dict, rng: np.random.Generator):
    perm = rng.permutation(n)
    n_train = int(cfg["split"]["train"])
    n_val = int(cfg["split"]["validation"])
    return (perm[:n_train], perm[n_train:n_train + n_val],
            perm[n_train + n_val:n_train + n_val + int(cfg["split"]["test"])])


def main() -> dict[str, Any]:
    t_start = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    data = generate_data(cfg)
    obs = data["obs"]
    dt_obs = float(cfg["data"]["dt_obs"])
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    idx_tr, idx_va, idx_te = _split_indices(len(obs), cfg, rng)
    splits = {"train": obs[idx_tr], "validation": obs[idx_va], "test": obs[idx_te]}
    m0_test = data["m0"][idx_te]

    tr = build_transitions(splits["train"], dt_obs)
    tr_va = build_transitions(splits["validation"], dt_obs)
    tr_te = build_transitions(splits["test"], dt_obs)
    q_tr = build_queries(splits["train"])
    q_va = build_queries(splits["validation"])
    q_te = build_queries(splits["test"])

    tcfg = cfg["teacher"]
    mv = _train_gaussian(tr, 2, int(tcfg["em_main"]["epochs"]),
                         int(tcfg["em_main"]["batch"]), float(tcfg["em_main"]["lr"]), 0)
    ito = _train_gaussian({**tr, "cur": tr["cur"][:, [0]]}, 1,
                          int(tcfg["em_main"]["epochs"]), int(tcfg["em_main"]["batch"]),
                          float(tcfg["em_main"]["lr"]), 0)
    coarse_tr = coarse_transitions(splits["train"], dt_obs, 10)
    n_coarse = len(coarse_tr["cur"])
    fine_sub = subsample_transitions(tr, n_coarse, 7)
    mv_coarse = _train_gaussian(coarse_tr, 2, int(tcfg["em_coarse"]["epochs"]),
                                int(tcfg["em_coarse"]["batch"]), float(tcfg["em_coarse"]["lr"]), 0)
    mv_fine = _train_gaussian(fine_sub, 2, int(tcfg["em_fine_sub"]["epochs"]),
                              int(tcfg["em_fine_sub"]["batch"]), float(tcfg["em_fine_sub"]["lr"]), 0)

    def field_err(predict_fn, model, target_fn):
        pred = predict_fn(model, q_te)
        return {"f_nrmse": nrmse(target_fn(q_te), pred["f"]),
                "f_rms": float(np.sqrt(np.mean((pred["f"] - target_fn(q_te)) ** 2))),
                "g_nrmse": nrmse(true_g(q_te), pred["g"]),
                "g_rms": float(np.sqrt(np.mean((pred["g"] - true_g(q_te)) ** 2)))}

    teacher_err = field_err(mv_predict, mv["model"], true_drift)
    ito_err = field_err(ito_predict, ito["model"], true_drift)
    coarse_err = field_err(mv_predict, mv_coarse["model"], true_drift)
    fine_err = field_err(mv_predict, mv_fine["model"], true_drift)

    # Regime splits: teacher drift error vs truth.
    def regime_err(predict_fn, model, mask):
        pred = predict_fn(model, q_te[mask])
        truth = true_drift(q_te[mask])
        return {"nrmse": nrmse(truth, pred["f"]),
                "rms": float(np.sqrt(np.mean((pred["f"] - truth) ** 2))), "n": int(mask.sum())}

    mask_m_small = np.abs(q_te[:, 1]) < 0.3
    mask_m_large = np.abs(q_te[:, 1]) > 0.8
    mask_x_lin = np.abs(q_te[:, 0]) < 0.5
    mask_x_cub = np.abs(q_te[:, 0]) > 1.5
    h6 = {
        "overall": {"coarse": coarse_err["f_nrmse"], "fine": fine_err["f_nrmse"],
                    "degradation": coarse_err["f_nrmse"] - fine_err["f_nrmse"]},
        "m_small": {"coarse": regime_err(mv_predict, mv_coarse["model"], mask_m_small)["nrmse"],
                    "fine": regime_err(mv_predict, mv_fine["model"], mask_m_small)["nrmse"]},
        "m_large": {"coarse": regime_err(mv_predict, mv_coarse["model"], mask_m_large)["nrmse"],
                    "fine": regime_err(mv_predict, mv_fine["model"], mask_m_large)["nrmse"]},
        "m_small_n": int(mask_m_small.sum()), "m_large_n": int(mask_m_large.sum()),
    }
    h6["m_small"]["degradation"] = h6["m_small"]["coarse"] - h6["m_small"]["fine"]
    h6["m_large"]["degradation"] = h6["m_large"]["coarse"] - h6["m_large"]["fine"]
    h6["regime_gap"] = h6["m_large"]["degradation"] - h6["m_small"]["degradation"]
    h6["g_coarse_nrmse"] = coarse_err["g_nrmse"]
    h6["g_fine_nrmse"] = fine_err["g_nrmse"]
    h6["sigma_coarse_nrmse"] = nrmse(true_g(q_te) ** 2, mv_predict(mv_coarse["model"], q_te)["g"] ** 2)
    h6["sigma_fine_nrmse"] = nrmse(true_g(q_te) ** 2, mv_predict(mv_fine["model"], q_te)["g"] ** 2)
    # Per-transition SNR by |m| regime, using truth f and g.
    snr = {}
    for name, mask in (("m_small", mask_m_small), ("m_large", mask_m_large)):
        f_abs = np.abs(true_drift(q_te[mask]))[:, 0]
        snr[name] = {"dt_0.05": float(np.mean(f_abs * 0.05 / (SIGMA * np.sqrt(0.05)))),
                     "dt_0.5": float(np.mean(f_abs * 0.5 / (SIGMA * np.sqrt(0.5))))}
    h6["snr"] = snr
    lin_nonlin = {
        "linear_regime": regime_err(mv_predict, mv["model"], mask_x_lin),
        "cubic_regime": regime_err(mv_predict, mv["model"], mask_x_cub),
    }

    # Collinearity gate at maximum selectable degree (3), constant excluded.
    train_mean = q_tr.mean(axis=0)
    train_scale = q_tr.std(axis=0)
    z_tr = (q_tr - train_mean) / train_scale
    powers3 = monomial_powers(2, 3)
    design3 = evaluate_powers(z_tr, powers3)
    keep = powers3.sum(axis=1) > 0
    vif = vif_scores(design3[:, keep])
    vif_labels = [tuple(p) for p in powers3[keep]]
    teacher_f00 = float(mv_predict(mv["model"], np.array([[0.0, 0.0]]))["f"][0, 0])

    # Distillation: validation-only selection, test once.
    y_tr_f = mv_predict(mv["model"], q_tr)["f"]
    y_va_f = mv_predict(mv["model"], q_va)["f"]
    y_tr_g = mv_predict(mv["model"], q_tr)["g"]
    y_va_g = mv_predict(mv["model"], q_va)["g"]
    dcfg = cfg["distillation"]
    field_f, val_f, meta_f = fit_field(q_tr, y_tr_f, q_va, y_va_f,
                                       dcfg["degrees"]["drift_full"], dcfg["thresholds"],
                                       dcfg["ridge"])
    field_g, val_g, meta_g = fit_field(q_tr, y_tr_g, q_va, y_va_g,
                                       dcfg["degrees"]["diffusion_full"], dcfg["thresholds"],
                                       dcfg["ridge"], kinds=("full",))
    pred_te_f = field_f.predict(q_te)
    pred_te_g = field_g.predict(q_te)
    teach_te_f = mv_predict(mv["model"], q_te)["f"]
    teach_te_g = mv_predict(mv["model"], q_te)["g"]
    dist = {
        "drift": {"selection": meta_f, "validation_nrmse": val_f,
                  "test_vs_teacher": nrmse(teach_te_f, pred_te_f),
                  "test_vs_truth": nrmse(true_drift(q_te), pred_te_f)},
        "g": {"selection": meta_g, "validation_nrmse": val_g,
              "test_vs_teacher": nrmse(teach_te_g, pred_te_g),
              "test_vs_truth": nrmse(true_g(q_te), pred_te_g),
              "sigma_test_vs_teacher": nrmse(teach_te_g ** 2, pred_te_g ** 2)},
    }
    # Oracle ceiling: same procedure fitted to truth field values.
    oracle_f, _, _ = fit_field(q_tr, true_drift(q_tr), q_va, true_drift(q_va),
                               dcfg["degrees"]["drift_full"], dcfg["thresholds"], dcfg["ridge"])
    oracle_g, _, _ = fit_field(q_tr, true_g(q_tr), q_va, true_g(q_va),
                               dcfg["degrees"]["diffusion_full"], dcfg["thresholds"],
                               dcfg["ridge"], kinds=("full",))
    dist["oracle_drift_test_vs_truth"] = nrmse(true_drift(q_te), oracle_f.predict(q_te))
    dist["oracle_g_test_vs_truth"] = nrmse(true_g(q_te), oracle_g.predict(q_te))

    # H3c diagnostic: pre-frozen factorized-free fit at threshold 0.05.
    diag = fit_field_powers(q_tr, y_tr_f, FACT_FREE, 0.05, dcfg["ridge"])
    diag_exp = expand_to_original(diag)[0]
    diag_test = nrmse(teach_te_f, diag.predict(q_te))
    h3c = {"expanded": {str(k): v for k, v in diag_exp.items()},
           "x_linear_error": abs(diag_exp.get((1, 0), 0.0) - 0.1),
           "m_error": abs(diag_exp.get((0, 1), 0.0) - 0.9),
           "test_vs_teacher": diag_test}

    # Grid diagnostic (report only).
    xs = np.arange(-3.0, 3.01, 0.5)
    ms = np.arange(-1.5, 1.51, 0.5)
    grid = np.array([(x, m) for m in ms for x in xs], dtype=np.float64)
    grid_f_teach = mv_predict(mv["model"], grid)["f"]
    grid_f_sym = field_f.predict(grid)
    grid_f_true = true_drift(grid)
    grid_g_teach = mv_predict(mv["model"], grid)["g"]
    grid_g_sym = field_g.predict(grid)
    grid_g_true = true_g(grid)
    in_range = np.abs(grid[:, 1]) <= 1.1
    grid_diag = {
        "n_points": len(grid),
        "f_sym_vs_teacher": nrmse(grid_f_teach, grid_f_sym),
        "f_sym_vs_teacher_rms": float(np.sqrt(np.mean((grid_f_sym - grid_f_teach) ** 2))),
        "f_teacher_vs_truth": nrmse(grid_f_true, grid_f_teach),
        "f_sym_vs_truth": nrmse(grid_f_true, grid_f_sym),
        "g_sym_vs_teacher": nrmse(grid_g_teach, grid_g_sym),
        "g_teacher_vs_truth": nrmse(grid_g_true, grid_g_teach),
        "g_sym_vs_truth": nrmse(grid_g_true, grid_g_sym),
        "f_sym_vs_teacher_in_range_m": nrmse(grid_f_teach[in_range], grid_f_sym[in_range]),
        "f_sym_vs_teacher_out_range_m": nrmse(grid_f_teach[~in_range], grid_f_sym[~in_range]),
        "g_teacher_m_variation_at_x0": float(np.max(grid_g_teach[np.abs(grid[:, 0]) < 1e-9])
                                              - np.min(grid_g_teach[np.abs(grid[:, 0]) < 1e-9])),
    }

    # Paired ensemble simulation.
    sim_cfg = cfg["simulation"]
    n_steps = int(round(float(sim_cfg["t_end"]) / float(sim_cfg["dt"])))
    sim_rng = np.random.default_rng(int(sim_cfg["seed"]))
    dW = sim_rng.standard_normal((len(splits["test"]), splits["test"].shape[2], n_steps))
    x0_test = splits["test"][:, 0, :]
    drift_teacher = lambda xm: mv_predict(mv["model"], xm)["f"]
    diff_teacher = lambda xm: mv_predict(mv["model"], xm)["g"]
    drift_sym = lambda xm: field_f.predict(xm)
    diff_sym = lambda xm: field_g.predict(xm)
    drift_truth = lambda xm: true_drift(xm)
    diff_truth = lambda xm: true_g(xm)
    drift_ito = lambda xm: ito_predict(ito["model"], xm)["f"]
    diff_ito = lambda xm: ito_predict(ito["model"], xm)["g"]
    sims = {}
    for name, df, gf in (("teacher", drift_teacher, diff_teacher),
                         ("symbolic", drift_sym, diff_sym),
                         ("truth", drift_truth, diff_truth),
                         ("ito", drift_ito, diff_ito)):
        sims[name] = simulate_mv(df, gf, x0_test, float(sim_cfg["dt"]), n_steps, dW,
                                 float(sim_cfg["failure_radius"]),
                                 float(sim_cfg["excursion_radius"]),
                                 float(sim_cfg["mean_excursion_radius"]))
    pair = sims["symbolic"]["states"] - sims["teacher"]["states"]
    finite_pair = np.isfinite(pair)
    paired_median = float(np.median(np.abs(pair[finite_pair]))) if finite_pair.any() else float("nan")
    sim_summary = {name: {"failure_rate": s["failure_rate"],
                           "excursion_rate": s["excursion_rate"],
                           "mean_excursion_rate": s["mean_excursion_rate"],
                           "final_mean_of_abs_means": float(np.nanmean(np.abs(s["final_means"]))) }
                    for name, s in sims.items()}
    # Phase persistence subset: test ensembles with |m0| >= 0.8.
    phase_mask = np.abs(m0_test) >= 0.8
    phase = {"subset_n": int(phase_mask.sum())}
    for name, s in sims.items():
        vals = np.abs(s["final_means"][phase_mask]) if phase_mask.any() else np.array([])
        phase[name + "_mean_abs_mT"] = float(np.mean(vals)) if len(vals) else None
    if phase_mask.any() and phase["truth_mean_abs_mT"]:
        phase["ito_to_truth_ratio"] = phase["ito_mean_abs_mT"] / phase["truth_mean_abs_mT"]
        phase["teacher_abs_diff_vs_truth"] = abs(phase["teacher_mean_abs_mT"] - phase["truth_mean_abs_mT"])
        phase["symbolic_abs_diff_vs_truth"] = abs(phase["symbolic_mean_abs_mT"] - phase["truth_mean_abs_mT"])
    else:
        phase["ito_to_truth_ratio"] = None
        phase["teacher_abs_diff_vs_truth"] = None
        phase["symbolic_abs_diff_vs_truth"] = None

    # Downstream: one-step MSE and NLL.
    mean_pred_teach = tr_te["cur"][:, [0]] + mv_predict(mv["model"], tr_te["cur"])["f"] * tr_te["dt"][:, None]
    mean_pred_sym = tr_te["cur"][:, [0]] + field_f.predict(tr_te["cur"]) * tr_te["dt"][:, None]
    mse_teach = float(np.mean((tr_te["nxt"] - mean_pred_teach) ** 2))
    mse_sym = float(np.mean((tr_te["nxt"] - mean_pred_sym) ** 2))
    per_pair = (tr_te["nxt"] - mean_pred_sym) ** 2 - (tr_te["nxt"] - mean_pred_teach) ** 2
    mse_diff = mse_sym - mse_teach
    mse_se = float(np.std(per_pair[:, 0], ddof=1) / np.sqrt(len(per_pair)))
    nll_mv = gaussian_nll_per_transition(mv_predict, mv["model"], tr_te)
    nll_ito = gaussian_nll_per_transition(ito_predict, ito["model"], tr_te)
    nll_diff = nll_ito - nll_mv
    nll = {"mv_mean": float(np.mean(nll_mv)), "ito_mean": float(np.mean(nll_ito)),
           "diff_mean": float(np.mean(nll_diff)),
           "paired_se": float(np.std(nll_diff, ddof=1) / np.sqrt(len(nll_diff)))}

    hyp = cfg["hypotheses"]
    verdicts = {
        "H1": bool(dist["drift"]["test_vs_teacher"] <= hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"]),
        "H2": bool(dist["g"]["test_vs_teacher"] <= hyp["H2_diffusion_distillation"]["symbolic_diffusion_test_nrmse_vs_teacher_max"]
                    and dist["g"]["sigma_test_vs_teacher"] <= hyp["H2_diffusion_distillation"]["symbolic_sigma_test_nrmse_vs_teacher_max"]),
        "H3": bool(dist["drift"]["test_vs_truth"] <= hyp["H3_truth_recovery"]["field_symbolic_drift_test_nrmse_vs_truth_max"]
                    and dist["g"]["test_vs_truth"] <= hyp["H3_truth_recovery"]["field_symbolic_g_test_nrmse_vs_truth_max"]),
        "H3c": bool(h3c["x_linear_error"] <= hyp["H3c_mean_field_coefficients"]["x_linear_coefficient_max_abs_error"]
                     and h3c["m_error"] <= hyp["H3c_mean_field_coefficients"]["m_coefficient_max_abs_error"]),
        "H4": bool(sims["symbolic"]["failure_rate"] <= hyp["H4_simulation"]["symbolic_failure_rate_max"]
                    and abs(sims["symbolic"]["failure_rate"] - sims["teacher"]["failure_rate"]) <= hyp["H4_simulation"]["symbolic_failure_rate_minus_teacher_max"]
                    and paired_median <= hyp["H4_simulation"]["median_paired_rmse_vs_teacher_max"]
                    and abs(sims["symbolic"]["excursion_rate"] - sims["teacher"]["excursion_rate"]) <= hyp["H4_simulation"]["domain_excursion_symbolic_minus_teacher_max"]
                    and sims["symbolic"]["mean_excursion_rate"] <= hyp["H4_simulation"]["mean_excursion_symbolic_max"]),
        "H5": bool(nll["diff_mean"] > hyp["H5_paper_axis_meanfield_vs_ito"]["nll_clause"]["ito_minus_mv_min"]
                    and nll["diff_mean"] > 2 * nll["paired_se"]
                    and phase["subset_n"] >= hyp["H5_paper_axis_meanfield_vs_ito"]["phase_clause"]["subset_min_ensembles"]
                    and phase["teacher_abs_diff_vs_truth"] is not None
                    and phase["teacher_abs_diff_vs_truth"] <= hyp["H5_paper_axis_meanfield_vs_ito"]["phase_clause"]["symbolic_and_teacher_abs_diff_vs_truth_max"]
                    and phase["symbolic_abs_diff_vs_truth"] <= hyp["H5_paper_axis_meanfield_vs_ito"]["phase_clause"]["symbolic_and_teacher_abs_diff_vs_truth_max"]
                    and phase["ito_to_truth_ratio"] <= hyp["H5_paper_axis_meanfield_vs_ito"]["phase_clause"]["ito_to_truth_ratio_max"]),
        "H6": bool(h6["overall"]["degradation"] >= hyp["H6_discretization_bias_em_coarsening"]["drift_nrmse_degradation_min"]
                    and h6["regime_gap"] >= hyp["H6_discretization_bias_em_coarsening"]["regime_gap_min"]),
        "H7": bool(abs(mse_diff) <= hyp["H7_downstream_one_step"]["mse_abs_diff_max"]),
    }
    verdicts["H7_statistically_established"] = bool(mse_se < hyp["H7_downstream_one_step"]["paired_se_indeterminate_threshold"])
    vif_max = float(np.max(vif))
    verdicts["H3c_vif_gate"] = "collinearity-limited (indeterminate)" if vif_max > 10 else "pass"

    results = {
        "config": "configs/daily_neural_sde_day09.json",
        "teacher_losses": {"mv": mv["loss"], "ito": ito["loss"],
                           "mv_coarse": mv_coarse["loss"], "mv_fine_sub": mv_fine["loss"]},
        "teacher_vs_truth_test": teacher_err,
        "ito_vs_truth_test": ito_err,
        "coarse_vs_truth_test": coarse_err,
        "fine_sub_vs_truth_test": fine_err,
        "h6_discretization": h6,
        "linear_vs_nonlinear_teacher_drift_error": lin_nonlin,
        "vif_degree3_full": {str(k): float(v) for k, v in zip(vif_labels, vif)},
        "vif_max": vif_max,
        "teacher_f_at_origin": teacher_f00,
        "distillation": dist,
        "h3c_diagnostic": h3c,
        "grid_diagnostic": grid_diag,
        "simulation": sim_summary,
        "simulation_paired_median_rmse_symbolic_vs_teacher": paired_median,
        "phase_persistence": phase,
        "one_step_mse": {"teacher": mse_teach, "symbolic": mse_sym,
                         "diff": mse_diff, "paired_se": mse_se},
        "nll_test_transitions": nll,
        "observed_ranges": {
            "train_query_x": [float(np.min(q_tr[:, 0])), float(np.max(q_tr[:, 0]))],
            "train_query_m": [float(np.min(q_tr[:, 1])), float(np.max(q_tr[:, 1]))],
            "train_query_m_std": float(np.std(q_tr[:, 1])),
            "test_query_m_std": float(np.std(q_te[:, 1])),
            "test_n_transitions": int(len(tr_te["cur"])),
            "train_n_transitions": int(len(tr["cur"])),
            "coarse_n_transitions": int(n_coarse),
        },
        "verdicts": verdicts,
        "runtime_seconds": time.time() - t_start,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    field_f.save(out / "symbolic_f.json")
    field_g.save(out / "symbolic_g.json")
    (out / "teacher_weights.json").write_text(json.dumps(_model_state(mv["model"]), indent=2))
    (out / "teacher_weights_ito.json").write_text(json.dumps(_model_state(ito["model"]), indent=2))
    expanded_f = expand_to_original(field_f)
    expanded_g = expand_to_original(field_g)
    tex = ("%% Day 9 symbolic McKean-Vlasov SDE (Yang et al. 2024 Neural "
           "McKean-Vlasov Processes; distilled MV teacher; x = X, m = E[X])\n"
           + tex_equations(expanded_f, ("\\hat f",), ("x", "m")) + "\n"
           + tex_equations(expanded_g, ("\\hat g",), ("x", "m")) + "\n"
           + "%% Mean-field SDE: dX = f(X, m) dt + g dW, m = E[X] "
             "(self-consistent ensemble mean).\n")
    (out / "symbolic_sde.tex").write_text(tex)
    print(json.dumps({"verdicts": verdicts, "distillation": dist,
                      "teacher_vs_truth": teacher_err, "phase": phase,
                      "simulation": sim_summary,
                      "paired_median": paired_median}, indent=2))
    return results


if __name__ == "__main__":
    main()
