"""Day 6: Eggen & Midtfjord (2023) Delay-SDE-Net -> SINDy distillation.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day06.json``):

1. Take the paper's Section 5.1.1 SDDE as the ground truth:
   dX(t) = f(t, Pi(X_t)) dt + g(t, Pi(eta)) dW(t), X(t) = eta(t) on
   [-4, 0], Pi(X_t) = (X(t-4), X(t-3), X(t-2), X(t-1)) in R^8
   (dim-major lag order, declared in the config). The drift is a sum
   of two tanh units per component with the Eq. (24) weights; the
   diffusion is diagonal with levels g^1 ~ 2.00 (t trend only) and
   g^2 ~ 0.091 (via X^2(-1) only), because the paper's Eq. (8) makes
   diffusion depend on the initial path only. Activation steepnesses
   alpha = lambda = 1 are declared (the paper gives no values).
   Trajectories are simulated by Euler-Maruyama at dt_fine = 0.01
   and observed every 0.05 on [0, 8]; every trajectory has its own
   random history eta(t) = [sin(z1 t), cos(z2 t)] (Day 5 reflection:
   multiple histories instead of a single initial condition).
2. Fit two-stage (paper-method) teachers: a delay teacher (drift
   inputs [t, lags], diffusion inputs [t, initial lags]) and a
   Markovian SDE-net teacher (inputs [t, current state]), plus EM
   likelihood teachers at dt_obs = 0.05 and at coarsened
   dt_obs = 0.5 (transition-count-matched, Day 1 discretization
   diagnostic). The distilled teacher is the delay two-stage
   teacher, fixed a priori because the paper's method is the
   Delay-SDE-net (its epistemic stage is cut, declared).
3. Distill the delay teacher into polynomial SINDy models in the
   augmented delay state (current + 4 lags x 2 dims) by direct
   function queries on trajectory-disjoint splits (validation-only
   selection, test once). A collinearity diagnostic on train query
   states precedes fitting (Day 1 reflection); a shuffled-lag
   diagnostic is the SDDE analogue of the Day 2 off-manifold grid
   (reporting only, never selection).
4. Validate by paired SDDE simulation from test history segments
   with shared Brownian increments, retaining failed-trajectory
   mass and counting radius excursions (> 50) as a domain-validity
   measure separate from failure (Day 4 reflection). Paper-axis
   proxy (H7): one-step-ahead X^1 MSE, symbolic vs teacher, with
   the paired standard error reported (Day 4 reflection).

Run from the repository root:
``python -m src.daily_neural_sde.day06_eggen_delay``
Outputs land in ``outputs/daily_neural_sde/day06/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, expand_to_original,
                     fit_polynomial_field, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day06.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day06"

DT_FINE = 0.01
LAGS = (-4.0, -3.0, -2.0, -1.0)

# Paper Eq. (24) weights times 1e-2, rows: w_1^1, w_2^1, w_1^2, w_2^2,
# w^1 (diffusion dim 1), w^2 (diffusion dim 2). Input order for every
# row: [t, X^1(t-4), X^1(t-3), X^1(t-2), X^1(t-1),
#           X^2(t-4), X^2(t-3), X^2(t-2), X^2(t-1)].
W_TRUTH = np.array([
    [0, 3, 2, 2, 5, -3, 1, -3, -1],
    [0, 1, 0, -0.5, 0, -1, 0, -0.5, 0],
    [0, 0, 2, 0, -3, 0, 1, 0, 0],
    [0, 0, 1, 0, -0.5, 0, 0, 0, -0.5],
    [-5 / 365, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.float64) * 1e-2


# ---------------------------------------------------------------- truth

def true_drift_from_u(u: np.ndarray) -> np.ndarray:
    """Truth drift for input rows [t, lags(8)] -> (n, 2)."""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    f1 = 5.0 * np.tanh(u @ W_TRUTH[0]) + 5.0 * np.tanh(u @ W_TRUTH[1])
    f2 = 5.0 * np.tanh(u @ W_TRUTH[2]) + 5.0 * np.tanh(u @ W_TRUTH[3])
    return np.stack([f1, f2], axis=1)


def true_g_from_u0(u0: np.ndarray) -> np.ndarray:
    """Truth diffusion levels for input rows [t, initial lags(8)]."""
    u0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))
    g1 = 4.0 / (1.0 + np.exp(-(u0 @ W_TRUTH[4])))
    g2 = (1.0 / 8.0) / (1.0 + np.exp(-(u0 @ W_TRUTH[5] + 1.0)))
    return np.stack([g1, g2], axis=1)


def true_tanh_args(u: np.ndarray) -> np.ndarray:
    """The four tanh-unit arguments (n, 4) for regime splits."""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    return np.stack([u @ W_TRUTH[i] for i in range(4)], axis=1)


def history_values(z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """eta(t) = [sin(z1 t), cos(z2 t)] for z (n, 2), times (m,) -> (n, m, 2)."""
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    return np.stack([np.sin(z[:, [0]] * t[None, :]),
                     np.cos(z[:, [1]] * t[None, :])], axis=2)


def generate_data(cfg: dict) -> dict[str, Any]:
    """Simulate all trajectories at dt_fine and observe every dt_obs.

    Returns observed trajectories (n, n_obs, 2), history parameters
    z (n, 2), and the per-trajectory initial lag vectors (n, 8).
    """
    rng = np.random.default_rng(int(cfg["seed"]))
    n_paths = int(cfg["data"]["n_trajectories"])
    t_end = float(cfg["data"]["t_end"])
    dt_obs = float(cfg["data"]["dt_obs"])
    n_fine = int(round(t_end / DT_FINE))
    n_hist = int(round(4.0 / DT_FINE))
    n_obs = int(round(t_end / dt_obs)) + 1
    obs_every = int(round(dt_obs / DT_FINE))
    z = rng.standard_normal((n_paths, 2))
    hist_t = -np.arange(n_hist, 0, -1) * DT_FINE  # -4.00 .. -0.01
    hist = history_values(z, hist_t)
    buf = np.zeros((n_paths, n_hist + n_fine + 1, 2))
    buf[:, :n_hist, :] = hist
    buf[:, n_hist, :] = np.stack([np.zeros(n_paths), np.ones(n_paths)], axis=1)
    lag_idx0 = [n_hist + int(lag / DT_FINE) for lag in LAGS]
    init_lag = np.concatenate([buf[:, lag_idx0, 0], buf[:, lag_idx0, 1]], axis=1)
    for k in range(1, n_fine + 1):
        idx = n_hist + k
        li = [idx + int(lag / DT_FINE) for lag in LAGS]
        lags = np.concatenate([buf[:, li, 0], buf[:, li, 1]], axis=1)
        u = np.concatenate([np.full((n_paths, 1), (k - 1) * DT_FINE), lags], axis=1)
        u0 = np.concatenate([np.full((n_paths, 1), (k - 1) * DT_FINE), init_lag], axis=1)
        dW = rng.standard_normal((n_paths, 2)) * np.sqrt(DT_FINE)
        buf[:, idx, :] = (buf[:, idx - 1, :] + true_drift_from_u(u) * DT_FINE
                          + true_g_from_u0(u0) * dW)
    obs = buf[:, n_hist::obs_every, :]
    assert obs.shape == (n_paths, n_obs, 2)
    return {"obs": obs, "z": z, "init_lag": init_lag}


def build_queries(obs: np.ndarray, z: np.ndarray, init_lag: np.ndarray,
                  dt_obs: float) -> dict[str, np.ndarray]:
    """Query arrays for one split.

    aug (n, 10) = [current, lags(8)]; t (n,) observation time;
    u_drift (n, 9) = [t, lags]; u0_diff (n, 9) = [t, init lags];
    cur (n, 2); nxt (n, 2) next observed state (transitions only).
    For observation times below 4, lag values come from the analytic
    history; otherwise from the observed trajectory itself.
    """
    n_paths, n_obs, _ = obs.shape
    times = np.arange(n_obs) * dt_obs
    rows_aug, rows_t, rows_cur = [], [], []
    for k in range(n_obs):
        tk = times[k]
        lag_vals = []
        for lag in LAGS:
            lt = tk + lag
            if lt < 0:
                lag_vals.append(history_values(z, np.array([lt]))[:, 0, :])
            else:
                lag_vals.append(obs[:, int(round(lt / dt_obs)), :])
        lags = np.concatenate([np.stack([lv[:, 0] for lv in lag_vals], axis=1),
                               np.stack([lv[:, 1] for lv in lag_vals], axis=1)], axis=1)
        rows_aug.append(np.concatenate([obs[:, k, :], lags], axis=1))
        rows_t.append(np.full(n_paths, tk))
        rows_cur.append(obs[:, k, :])
    aug = np.concatenate(rows_aug, axis=0)
    t = np.concatenate(rows_t, axis=0)
    cur = np.concatenate(rows_cur, axis=0)
    init_rep = np.repeat(init_lag, n_obs, axis=0)
    u_drift = np.concatenate([t[:, None], aug[:, 2:]], axis=1)
    u0_diff = np.concatenate([t[:, None], init_rep], axis=1)
    return {"aug": aug, "t": t, "cur": cur, "u_drift": u_drift,
            "u0_diff": u0_diff, "init_rep": init_rep}


def build_transitions(q: dict[str, np.ndarray], obs: np.ndarray,
                      dt_obs: float) -> dict[str, np.ndarray]:
    """One-step transition arrays aligned with query rows k >= 1."""
    n_paths, n_obs, _ = obs.shape
    idx = []
    for k in range(n_obs - 1):
        idx.append(np.arange(k * n_paths, (k + 1) * n_paths))
    sel = np.concatenate(idx)
    return {"u_drift": q["u_drift"][sel], "u0_diff": q["u0_diff"][sel],
            "cur": q["cur"][sel], "aug": q["aug"][sel],
            "nxt": obs[:, 1:, :].transpose(1, 0, 2).reshape(-1, 2), "t": q["t"][sel]}


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


def _minibatch_train(drift: MLP | None, diff: MLP | None, tr: dict,
                     dt: float, mode: str, epochs: int, batch: int,
                     lr: float, seed: int) -> dict[str, float]:
    """Train drift and/or diffusion nets.

    mode 'two_stage_drift': MSE on predicted next state.
    mode 'two_stage_diff': MSE on g^2 * dt vs squared residuals
    (drift net frozen; pass drift for predictions).
    mode 'em': joint Gaussian NLL with frozen interval coefficients.
    """
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    nets = [m.net for m in (drift, diff) if m is not None]
    opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=lr)
    n = len(tr["cur"])
    last: dict[str, float] = {}
    for _ in range(epochs):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            cur = torch.tensor(tr["cur"][idx], dtype=torch.float32)
            nxt = torch.tensor(tr["nxt"][idx], dtype=torch.float32)
            if mode == "two_stage_drift":
                assert drift is not None
                pred = cur + drift.forward_raw(tr["u_drift"][idx]) * dt
                loss = torch.mean((pred - nxt) ** 2)
            elif mode == "two_stage_diff":
                assert drift is not None and diff is not None
                with torch.no_grad():
                    pred = cur + drift.forward_raw(tr["u_drift"][idx]) * dt
                resid2 = (pred - nxt) ** 2
                raw = diff.forward_raw(tr["u0_diff"][idx])
                g = torch.nn.functional.softplus(raw) + 0.01
                loss = torch.mean((g ** 2 * dt - resid2) ** 2)
            elif mode == "em":
                assert drift is not None and diff is not None
                f = drift.forward_raw(tr["u_drift"][idx])
                raw = diff.forward_raw(tr["u0_diff"][idx])
                g = torch.nn.functional.softplus(raw) + 0.01
                var = g ** 2 * dt
                resid = nxt - cur - f * dt
                loss = torch.mean(0.5 * (np.log(2 * np.pi)
                                         + torch.log(var) + resid ** 2 / var))
            else:  # pragma: no cover
                raise ValueError(mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        last["loss"] = tot / max(cnt, 1)
    return last


def teacher_drift_predict(model: dict, u_drift: np.ndarray) -> np.ndarray:
    return model["drift"].predict(u_drift)


def teacher_g_predict(model: dict, u0_diff: np.ndarray) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        raw = model["diff"].forward_raw(u0_diff).numpy().astype(np.float64)
    # softplus in float64 for stability
    return np.logaddexp(0.0, raw) + 0.01


def build_teacher(in_drift: int, in_diff: int) -> dict:
    return {"drift": MLP(in_drift, 2), "diff": MLP(in_diff, 2)}


# ---------------------------------------------------------------- SINDy

def fit_field(x: np.ndarray, y: np.ndarray, x_val: np.ndarray,
              y_val: np.ndarray, degrees, thresholds, ridge: float):
    """Validation-selected polynomial field; returns (field, val, meta)."""
    best = None
    for degree in degrees:
        for thr in thresholds:
            field = fit_polynomial_field(x, y, degree, thr, ridge)
            val = nrmse(y_val, field.predict(x_val))
            if best is None or val < best[1]:
                best = (field, val, {"degree": degree, "threshold": thr})
    assert best is not None
    return best


def vif_scores(x: np.ndarray) -> np.ndarray:
    """Variance inflation factors for columns of x (train queries)."""
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

def simulate_sdde(drift_fn, diff_fn, z: np.ndarray, dt: float, n_steps: int,
                  increments: np.ndarray, failure_radius: float,
                  excursion_radius: float) -> dict[str, Any]:
    """Paired SDDE simulation from history segments eta.

    drift_fn takes (t, lags(8)) rows; diff_fn takes (t, init lags(8))
    rows. Lags below t = 0 come from the analytic history; later lags
    come from the simulated path itself. Failures latch and are NaN
    afterwards; excursions are counted separately from failures.
    """
    n = len(z)
    hist_t = np.array([-4.0, -3.0, -2.0, -1.0])
    hist_at_lags = history_values(z, hist_t)  # (n, 4, 2)
    init_lag = np.concatenate([hist_at_lags[:, :, 0], hist_at_lags[:, :, 1]], axis=1)
    sim = np.zeros((n, n_steps + 1, 2))
    sim[:, 0, :] = np.stack([np.zeros(n), np.ones(n)], axis=1)
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    ever_excursion = np.zeros(n, dtype=bool)
    for k in range(1, n_steps + 1):
        tk = (k - 1) * dt
        lag_vals = []
        for lag in LAGS:
            lt = tk + lag
            if lt < 0:
                lag_vals.append(history_values(z, np.array([lt]))[:, 0, :])
            else:
                lag_vals.append(sim[:, int(round(lt / dt)), :])
        lags = np.concatenate([np.stack([lv[:, 0] for lv in lag_vals], axis=1),
                               np.stack([lv[:, 1] for lv in lag_vals], axis=1)], axis=1)
        active = ~failed
        if active.any():
            u_drift = np.concatenate([np.full((int(active.sum()), 1), tk),
                                      lags[active]], axis=1)
            u0_diff = np.concatenate([np.full((int(active.sum()), 1), tk),
                                      init_lag[active]], axis=1)
            f = drift_fn(u_drift)
            g = diff_fn(u0_diff)
            sim[active, k, :] = (sim[active, k - 1, :]
                                 + f * dt + g * increments[active, k - 1] * np.sqrt(dt))
        radii = np.linalg.norm(sim[:, k, :], axis=1)
        ever_excursion = ever_excursion | ((radii > excursion_radius) & active)
        bad_nonfinite = ~np.isfinite(sim[:, k, :]).all(axis=1)
        bad_radius = radii > failure_radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        sim[failed, k, :] = np.nan
    return {"states": sim, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion_rate": float(np.mean(ever_excursion))}


# ---------------------------------------------------------------- main

def _split_indices(n: int, cfg: dict, rng: np.random.Generator):
    perm = rng.permutation(n)
    n_train = int(cfg["split"]["train"])
    n_val = int(cfg["split"]["validation"])
    return (perm[:n_train], perm[n_train:n_train + n_val],
            perm[n_train + n_val:n_train + n_val + int(cfg["split"]["test"])])


def _field_table(name: str, pred: np.ndarray, ref: np.ndarray) -> dict:
    return {"nrmse": nrmse(ref, pred),
            "rms_error": float(np.sqrt(np.mean((pred - ref) ** 2))),
            "ref_rms": float(np.sqrt(np.mean(ref ** 2)))}


def main() -> None:
    import torch
    t_start = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    dt_obs = float(cfg["data"]["dt_obs"])
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    print("generating data ...", flush=True)
    data = generate_data(cfg)
    obs, z, init_lag = data["obs"], data["z"], data["init_lag"]
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    idx_tr, idx_va, idx_te = _split_indices(len(obs), cfg, rng)
    splits = {}
    for name, idx in (("train", idx_tr), ("validation", idx_va), ("test", idx_te)):
        q = build_queries(obs[idx], z[idx], init_lag[idx], dt_obs)
        splits[name] = {"q": q, "tr": build_transitions(q, obs[idx], dt_obs),
                        "z": z[idx], "init_lag": init_lag[idx],
                        "obs": obs[idx]}
    print("state stats train: X1 mean %.3f std %.3f range [%.1f, %.1f]; "
          "X2 mean %.3f std %.3f range [%.1f, %.1f]" % (
              splits["train"]["q"]["cur"][:, 0].mean(),
              splits["train"]["q"]["cur"][:, 0].std(),
              splits["train"]["q"]["cur"][:, 0].min(),
              splits["train"]["q"]["cur"][:, 0].max(),
              splits["train"]["q"]["cur"][:, 1].mean(),
              splits["train"]["q"]["cur"][:, 1].std(),
              splits["train"]["q"]["cur"][:, 1].min(),
              splits["train"]["q"]["cur"][:, 1].max()), flush=True)

    # --- collinearity gate (Day 1) on train query states
    aug_tr = splits["train"]["q"]["aug"]
    corr = np.corrcoef(aug_tr.T)
    vif = vif_scores(aug_tr)
    aug_names = ["x", "y", "x_{t-4}", "x_{t-3}", "x_{t-2}", "x_{t-1}",
                 "y_{t-4}", "y_{t-3}", "y_{t-2}", "y_{t-1}"]
    print("collinearity: max |corr| current-vs-lags %.3f, "
          "max |corr| within lags %.3f, max VIF %.1f" % (
              np.abs(corr[:2, 2:]).max(),
              np.abs(corr[2:, 2:] - np.eye(8)).max(), vif.max()), flush=True)

    # --- teachers
    tcfg = cfg["teacher"]
    hidden = tuple(tcfg["hidden"])

    def make_delay() -> dict:
        m = build_teacher(9, 9)
        m["drift"] = MLP(9, 2, hidden)
        m["diff"] = MLP(9, 2, hidden)
        return m

    def make_markov() -> dict:
        m = {"drift": MLP(3, 2, hidden), "diff": MLP(3, 2, hidden)}
        return m

    def markov_inputs(q, tr):
        tr_m = dict(tr)
        tr_m["u_drift"] = np.concatenate([tr["t"][:, None], tr["cur"]], axis=1)
        tr_m["u0_diff"] = np.concatenate([tr["t"][:, None], tr["cur"]], axis=1)
        return tr_m

    print("training delay two-stage teacher ...", flush=True)
    delay = make_delay()
    delay["drift"].fit_standardizer(splits["train"]["tr"]["u_drift"])
    delay["diff"].fit_standardizer(splits["train"]["tr"]["u0_diff"])
    _minibatch_train(delay["drift"], None, splits["train"]["tr"], dt_obs,
                     "two_stage_drift", int(tcfg["two_stage_drift"]["epochs"]),
                     int(tcfg["two_stage_drift"]["batch"]),
                     float(tcfg["two_stage_drift"]["lr"]), 0)
    _minibatch_train(delay["drift"], delay["diff"], splits["train"]["tr"], dt_obs,
                     "two_stage_diff", int(tcfg["two_stage_diffusion"]["epochs"]),
                     int(tcfg["two_stage_diffusion"]["batch"]),
                     float(tcfg["two_stage_diffusion"]["lr"]), 0)

    print("training markovian two-stage teacher ...", flush=True)
    markov = make_markov()
    tr_m = markov_inputs(None, splits["train"]["tr"])
    markov["drift"].fit_standardizer(tr_m["u_drift"])
    markov["diff"].fit_standardizer(tr_m["u0_diff"])
    _minibatch_train(markov["drift"], None, tr_m, dt_obs, "two_stage_drift",
                     int(tcfg["two_stage_drift"]["epochs"]),
                     int(tcfg["two_stage_drift"]["batch"]),
                     float(tcfg["two_stage_drift"]["lr"]), 0)
    _minibatch_train(markov["drift"], markov["diff"], tr_m, dt_obs,
                     "two_stage_diff", int(tcfg["two_stage_diffusion"]["epochs"]),
                     int(tcfg["two_stage_diffusion"]["batch"]),
                     float(tcfg["two_stage_diffusion"]["lr"]), 0)

    print("training EM teachers ...", flush=True)
    em = make_delay()
    em["drift"].fit_standardizer(splits["train"]["tr"]["u_drift"])
    em["diff"].fit_standardizer(splits["train"]["tr"]["u0_diff"])
    _minibatch_train(em["drift"], em["diff"], splits["train"]["tr"], dt_obs,
                     "em", int(tcfg["em_main"]["epochs"]),
                     int(tcfg["em_main"]["batch"]), float(tcfg["em_main"]["lr"]), 0)

    # Coarsened EM (dt = 0.5): subsample train observations every 10th step.
    coarse_obs = splits["train"]["obs"][:, ::10, :]
    q_coarse = build_queries(coarse_obs, splits["train"]["z"],
                             splits["train"]["init_lag"], 0.5)
    tr_coarse = build_transitions(q_coarse, coarse_obs, 0.5)
    em_coarse = make_delay()
    em_coarse["drift"].fit_standardizer(tr_coarse["u_drift"])
    em_coarse["diff"].fit_standardizer(tr_coarse["u0_diff"])
    _minibatch_train(em_coarse["drift"], em_coarse["diff"], tr_coarse, 0.5,
                     "em", int(tcfg["em_coarsened"]["epochs"]),
                     int(tcfg["em_coarsened"]["batch"]),
                     float(tcfg["em_coarsened"]["lr"]), 0)
    # Count-matched fine EM: same number of transitions, dt = 0.05.
    rng_sub = np.random.default_rng(7)
    sub = rng_sub.choice(len(splits["train"]["tr"]["cur"]),
                         size=len(tr_coarse["cur"]), replace=False)
    tr_fine_sub = {k: v[sub] for k, v in splits["train"]["tr"].items()}
    em_fine_sub = make_delay()
    em_fine_sub["drift"].fit_standardizer(tr_fine_sub["u_drift"])
    em_fine_sub["diff"].fit_standardizer(tr_fine_sub["u0_diff"])
    _minibatch_train(em_fine_sub["drift"], em_fine_sub["diff"], tr_fine_sub,
                     dt_obs, "em", int(tcfg["em_fine_sub"]["epochs"]),
                     int(tcfg["em_fine_sub"]["batch"]),
                     float(tcfg["em_fine_sub"]["lr"]), 0)

    # Delta t = 1 diagnostic pair: two-stage teachers retrained on
    # observations subsampled to every 20th step (paper regime).
    diag_obs = splits["train"]["obs"][:, ::20, :]
    q_diag = build_queries(diag_obs, splits["train"]["z"],
                           splits["train"]["init_lag"], 1.0)
    tr_diag = build_transitions(q_diag, diag_obs, 1.0)
    delay_d1 = make_delay()
    delay_d1["drift"].fit_standardizer(tr_diag["u_drift"])
    delay_d1["diff"].fit_standardizer(tr_diag["u0_diff"])
    _minibatch_train(delay_d1["drift"], None, tr_diag, 1.0, "two_stage_drift",
                     40, 2048, 0.003, 0)
    markov_d1 = make_markov()
    tr_diag_m = markov_inputs(None, tr_diag)
    markov_d1["drift"].fit_standardizer(tr_diag_m["u_drift"])
    markov_d1["diff"].fit_standardizer(tr_diag_m["u0_diff"])
    _minibatch_train(markov_d1["drift"], None, tr_diag_m, 1.0,
                     "two_stage_drift", 40, 2048, 0.003, 0)

    # --- test query evaluations
    te = splits["test"]["q"]
    va = splits["validation"]["q"]
    tr_q = splits["train"]["q"]
    truth_drift_te = true_drift_from_u(te["u_drift"])
    truth_g_te = true_g_from_u0(te["u0_diff"])
    truth_drift_va = true_drift_from_u(va["u_drift"])

    def teacher_fields(model, drift_in, diff_in, q):
        return (model["drift"].predict(drift_in),
                teacher_g_predict(model, diff_in))

    delay_drift_te, delay_g_te = teacher_fields(
        delay, te["u_drift"], te["u0_diff"], te)
    markov_drift_te, markov_g_te = teacher_fields(
        markov, np.concatenate([te["t"][:, None], te["cur"]], axis=1),
        np.concatenate([te["t"][:, None], te["cur"]], axis=1), te)
    em_drift_te, em_g_te = teacher_fields(em, te["u_drift"], te["u0_diff"], te)
    emc_drift_te, emc_g_te = teacher_fields(em_coarse, te["u_drift"],
                                            te["u0_diff"], te)
    emf_drift_te, emf_g_te = teacher_fields(em_fine_sub, te["u_drift"],
                                            te["u0_diff"], te)
    delay_d1_drift_te, _ = teacher_fields(
        delay_d1, te["u_drift"], te["u0_diff"], te)
    markov_d1_drift_te, _ = teacher_fields(
        markov_d1, np.concatenate([te["t"][:, None], te["cur"]], axis=1),
        np.concatenate([te["t"][:, None], te["cur"]], axis=1), te)

    def sigma_vec(g):
        return g ** 2  # diagonal Sigma: (g1^2, g2^2)

    teachers_table = {}
    for name, f, g in (("delay_two_stage", delay_drift_te, delay_g_te),
                       ("markovian_two_stage", markov_drift_te, markov_g_te),
                       ("em", em_drift_te, em_g_te),
                       ("em_coarse_dt0.5", emc_drift_te, emc_g_te),
                       ("em_fine_sub", emf_drift_te, emf_g_te)):
        teachers_table[name] = {
            "drift_vs_truth": _field_table(name, f, truth_drift_te),
            "drift_x1_vs_truth": _field_table(name, f[:, [0]], truth_drift_te[:, [0]]),
            "g_vs_truth": _field_table(name, g, truth_g_te),
            "sigma_vs_truth": _field_table(name, sigma_vec(g), sigma_vec(truth_g_te)),
        }
    print("teachers vs truth (drift NRMSE): " + ", ".join(
        f"{k}={v['drift_vs_truth']['nrmse']:.4f}"
        for k, v in teachers_table.items()), flush=True)

    # --- H5 paired comparison: delay vs markovian teachers (test queries)
    def paired_gap(err_a, err_b):
        d = err_a ** 2 - err_b ** 2  # markovian minus delay
        return {"mean": float(np.mean(d)),
                "se": float(np.std(d, ddof=1) / np.sqrt(len(d))),
                "rmse_a": float(np.sqrt(np.mean(err_a ** 2))),
                "rmse_b": float(np.sqrt(np.mean(err_b ** 2)))}
    h5_drift = paired_gap(markov_drift_te[:, 0] - truth_drift_te[:, 0],
                          delay_drift_te[:, 0] - truth_drift_te[:, 0])
    h5_var = paired_gap(markov_g_te[:, 0] ** 2 - truth_g_te[:, 0] ** 2,
                        delay_g_te[:, 0] ** 2 - truth_g_te[:, 0] ** 2)
    h5_drift_d1 = paired_gap(markov_d1_drift_te[:, 0] - truth_drift_te[:, 0],
                             delay_d1_drift_te[:, 0] - truth_drift_te[:, 0])
    h5 = {"drift_x1": h5_drift, "variance_x1": h5_var,
          "drift_x1_dt1_diagnostic": h5_drift_d1,
          "pass_drift": bool(h5_drift["mean"] > 0
                             and h5_drift["mean"] > 2 * h5_drift["se"]),
          "pass_variance": bool(h5_var["mean"] > 0
                                and h5_var["mean"] > 2 * h5_var["se"])}
    h5["pass"] = bool(h5["pass_drift"] and h5["pass_variance"])
    print("H5 paired gaps: drift %s variance %s" % (h5_drift, h5_var), flush=True)

    # --- H6 discretization diagnostic: EM coarse vs fine-sub
    args_te = true_tanh_args(te["u_drift"])
    max_arg = np.abs(args_te).max(axis=1)
    near = max_arg < 0.3
    satur = max_arg > 0.7

    def regime_nrmse(f, ref, mask):
        return nrmse(ref[mask], f[mask]) if mask.any() else float("nan")
    h6 = {
        "nrmse_em_fine_sub": nrmse(truth_drift_te, emf_drift_te),
        "nrmse_em_coarse": nrmse(truth_drift_te, emc_drift_te),
        "nrmse_em_fine_sub_near": regime_nrmse(emf_drift_te, truth_drift_te, near),
        "nrmse_em_coarse_near": regime_nrmse(emc_drift_te, truth_drift_te, near),
        "nrmse_em_fine_sub_satur": regime_nrmse(emf_drift_te, truth_drift_te, satur),
        "nrmse_em_coarse_satur": regime_nrmse(emc_drift_te, truth_drift_te, satur),
        "n_near": int(near.sum()), "n_satur": int(satur.sum()),
        "sigma_nrmse_em_fine_sub": nrmse(sigma_vec(truth_g_te), sigma_vec(emf_g_te)),
        "sigma_nrmse_em_coarse": nrmse(sigma_vec(truth_g_te), sigma_vec(emc_g_te)),
    }
    h6["degradation"] = h6["nrmse_em_coarse"] - h6["nrmse_em_fine_sub"]
    h6["degradation_near"] = h6["nrmse_em_coarse_near"] - h6["nrmse_em_fine_sub_near"]
    h6["degradation_satur"] = h6["nrmse_em_coarse_satur"] - h6["nrmse_em_fine_sub_satur"]
    h6["regime_gap"] = h6["degradation_satur"] - h6["degradation_near"]
    h6["pass_a"] = bool(h6["degradation"] >= 0.03)
    h6["pass_b"] = bool(h6["regime_gap"] >= 0.02)
    h6["pass"] = bool(h6["pass_a"] and h6["pass_b"])
    print("H6: degradation %.4f near %.4f satur %.4f gap %.4f" % (
        h6["degradation"], h6["degradation_near"], h6["degradation_satur"],
        h6["regime_gap"]), flush=True)

    # --- SINDy distillation of the delay teacher
    dist = cfg["distillation"]
    degrees = dist["drift_degrees"]
    thresholds = dist["thresholds"]
    ridge = float(dist["ridge"])
    delay_drift_tr = delay["drift"].predict(tr_q["u_drift"])
    delay_drift_va = delay["drift"].predict(va["u_drift"])
    delay_g_tr = teacher_g_predict(delay, tr_q["u0_diff"])
    delay_g_va = teacher_g_predict(delay, va["u0_diff"])
    drift_field, drift_val, drift_meta = fit_field(
        tr_q["aug"], delay_drift_tr, va["aug"], delay_drift_va,
        degrees, thresholds, ridge)
    diff_field, diff_val, diff_meta = fit_field(
        tr_q["u0_diff"], delay_g_tr, va["u0_diff"], delay_g_va,
        dist["diffusion_degrees"], thresholds, ridge)
    sym_drift_te = drift_field.predict(te["aug"])
    sym_g_te = diff_field.predict(te["u0_diff"])
    # Markovian symbolic (its own teacher) and oracles (truth fields).
    markov_drift_tr = markov["drift"].predict(
        np.concatenate([tr_q["t"][:, None], tr_q["cur"]], axis=1))
    markov_drift_va = markov["drift"].predict(
        np.concatenate([va["t"][:, None], va["cur"]], axis=1))
    markov_field, _, markov_meta = fit_field(
        tr_q["cur"], markov_drift_tr, va["cur"], markov_drift_va,
        degrees, thresholds, ridge)
    markov_sym_te = markov_field.predict(te["cur"])
    truth_drift_tr = true_drift_from_u(tr_q["u_drift"])
    oracle_drift, _, oracle_drift_meta = fit_field(
        tr_q["aug"], truth_drift_tr, va["aug"], truth_drift_va,
        degrees, thresholds, ridge)
    oracle_drift_te = oracle_drift.predict(te["aug"])
    truth_g_tr = true_g_from_u0(tr_q["u0_diff"])
    truth_g_va = true_g_from_u0(va["u0_diff"])
    oracle_diff, _, oracle_diff_meta = fit_field(
        tr_q["u0_diff"], truth_g_tr, va["u0_diff"], truth_g_va,
        dist["diffusion_degrees"], thresholds, ridge)
    oracle_g_te = oracle_diff.predict(te["u0_diff"])

    dist_table = {
        "drift_selection": drift_meta, "drift_validation_nrmse": drift_val,
        "diffusion_selection": diff_meta, "diffusion_validation_nrmse": diff_val,
        "drift_vs_teacher": _field_table("drift", sym_drift_te, delay_drift_te),
        "g_vs_teacher": _field_table("g", sym_g_te, delay_g_te),
        "sigma_vs_teacher": _field_table("sigma", sigma_vec(sym_g_te),
                                         sigma_vec(delay_g_te)),
        "drift_vs_truth": _field_table("drift", sym_drift_te, truth_drift_te),
        "g_vs_truth": _field_table("g", sym_g_te, truth_g_te),
        "sigma_vs_truth": _field_table("sigma", sigma_vec(sym_g_te),
                                       sigma_vec(truth_g_te)),
        "oracle_drift_vs_truth": _field_table("oracle", oracle_drift_te,
                                              truth_drift_te),
        "oracle_g_vs_truth": _field_table("oracle", oracle_g_te, truth_g_te),
        "oracle_drift_selection": oracle_drift_meta,
        "oracle_diff_selection": oracle_diff_meta,
        "markovian_symbolic_vs_teacher": _field_table(
            "markov", markov_sym_te, markov_drift_te),
        "markovian_symbolic_vs_truth": _field_table(
            "markov", markov_sym_te, truth_drift_te),
        "markovian_selection": markov_meta,
    }
    print("distillation: drift vs teacher %.4f, g vs teacher %.4f, "
          "drift vs truth %.4f, oracle drift vs truth %.4f" % (
              dist_table["drift_vs_teacher"]["nrmse"],
              dist_table["g_vs_teacher"]["nrmse"],
              dist_table["drift_vs_truth"]["nrmse"],
              dist_table["oracle_drift_vs_truth"]["nrmse"]), flush=True)

    # --- shuffled-lag diagnostic (Day 2 off-manifold analogue)
    rng_shuf = np.random.default_rng(99)
    perm = rng_shuf.permutation(len(te["aug"]))
    shuf_aug = te["aug"].copy()
    shuf_aug[:, 2:] = te["aug"][perm][:, 2:]
    shuf_u = np.concatenate([te["t"][:, None], shuf_aug[:, 2:]], axis=1)
    shuf_truth = true_drift_from_u(shuf_u)
    shuf_teacher = delay["drift"].predict(shuf_u)
    shuf_sym = drift_field.predict(shuf_aug)
    shuffled = {
        "symbolic_vs_teacher": _field_table("shuf", shuf_sym, shuf_teacher),
        "symbolic_vs_truth": _field_table("shuf", shuf_sym, shuf_truth),
        "teacher_vs_truth": _field_table("shuf", shuf_teacher, shuf_truth),
    }
    print("shuffled-lag: sym-vs-teacher %.4f sym-vs-truth %.4f "
          "teacher-vs-truth %.4f" % (
              shuffled["symbolic_vs_teacher"]["nrmse"],
              shuffled["symbolic_vs_truth"]["nrmse"],
              shuffled["teacher_vs_truth"]["nrmse"]), flush=True)

    # --- paired SDDE simulation (test history segments)
    sim_cfg = cfg["simulation"]
    n_te = len(idx_te)
    rng_sim = np.random.default_rng(int(sim_cfg["seed"]))
    inc = rng_sim.standard_normal((n_te, int(cfg["data"]["t_end"] / dt_obs), 2))
    z_te = splits["test"]["z"]
    truth_sim = simulate_sdde(
        lambda u: true_drift_from_u(u), lambda u: true_g_from_u0(u),
        z_te, dt_obs, inc.shape[1], inc, float(sim_cfg["failure_radius"]),
        float(sim_cfg["domain_excursion_radius"]))
    teacher_sim = simulate_sdde(
        lambda u: delay["drift"].predict(u),
        lambda u: teacher_g_predict(delay, u),
        z_te, dt_obs, inc.shape[1], inc, float(sim_cfg["failure_radius"]),
        float(sim_cfg["domain_excursion_radius"]))
    # The symbolic SDDE needs the current state inside the drift call,
    # so it is simulated with its own loop (same failure/excursion
    # rules as simulate_sdde).
    sym_sim = _simulate_symbolic(
        drift_field, diff_field, z_te, dt_obs, inc.shape[1], inc,
        float(sim_cfg["failure_radius"]),
        float(sim_cfg["domain_excursion_radius"]))
    paired_rmse = []
    for i in range(n_te):
        a = teacher_sim["states"][i]
        b = sym_sim["states"][i]
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if ok.sum() >= 10:
            paired_rmse.append(np.sqrt(np.mean(
                np.sum((a[ok] - b[ok]) ** 2, axis=1))))
    paired_rmse = np.asarray(paired_rmse)
    sim_table = {
        "teacher": {"failure_rate": teacher_sim["failure_rate"],
                    "excursion_rate": teacher_sim["excursion_rate"]},
        "symbolic": {"failure_rate": sym_sim["failure_rate"],
                     "excursion_rate": sym_sim["excursion_rate"]},
        "truth": {"failure_rate": truth_sim["failure_rate"],
                  "excursion_rate": truth_sim["excursion_rate"]},
        "paired_rmse_median": float(np.median(paired_rmse)) if len(paired_rmse) else None,
        "paired_rmse_mean": float(np.mean(paired_rmse)) if len(paired_rmse) else None,
        "n_paired": int(len(paired_rmse)),
    }
    print("simulation: teacher fail %.4f exc %.4f; symbolic fail %.4f "
          "exc %.4f; paired RMSE median %s" % (
              sim_table["teacher"]["failure_rate"],
              sim_table["teacher"]["excursion_rate"],
              sim_table["symbolic"]["failure_rate"],
              sim_table["symbolic"]["excursion_rate"],
              sim_table["paired_rmse_median"]), flush=True)

    # --- H7 downstream: one-step X^1 MSE, symbolic vs teacher
    te_tr = splits["test"]["tr"]
    tea_pred = te_tr["cur"][:, [0]] + delay["drift"].predict(
        te_tr["u_drift"])[:, [0]] * dt_obs
    sym_pred = te_tr["cur"][:, [0]] + drift_field.predict(
        te_tr["aug"])[:, [0]] * dt_obs
    mse_tea = float(np.mean((tea_pred - te_tr["nxt"][:, [0]]) ** 2))
    mse_sym = float(np.mean((sym_pred - te_tr["nxt"][:, [0]]) ** 2))
    d = (sym_pred - te_tr["nxt"][:, [0]]) ** 2 - (tea_pred - te_tr["nxt"][:, [0]]) ** 2
    h7 = {"mse_teacher": mse_tea, "mse_symbolic": mse_sym,
          "abs_diff": abs(mse_sym - mse_tea),
          "paired_se": float(np.std(d.ravel(), ddof=1) / np.sqrt(d.size)),
          "pass": bool(abs(mse_sym - mse_tea) <= 0.005
                        and float(np.std(d.ravel(), ddof=1) / np.sqrt(d.size)) < 0.0025)}
    print("H7: MSE teacher %.5f symbolic %.5f diff %.5f SE %.5f" % (
        mse_tea, mse_sym, mse_sym - mse_tea, h7["paired_se"]), flush=True)

    # --- verdicts (frozen thresholds in the config)
    hyp = cfg["hypotheses"]
    verdicts = {
        "H1_drift_distillation": bool(
            dist_table["drift_vs_teacher"]["nrmse"]
            <= hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"]),
        "H2_diffusion_distillation": bool(
            dist_table["g_vs_teacher"]["nrmse"]
            <= hyp["H2_diffusion_distillation"]["symbolic_diffusion_test_nrmse_vs_teacher_max"]
            and dist_table["sigma_vs_teacher"]["nrmse"]
            <= hyp["H2_diffusion_distillation"]["symbolic_sigma_test_nrmse_vs_teacher_max"]),
        "H3_truth_recovery": bool(
            dist_table["drift_vs_truth"]["nrmse"]
            <= hyp["H3_truth_recovery"]["field_symbolic_drift_test_nrmse_vs_truth_max"]),
        "H4_simulation": bool(
            sim_table["symbolic"]["failure_rate"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_max"]
            and abs(sim_table["symbolic"]["failure_rate"]
                    - sim_table["teacher"]["failure_rate"])
            <= hyp["H4_simulation"]["symbolic_failure_rate_minus_teacher_max"]
            and sim_table["paired_rmse_median"]
            <= hyp["H4_simulation"]["median_paired_rmse_vs_teacher_max"]
            and (sim_table["symbolic"]["excursion_rate"]
                 - sim_table["teacher"]["excursion_rate"])
            <= hyp["H4_simulation"]["domain_excursion_symbolic_minus_teacher_max"]),
        "H5_paper_axis_delay_vs_markovian": h5["pass"],
        "H6_discretization_bias": h6["pass"],
        "H7_downstream_one_step": h7["pass"],
    }
    print("verdicts:", verdicts, flush=True)

    # --- TeX symbolic SDDE (original coordinates + delay terms)
    drift_exp = expand_to_original(drift_field)
    diff_exp = expand_to_original(diff_field)
    drift_tex = tex_equations(
        drift_exp, ("\\hat f_x", "\\hat f_y"),
        tuple(aug_names))
    diff_tex = tex_equations(
        diff_exp, ("\\hat g^1", "\\hat g^2"),
        ("t", "x_{-4}", "x_{-3}", "x_{-2}", "x_{-1}",
         "y_{-4}", "y_{-3}", "y_{-2}", "y_{-1}"))
    tex_doc = ("%% Day 6 symbolic SDDE (Eggen & Midtfjord 2023 Delay-SDE-Net, "
               "distilled delay teacher; x = X^1, y = X^2; "
               "x_{t-k}, y_{t-k} are lagged states; "
               "x_{-k}, y_{-k} in the diffusion are initial-segment values)\n"
               + drift_tex + "\n" + diff_tex + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "symbolic_sde.tex").write_text(tex_doc)
    drift_field.save(OUT_DIR / "symbolic_drift.json")
    diff_field.save(OUT_DIR / "symbolic_diffusion.json")
    markov_field.save(OUT_DIR / "markovian_symbolic_drift.json")
    teacher_states = {
        "delay_two_stage": {"drift": delay["drift"].state(),
                            "diff": delay["diff"].state()},
        "markovian_two_stage": {"drift": markov["drift"].state(),
                                "diff": markov["diff"].state()},
        "em": {"drift": em["drift"].state(), "diff": em["diff"].state()},
    }
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(teacher_states))
    results = {
        "day": 6, "date": cfg["date"], "paper": cfg["paper"],
        "split_sizes": {"train": int(len(idx_tr)), "validation": int(len(idx_va)),
                        "test": int(len(idx_te))},
        "state_stats_train": {
            "x1": {"mean": float(splits["train"]["q"]["cur"][:, 0].mean()),
                   "std": float(splits["train"]["q"]["cur"][:, 0].std())},
            "x2": {"mean": float(splits["train"]["q"]["cur"][:, 1].mean()),
                   "std": float(splits["train"]["q"]["cur"][:, 1].std())}},
        "collinearity": {
            "max_abs_corr_current_vs_lags": float(np.abs(corr[:2, 2:]).max()),
            "max_abs_corr_within_lags": float(np.abs(corr[2:, 2:] - np.eye(8)).max()),
            "vif": {name: float(v) for name, v in zip(aug_names, vif)}},
        "teachers_vs_truth": teachers_table,
        "distillation": dist_table,
        "shuffled_lag": shuffled,
        "H5": h5, "H6": h6, "H7": h7,
        "simulation": sim_table,
        "verdicts": verdicts,
        "runtime_seconds": round(time.time() - t_start, 1),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUT_DIR, flush=True)


def _simulate_symbolic(drift_field: PolynomialField, diff_field: PolynomialField,
                       z: np.ndarray, dt: float, n_steps: int,
                       increments: np.ndarray, failure_radius: float,
                       excursion_radius: float) -> dict[str, Any]:
    """SDDE simulation where the drift is a polynomial in (current, lags).

    Mirrors simulate_sdde's failure/excursion rules; the drift call
    needs the current state, so the loop is specialized here.
    """
    n = len(z)
    hist_t = np.array([-4.0, -3.0, -2.0, -1.0])
    hist_at_lags = history_values(z, hist_t)
    init_lag = np.concatenate([hist_at_lags[:, :, 0], hist_at_lags[:, :, 1]], axis=1)
    sim = np.zeros((n, n_steps + 1, 2))
    sim[:, 0, :] = np.stack([np.zeros(n), np.ones(n)], axis=1)
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    ever_excursion = np.zeros(n, dtype=bool)
    for k in range(1, n_steps + 1):
        tk = (k - 1) * dt
        lag_vals = []
        for lag in LAGS:
            lt = tk + lag
            if lt < 0:
                lag_vals.append(history_values(z, np.array([lt]))[:, 0, :])
            else:
                lag_vals.append(sim[:, int(round(lt / dt)), :])
        lags = np.concatenate([np.stack([lv[:, 0] for lv in lag_vals], axis=1),
                               np.stack([lv[:, 1] for lv in lag_vals], axis=1)], axis=1)
        active = ~failed
        if active.any():
            aug = np.concatenate([sim[active, k - 1, :], lags[active]], axis=1)
            u0_diff = np.concatenate([np.full((int(active.sum()), 1), tk),
                                      init_lag[active]], axis=1)
            f = drift_field.predict(aug)
            g = diff_field.predict(u0_diff)
            sim[active, k, :] = (sim[active, k - 1, :]
                                 + f * dt + g * increments[active, k - 1] * np.sqrt(dt))
        radii = np.linalg.norm(sim[:, k, :], axis=1)
        ever_excursion = ever_excursion | ((radii > excursion_radius) & active)
        bad_nonfinite = ~np.isfinite(sim[:, k, :]).all(axis=1)
        bad_radius = radii > failure_radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        sim[failed, k, :] = np.nan
    return {"states": sim, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion_rate": float(np.mean(ever_excursion))}


if __name__ == "__main__":
    main()
