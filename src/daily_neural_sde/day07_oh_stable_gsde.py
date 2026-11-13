"""Day 7: Oh, Lim & Kim (2024) Stable Neural SDEs (Geometric SDE) -> SINDy.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day07.json``):

1. Ground truth: an identifiable observable 2D Geometric SDE in the
   paper's Eq. (6) form dZ/Z = gamma dt + sigma dW, i.e.
   dX^1 = X^1 * (0.60 - 0.25 X^1 - 0.10 X^2) dt + 0.30 X^1 dW^1,
   dX^2 = X^2 * (-0.10 + 0.20 X^1 - 0.30 X^2) dt + 0.25 X^2 dW^2.
   The interior equilibrium is (2, 1) with a stable-spiral Jacobian.
   Trajectories are simulated by Euler-Maruyama at dt_fine = 0.01
   and observed every 0.05 on [0, 10]; initial states are lognormal
   (Day 5 reflection: many initial conditions, never one).
   The paper's latent classification pipeline, controlled path, and
   real missing-data benchmarks are cut and declared in the config:
   the teacher is reconstructed in identifiable coordinates.
2. Teachers (all EM likelihood, variable dt for thinned data):
   a GSDE-structured teacher (gamma net with bounded 1.5*tanh output,
   sigma net on t only, effective f^j = X^j gamma^j,
   g^j = X^j sigma^j(t)) and a naive free-MLP teacher, each trained
   on dense (dt = 0.05) and on 50%-thinned (paper's missing-rate
   regime) train transitions. The distilled teacher is the
   GSDE-thinned teacher, fixed a priori because the paper's setting
   is irregular (missing) data. A GSDE-structured EM pair at
   dt = 0.5 vs transition-count-matched dt = 0.05 is the Day 1
   discretization diagnostic (H6), regime-split by state radius.
3. Distillation of the GSDE-thinned teacher into polynomial SINDy:
   drift library in (X^1, X^2) (truth is time-homogeneous),
   diffusion per component in (t, own state) only (the GSDE
   structure; Day 6 reflection: diffusion inputs verified
   separately from drift). Direct function queries on
   trajectory-disjoint splits, validation-only selection, test once.
   A collinearity gate precedes fitting (Day 1): linear VIF ~1.05
   supports a linear-coefficient clause (H3c), while the degree-2
   library VIF (16-26 on prototype queries) forbids a quadratic
   coefficient verdict, so quadratic coefficients are tabulated
   without pass/fail. A positive-quadrant grid (including the
   near-zero corner) is the Day 2 off-manifold diagnostic, reported
   only, never used for selection (Day 5).
4. Validation by paired simulation from test initial states with
   shared Brownian increments, retaining failed-trajectory mass and
   counting radius excursions (> 8) and negativity (any component
   < 0, the GSDE-specific domain from Proposition 3.4(ii)) as
   domain measures separate from failure (Day 4). Paper-axis proxy:
   H5 compares GSDE vs naive teachers on (a) missing-rate
   degradation of the drift field and (b) negativity under
   coarse-step (dt = 0.5) simulation, both with paired-SE rules.

Run from the repository root:
``python -m src.daily_neural_sde.day07_oh_stable_gsde``
Outputs land in ``outputs/daily_neural_sde/day07/``.
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
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day07.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day07"

DT_FINE = 0.01


# ---------------------------------------------------------------- truth

def true_gamma(x: np.ndarray) -> np.ndarray:
    """Linear gamma (n, 2) for states (n, 2)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    g1 = 0.60 - 0.25 * x[:, 0] - 0.10 * x[:, 1]
    g2 = -0.10 + 0.20 * x[:, 0] - 0.30 * x[:, 1]
    return np.stack([g1, g2], axis=1)


def true_drift(x: np.ndarray) -> np.ndarray:
    """Effective drift f^j = X^j * gamma^j."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return x * true_gamma(x)


def true_g(x: np.ndarray) -> np.ndarray:
    """Effective diffusion levels g^j = sigma^j * X^j."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.stack([0.30 * x[:, 0], 0.25 * x[:, 1]], axis=1)


def generate_data(cfg: dict) -> dict[str, Any]:
    """Simulate trajectories at dt_fine, observe every dt_obs."""
    rng = np.random.default_rng(int(cfg["seed"]))
    n_paths = int(cfg["data"]["n_trajectories"])
    t_end = float(cfg["data"]["t_end"])
    dt_obs = float(cfg["data"]["dt_obs"])
    n_fine = int(round(t_end / DT_FINE))
    obs_every = int(round(dt_obs / DT_FINE))
    n_obs = int(round(t_end / dt_obs)) + 1
    x0 = np.stack([rng.lognormal(np.log(2.0), 0.4, size=n_paths),
                   rng.lognormal(0.0, 0.4, size=n_paths)], axis=1)
    buf = np.zeros((n_paths, n_obs, 2))
    buf[:, 0, :] = x0
    cur = x0.copy()
    obs_idx = 1
    for k in range(1, n_fine + 1):
        dW = rng.standard_normal((n_paths, 2)) * np.sqrt(DT_FINE)
        cur = cur + true_drift(cur) * DT_FINE + true_g(cur) * dW
        if k % obs_every == 0:
            buf[:, obs_idx, :] = cur
            obs_idx += 1
    assert obs_idx == n_obs
    return {"obs": buf, "x0": x0}


def build_queries(obs: np.ndarray, dt_obs: float) -> dict[str, np.ndarray]:
    """Time-major query arrays: cur (n, 2), t (n,)."""
    n_paths, n_obs, _ = obs.shape
    times = np.arange(n_obs) * dt_obs
    cur = obs.transpose(1, 0, 2).reshape(-1, 2)
    t = np.repeat(times, n_paths)
    return {"cur": cur, "t": t}


def build_transitions(obs: np.ndarray, dt_obs: float) -> dict[str, np.ndarray]:
    """Regular one-step transitions with constant dt."""
    n_paths, n_obs, _ = obs.shape
    cur = obs[:, :-1, :].transpose(1, 0, 2).reshape(-1, 2)
    nxt = obs[:, 1:, :].transpose(1, 0, 2).reshape(-1, 2)
    t = np.repeat(np.arange(n_obs - 1) * dt_obs, n_paths)
    dt = np.full(len(cur), dt_obs)
    return {"cur": cur, "nxt": nxt, "t": t, "dt": dt}


def build_thinned_transitions(obs: np.ndarray, dt_obs: float,
                              drop_p: float, seed: int) -> dict[str, np.ndarray]:
    """Transitions joining consecutive kept observations (variable dt).

    Interior observations are dropped independently with probability
    drop_p (seed-fixed); first and last observations are always kept.
    """
    rng = np.random.default_rng(seed)
    n_paths, n_obs, _ = obs.shape
    curs, nxts, ts, dts = [], [], [], []
    for i in range(n_paths):
        keep = np.ones(n_obs, dtype=bool)
        keep[1:-1] = rng.random(n_obs - 2) >= drop_p
        idx = np.flatnonzero(keep)
        for a, b in zip(idx[:-1], idx[1:]):
            curs.append(obs[i, a, :])
            nxts.append(obs[i, b, :])
            ts.append(a * dt_obs)
            dts.append((b - a) * dt_obs)
    return {"cur": np.asarray(curs), "nxt": np.asarray(nxts),
            "t": np.asarray(ts), "dt": np.asarray(dts)}


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


def _train_em(kind: str, drift: MLP, diff: MLP, tr: dict,
              epochs: int, batch: int, lr: float, seed: int) -> dict[str, float]:
    """Variable-dt EM Gaussian NLL for a GSDE or naive teacher.

    kind 'gsde': f = cur * 1.5*tanh(gamma_raw(t, cur)),
    g = cur * (softplus(sigma_raw(t)) + 0.01).
    kind 'naive': f = drift_raw(t, cur),
    g = softplus(diff_raw(t, cur)) + 0.01.
    tr carries u_drift, u_diff, cur, nxt, dt.
    """
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(list(drift.net.parameters())
                           + list(diff.net.parameters()), lr=lr)
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
            if kind == "gsde":
                gamma = 1.5 * torch.tanh(drift.forward_raw(tr["u_drift"][idx]))
                f = cur * gamma
                sigma = torch.nn.functional.softplus(
                    diff.forward_raw(tr["u_diff"][idx])) + 0.01
                g = cur * sigma
            elif kind == "naive":
                f = drift.forward_raw(tr["u_drift"][idx])
                g = torch.nn.functional.softplus(
                    diff.forward_raw(tr["u_diff"][idx])) + 0.01
            else:  # pragma: no cover
                raise ValueError(kind)
            var = g ** 2 * dt + 1e-8
            resid = nxt - cur - f * dt
            loss = torch.mean(0.5 * (np.log(2 * np.pi)
                                     + torch.log(var) + resid ** 2 / var))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        last["loss"] = tot / max(cnt, 1)
    return last


def gsde_drift_predict(model: dict, t: np.ndarray, cur: np.ndarray) -> np.ndarray:
    u = np.concatenate([np.asarray(t)[:, None], cur], axis=1)
    raw = model["gamma"].predict(u)
    return cur * (1.5 * np.tanh(raw))


def gsde_g_predict(model: dict, t: np.ndarray, cur: np.ndarray) -> np.ndarray:
    raw = model["sigma"].predict(np.asarray(t)[:, None])
    sigma = np.logaddexp(0.0, raw) + 0.01
    return cur * sigma


def naive_drift_predict(model: dict, t: np.ndarray, cur: np.ndarray) -> np.ndarray:
    u = np.concatenate([np.asarray(t)[:, None], cur], axis=1)
    return model["drift"].predict(u)


def naive_g_predict(model: dict, t: np.ndarray, cur: np.ndarray) -> np.ndarray:
    u = np.concatenate([np.asarray(t)[:, None], cur], axis=1)
    raw = model["diff"].predict(u)
    return np.logaddexp(0.0, raw) + 0.01


def _prep_tr(tr: dict, kind: str) -> dict:
    """Attach u_drift / u_diff arrays for a transition set."""
    out = dict(tr)
    out["u_drift"] = np.concatenate([tr["t"][:, None], tr["cur"]], axis=1)
    if kind == "gsde":
        out["u_diff"] = tr["t"][:, None]
    else:
        out["u_diff"] = out["u_drift"]
    return out


def build_gsde_teacher(tcfg: dict) -> dict:
    hidden = tuple(tcfg["hidden"])
    return {"gamma": MLP(3, 2, hidden),
            "sigma": MLP(1, 2, tuple(tcfg["sigma_hidden"]))}


def build_naive_teacher(tcfg: dict) -> dict:
    hidden = tuple(tcfg["hidden"])
    return {"drift": MLP(3, 2, hidden), "diff": MLP(3, 2, hidden)}


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

def simulate(drift_fn, diff_fn, x0: np.ndarray, dt: float, n_steps: int,
             increments: np.ndarray, failure_radius: float,
             excursion_radius: float) -> dict[str, Any]:
    """Paired Euler-Maruyama simulation with time-dependent fields.

    drift_fn(t, x) / diff_fn(t, x) take a scalar time and states
    (n, 2). Failures latch (non-finite or radius > failure_radius);
    radius excursions (> excursion_radius) and negativity (any
    component < 0) are counted separately from failure.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    n = len(x)
    sim = np.zeros((n, n_steps + 1, 2))
    sim[:, 0, :] = x
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    ever_excursion = np.zeros(n, dtype=bool)
    ever_negative = np.zeros(n, dtype=bool)
    for k in range(1, n_steps + 1):
        tk = (k - 1) * dt
        active = ~failed
        if active.any():
            f = drift_fn(tk, x[active])
            g = diff_fn(tk, x[active])
            x[active] = x[active] + f * dt + g * increments[active, k - 1] * np.sqrt(dt)
        radii = np.linalg.norm(x, axis=1)
        ever_excursion = ever_excursion | ((radii > excursion_radius) & active)
        ever_negative = ever_negative | ((x < 0).any(axis=1) & active)
        bad_nonfinite = ~np.isfinite(x).all(axis=1)
        bad_radius = radii > failure_radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        x[failed] = np.nan
        sim[:, k, :] = x
    return {"states": sim, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion_rate": float(np.mean(ever_excursion)),
            "negativity_rate": float(np.mean(ever_negative))}


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


def main() -> None:
    import torch
    t_start = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    dt_obs = float(cfg["data"]["dt_obs"])
    t_end = float(cfg["data"]["t_end"])
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    print("generating data ...", flush=True)
    data = generate_data(cfg)
    obs, x0 = data["obs"], data["x0"]
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    idx_tr, idx_va, idx_te = _split_indices(len(obs), cfg, rng)
    splits = {}
    for name, idx in (("train", idx_tr), ("validation", idx_va), ("test", idx_te)):
        splits[name] = {"q": build_queries(obs[idx], dt_obs),
                        "tr": build_transitions(obs[idx], dt_obs),
                        "obs": obs[idx], "x0": x0[idx]}
    print("state stats train: X1 mean %.3f std %.3f range [%.2f, %.2f]; "
          "X2 mean %.3f std %.3f range [%.2f, %.2f]" % (
              splits["train"]["q"]["cur"][:, 0].mean(),
              splits["train"]["q"]["cur"][:, 0].std(),
              splits["train"]["q"]["cur"][:, 0].min(),
              splits["train"]["q"]["cur"][:, 0].max(),
              splits["train"]["q"]["cur"][:, 1].mean(),
              splits["train"]["q"]["cur"][:, 1].std(),
              splits["train"]["q"]["cur"][:, 1].min(),
              splits["train"]["q"]["cur"][:, 1].max()), flush=True)

    # --- collinearity gate (Day 1) on train query states
    cur_tr = splits["train"]["q"]["cur"]
    corr = float(np.corrcoef(cur_tr.T)[0, 1])
    vif_lin = vif_scores(cur_tr)
    lib2 = np.stack([cur_tr[:, 0], cur_tr[:, 1], cur_tr[:, 0] ** 2,
                     cur_tr[:, 0] * cur_tr[:, 1], cur_tr[:, 1] ** 2], axis=1)
    vif_deg2 = vif_scores(lib2)
    print("collinearity: corr(X1,X2) %.3f, VIF linear max %.2f, "
          "VIF degree-2 max %.1f" % (corr, vif_lin.max(), vif_deg2.max()), flush=True)

    # --- teachers: GSDE and naive, dense and 50%-thinned
    tcfg = cfg["teacher"]
    em_cfg = tcfg["em_main"]
    tr_dense = _prep_tr(splits["train"]["tr"], "gsde")
    tr_thin_gsde = _prep_tr(build_thinned_transitions(
        splits["train"]["obs"], dt_obs, float(cfg["data"]["missing_rate_training"]), 123),
        "gsde")
    tr_thin_naive = _prep_tr(build_thinned_transitions(
        splits["train"]["obs"], dt_obs, float(cfg["data"]["missing_rate_training"]), 123),
        "naive")
    tr_dense_naive = _prep_tr(splits["train"]["tr"], "naive")
    print("transitions: dense %d, thinned %d" % (
        len(tr_dense["cur"]), len(tr_thin_gsde["cur"])), flush=True)

    def train_gsde(tr):
        m = build_gsde_teacher(tcfg)
        m["gamma"].fit_standardizer(tr["u_drift"])
        m["sigma"].fit_standardizer(tr["u_diff"])
        _train_em("gsde", m["gamma"], m["sigma"], tr,
                  int(em_cfg["epochs"]), int(em_cfg["batch"]),
                  float(em_cfg["lr"]), 0)
        return m

    def train_naive(tr):
        m = build_naive_teacher(tcfg)
        m["drift"].fit_standardizer(tr["u_drift"])
        m["diff"].fit_standardizer(tr["u_diff"])
        _train_em("naive", m["drift"], m["diff"], tr,
                  int(em_cfg["epochs"]), int(em_cfg["batch"]),
                  float(em_cfg["lr"]), 0)
        return m

    print("training GSDE dense ...", flush=True)
    gsde_dense = train_gsde(tr_dense)
    print("training GSDE thinned (distilled teacher) ...", flush=True)
    gsde = train_gsde(tr_thin_gsde)
    print("training naive dense ...", flush=True)
    naive_dense = train_naive(tr_dense_naive)
    print("training naive thinned ...", flush=True)
    naive = train_naive(tr_thin_naive)

    # --- H6 pair: GSDE EM coarse (dt = 0.5) vs count-matched fine
    coarse_obs = splits["train"]["obs"][:, ::10, :]
    tr_coarse = _prep_tr(build_transitions(coarse_obs, 0.5), "gsde")
    rng_sub = np.random.default_rng(7)
    sub = rng_sub.choice(len(tr_dense["cur"]), size=len(tr_coarse["cur"]),
                         replace=False)
    tr_fine_sub = {k: v[sub] for k, v in tr_dense.items()}
    print("training GSDE coarse / fine-sub ...", flush=True)
    gsde_coarse = train_gsde(tr_coarse)
    gsde_fine_sub = train_gsde(tr_fine_sub)

    # --- test query evaluations
    te = splits["test"]["q"]
    va = splits["validation"]["q"]
    tr_q = splits["train"]["q"]
    truth_drift_te = true_drift(te["cur"])
    truth_g_te = true_g(te["cur"])
    truth_drift_va = true_drift(va["cur"])

    def gsde_fields(m):
        return (gsde_drift_predict(m, te["t"], te["cur"]),
                gsde_g_predict(m, te["t"], te["cur"]))

    def naive_fields(m):
        return (naive_drift_predict(m, te["t"], te["cur"]),
                naive_g_predict(m, te["t"], te["cur"]))

    gd_te, gg_te = gsde_fields(gsde)
    gdd_te, ggd_te = gsde_fields(gsde_dense)
    nd_te, ng_te = naive_fields(naive)
    ndd_te, ngd_te = naive_fields(naive_dense)
    gc_te, ggc_te = gsde_fields(gsde_coarse)
    gf_te, ggf_te = gsde_fields(gsde_fine_sub)

    def sigma_vec(g):
        return g ** 2

    teachers_table = {}
    for name, f, g in (("gsde_thinned", gd_te, gg_te),
                       ("gsde_dense", gdd_te, ggd_te),
                       ("naive_thinned", nd_te, ng_te),
                       ("naive_dense", ndd_te, ngd_te),
                       ("gsde_coarse_dt0.5", gc_te, ggc_te),
                       ("gsde_fine_sub", gf_te, ggf_te)):
        teachers_table[name] = {
            "drift_vs_truth": _field_table(f, truth_drift_te),
            "g_vs_truth": _field_table(g, truth_g_te),
            "sigma_vs_truth": _field_table(sigma_vec(g), sigma_vec(truth_g_te)),
        }
    print("teachers vs truth (drift NRMSE): " + ", ".join(
        f"{k}={v['drift_vs_truth']['nrmse']:.4f}"
        for k, v in teachers_table.items()), flush=True)

    # --- H5(a): missing-rate degradation, GSDE vs naive (paired)
    def sq_err(f):
        return np.mean((f - truth_drift_te) ** 2, axis=1)
    sq_gsde_thin, sq_gsde_dense = sq_err(gd_te), sq_err(gdd_te)
    sq_naive_thin, sq_naive_dense = sq_err(nd_te), sq_err(ndd_te)
    degr_gsde = float(np.mean(sq_gsde_thin) - np.mean(sq_gsde_dense))
    degr_naive = float(np.mean(sq_naive_thin) - np.mean(sq_naive_dense))
    d_degr = (sq_naive_thin - sq_naive_dense) - (sq_gsde_thin - sq_gsde_dense)
    h5a = {"degradation_gsde": degr_gsde, "degradation_naive": degr_naive,
           "difference": float(np.mean(d_degr)),
           "se": float(np.std(d_degr, ddof=1) / np.sqrt(len(d_degr))),
           "pass": bool(np.mean(d_degr) > 0
                        and np.mean(d_degr) > 2 * (np.std(d_degr, ddof=1)
                                                   / np.sqrt(len(d_degr))))}
    print("H5(a) degradations: gsde %.5f naive %.5f diff %.5f SE %.5f" % (
        degr_gsde, degr_naive, h5a["difference"], h5a["se"]), flush=True)

    # --- H6 discretization diagnostic (regime-split by radius)
    radii_te = np.linalg.norm(te["cur"], axis=1)
    small = radii_te < 2.0
    large = radii_te > 3.0

    def regime_nrmse(f, ref, mask):
        return nrmse(ref[mask], f[mask]) if mask.any() else float("nan")
    h6 = {
        "nrmse_gsde_fine_sub": nrmse(truth_drift_te, gf_te),
        "nrmse_gsde_coarse": nrmse(truth_drift_te, gc_te),
        "nrmse_gsde_fine_sub_small": regime_nrmse(gf_te, truth_drift_te, small),
        "nrmse_gsde_coarse_small": regime_nrmse(gc_te, truth_drift_te, small),
        "nrmse_gsde_fine_sub_large": regime_nrmse(gf_te, truth_drift_te, large),
        "nrmse_gsde_coarse_large": regime_nrmse(gc_te, truth_drift_te, large),
        "n_small": int(small.sum()), "n_large": int(large.sum()),
        "sigma_nrmse_gsde_fine_sub": nrmse(sigma_vec(truth_g_te), sigma_vec(ggf_te)),
        "sigma_nrmse_gsde_coarse": nrmse(sigma_vec(truth_g_te), sigma_vec(ggc_te)),
    }
    h6["degradation"] = h6["nrmse_gsde_coarse"] - h6["nrmse_gsde_fine_sub"]
    h6["degradation_small"] = (h6["nrmse_gsde_coarse_small"]
                               - h6["nrmse_gsde_fine_sub_small"])
    h6["degradation_large"] = (h6["nrmse_gsde_coarse_large"]
                               - h6["nrmse_gsde_fine_sub_large"])
    h6["regime_gap"] = h6["degradation_large"] - h6["degradation_small"]
    h6["pass_a"] = bool(h6["degradation"] >= 0.03)
    h6["pass_b"] = bool(h6["regime_gap"] >= 0.02)
    h6["pass"] = bool(h6["pass_a"] and h6["pass_b"])
    print("H6: degradation %.4f small %.4f large %.4f gap %.4f" % (
        h6["degradation"], h6["degradation_small"], h6["degradation_large"],
        h6["regime_gap"]), flush=True)

    # --- SINDy distillation of the GSDE-thinned teacher
    dist = cfg["distillation"]
    degrees = dist["drift_degrees"]
    thresholds = dist["thresholds"]
    ridge = float(dist["ridge"])
    gsde_drift_tr = gsde_drift_predict(gsde, tr_q["t"], tr_q["cur"])
    gsde_drift_va = gsde_drift_predict(gsde, va["t"], va["cur"])
    gsde_g_tr = gsde_g_predict(gsde, tr_q["t"], tr_q["cur"])
    gsde_g_va = gsde_g_predict(gsde, va["t"], va["cur"])
    drift_field, drift_val, drift_meta = fit_field(
        tr_q["cur"], gsde_drift_tr, va["cur"], gsde_drift_va,
        degrees, thresholds, ridge)
    # Diffusion per component on (t, own state).
    diff_fields, diff_metas, diff_vals = [], [], []
    for j in range(2):
        xj_tr = np.stack([tr_q["t"], tr_q["cur"][:, j]], axis=1)
        xj_va = np.stack([va["t"], va["cur"][:, j]], axis=1)
        fj, vj, mj = fit_field(xj_tr, gsde_g_tr[:, [j]], xj_va,
                               gsde_g_va[:, [j]],
                               dist["diffusion_degrees"], thresholds, ridge)
        diff_fields.append(fj)
        diff_vals.append(vj)
        diff_metas.append(mj)
    sym_drift_te = drift_field.predict(te["cur"])
    sym_g_te = np.stack([
        diff_fields[0].predict(np.stack([te["t"], te["cur"][:, 0]], axis=1))[:, 0],
        diff_fields[1].predict(np.stack([te["t"], te["cur"][:, 1]], axis=1))[:, 0],
    ], axis=1)
    # Oracles (same procedure on truth fields).
    truth_drift_tr = true_drift(tr_q["cur"])
    oracle_drift, _, oracle_drift_meta = fit_field(
        tr_q["cur"], truth_drift_tr, va["cur"], truth_drift_va,
        degrees, thresholds, ridge)
    oracle_drift_te = oracle_drift.predict(te["cur"])
    truth_g_tr = true_g(tr_q["cur"])
    truth_g_va = true_g(va["cur"])
    oracle_diff_fields = []
    for j in range(2):
        xj_tr = np.stack([tr_q["t"], tr_q["cur"][:, j]], axis=1)
        xj_va = np.stack([va["t"], va["cur"][:, j]], axis=1)
        fj, _, _ = fit_field(xj_tr, truth_g_tr[:, [j]], xj_va,
                             truth_g_va[:, [j]],
                             dist["diffusion_degrees"], thresholds, ridge)
        oracle_diff_fields.append(fj)
    oracle_g_te = np.stack([
        oracle_diff_fields[0].predict(np.stack([te["t"], te["cur"][:, 0]], axis=1))[:, 0],
        oracle_diff_fields[1].predict(np.stack([te["t"], te["cur"][:, 1]], axis=1))[:, 0],
    ], axis=1)
    # Full-library diffusion diagnostic (spurious cross-dependence).
    full_diff_fields = []
    for j in range(2):
        xj_tr = np.stack([tr_q["t"], tr_q["cur"][:, 0], tr_q["cur"][:, 1]], axis=1)
        xj_va = np.stack([va["t"], va["cur"][:, 0], va["cur"][:, 1]], axis=1)
        fj, _, _ = fit_field(xj_tr, gsde_g_tr[:, [j]], xj_va,
                             gsde_g_va[:, [j]],
                             dist["diffusion_degrees"], thresholds, ridge)
        full_diff_fields.append(fj)
    full_exp = [expand_to_original(f)[0] for f in full_diff_fields]
    cross_max = []
    for j, exp in enumerate(full_exp):
        other = 1 - j  # variable index in (t, X1, X2): X1 is 1, X2 is 2
        cross_idx = 1 if j == 1 else 2
        vals = [abs(c) for e, c in exp.items() if e[cross_idx] > 0]
        cross_max.append(float(max(vals)) if vals else 0.0)

    dist_table = {
        "drift_selection": drift_meta, "drift_validation_nrmse": drift_val,
        "diffusion_selection": diff_metas, "diffusion_validation_nrmse": diff_vals,
        "drift_vs_teacher": _field_table(sym_drift_te, gd_te),
        "g_vs_teacher": _field_table(sym_g_te, gg_te),
        "sigma_vs_teacher": _field_table(sigma_vec(sym_g_te), sigma_vec(gg_te)),
        "drift_vs_truth": _field_table(sym_drift_te, truth_drift_te),
        "g_vs_truth": _field_table(sym_g_te, truth_g_te),
        "sigma_vs_truth": _field_table(sigma_vec(sym_g_te), sigma_vec(truth_g_te)),
        "oracle_drift_vs_truth": _field_table(oracle_drift_te, truth_drift_te),
        "oracle_g_vs_truth": _field_table(oracle_g_te, truth_g_te),
        "oracle_drift_selection": oracle_drift_meta,
        "full_library_diffusion_cross_max": cross_max,
    }
    print("distillation: drift vs teacher %.4f, g vs teacher %.4f, "
          "drift vs truth %.4f, oracle drift vs truth %.4f" % (
              dist_table["drift_vs_teacher"]["nrmse"],
              dist_table["g_vs_teacher"]["nrmse"],
              dist_table["drift_vs_truth"]["nrmse"],
              dist_table["oracle_drift_vs_truth"]["nrmse"]), flush=True)

    # --- H3c linear coefficients (expanded symbolic drift)
    drift_exp = expand_to_original(drift_field)
    truth_lin = [{(1, 0): 0.60, (0, 1): 0.0}, {(1, 0): 0.0, (0, 1): -0.10}]
    lin_errors = {}
    for j in range(2):
        for exp, tv in truth_lin[j].items():
            lin_errors[f"f{j + 1}_{exp}"] = float(drift_exp[j].get(exp, 0.0) - tv)
    h3c = {"linear_coefficient_errors": lin_errors,
           "max_abs_error": float(max(abs(v) for v in lin_errors.values())),
           "pass": bool(max(abs(v) for v in lin_errors.values()) <= 0.20)}
    quad_truth = [{(2, 0): -0.25, (1, 1): -0.10, (0, 2): 0.0},
                  {(2, 0): 0.0, (1, 1): 0.20, (0, 2): -0.30}]
    quad_errors = {}
    for j in range(2):
        for exp, tv in quad_truth[j].items():
            quad_errors[f"f{j + 1}_{exp}"] = float(drift_exp[j].get(exp, 0.0) - tv)
    h3c["quadratic_coefficient_errors_report_only"] = quad_errors
    print("H3c linear max abs error %.4f" % h3c["max_abs_error"], flush=True)

    # --- off-manifold grid diagnostic (Day 2, reporting only)
    xs = np.array([0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    ys = np.array([0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid_t = np.full(len(grid), 5.0)
    grid_truth_d = true_drift(grid)
    grid_teach_d = gsde_drift_predict(gsde, grid_t, grid)
    grid_sym_d = drift_field.predict(grid)
    grid_truth_g = true_g(grid)
    grid_teach_g = gsde_g_predict(gsde, grid_t, grid)
    grid_sym_g = np.stack([
        diff_fields[0].predict(np.stack([grid_t, grid[:, 0]], axis=1))[:, 0],
        diff_fields[1].predict(np.stack([grid_t, grid[:, 1]], axis=1))[:, 0],
    ], axis=1)
    corner = (grid[:, 0] < 0.5) & (grid[:, 1] < 0.5)
    grid_diag = {
        "n_points": int(len(grid)), "n_corner": int(corner.sum()),
        "drift_symbolic_vs_teacher": _field_table(grid_sym_d, grid_teach_d),
        "drift_symbolic_vs_truth": _field_table(grid_sym_d, grid_truth_d),
        "drift_teacher_vs_truth": _field_table(grid_teach_d, grid_truth_d),
        "g_symbolic_vs_teacher": _field_table(grid_sym_g, grid_teach_g),
        "g_symbolic_vs_truth": _field_table(grid_sym_g, grid_truth_g),
        "drift_symbolic_vs_teacher_corner": _field_table(
            grid_sym_d[corner], grid_teach_d[corner]),
        "drift_symbolic_vs_truth_corner": _field_table(
            grid_sym_d[corner], grid_truth_d[corner]),
        "drift_teacher_vs_truth_corner": _field_table(
            grid_teach_d[corner], grid_truth_d[corner]),
    }
    print("grid: sym-vs-teacher drift %.4f sym-vs-truth %.4f "
          "teacher-vs-truth %.4f" % (
              grid_diag["drift_symbolic_vs_teacher"]["nrmse"],
              grid_diag["drift_symbolic_vs_truth"]["nrmse"],
              grid_diag["drift_teacher_vs_truth"]["nrmse"]), flush=True)

    # --- teacher t-dependence diagnostic (truth is time-homogeneous)
    rng_t = np.random.default_rng(5)
    sample_idx = rng_t.choice(len(te["cur"]), size=20, replace=False)
    sample_states = te["cur"][sample_idx]
    t_vals = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
    field_by_t = np.stack([gsde_drift_predict(gsde, np.full(20, tv), sample_states)
                           for tv in t_vals], axis=0)
    t_std = float(np.mean(np.std(field_by_t, axis=0)))
    t_diag = {"mean_std_across_t": t_std,
              "field_rms": float(np.sqrt(np.mean(field_by_t ** 2))),
              "ratio": float(t_std / max(np.sqrt(np.mean(field_by_t ** 2)), 1e-12))}

    # --- paired simulation (dt = 0.05) and coarse simulation (dt = 0.5)
    sim_cfg = cfg["simulation"]
    n_te = len(idx_te)
    rng_sim = np.random.default_rng(int(sim_cfg["seed"]))
    n_steps = int(round(t_end / dt_obs))
    inc = rng_sim.standard_normal((n_te, n_steps, 2))
    x0_te = splits["test"]["x0"]

    def gsde_fns(m):
        return (lambda tk, x: gsde_drift_predict(m, np.full(len(x), tk), x),
                lambda tk, x: gsde_g_predict(m, np.full(len(x), tk), x))

    def naive_fns(m):
        return (lambda tk, x: naive_drift_predict(m, np.full(len(x), tk), x),
                lambda tk, x: naive_g_predict(m, np.full(len(x), tk), x))

    def sym_fns():
        def f_fn(tk, x):
            return drift_field.predict(x)
        def g_fn(tk, x):
            return np.stack([
                diff_fields[0].predict(np.stack(
                    [np.full(len(x), tk), x[:, 0]], axis=1))[:, 0],
                diff_fields[1].predict(np.stack(
                    [np.full(len(x), tk), x[:, 1]], axis=1))[:, 0],
            ], axis=1)
        return f_fn, g_fn

    truth_f, truth_g = (lambda tk, x: true_drift(x),
                        lambda tk, x: true_g(x))
    truth_sim = simulate(truth_f, truth_g, x0_te, dt_obs, n_steps, inc,
                         float(sim_cfg["failure_radius"]),
                         float(sim_cfg["domain_excursion_radius"]))
    gf, gg = gsde_fns(gsde)
    teacher_sim = simulate(gf, gg, x0_te, dt_obs, n_steps, inc,
                           float(sim_cfg["failure_radius"]),
                           float(sim_cfg["domain_excursion_radius"]))
    sf, sg = sym_fns()
    sym_sim = simulate(sf, sg, x0_te, dt_obs, n_steps, inc,
                       float(sim_cfg["failure_radius"]),
                       float(sim_cfg["domain_excursion_radius"]))
    nf, ng = naive_fns(naive)
    naive_sim = simulate(nf, ng, x0_te, dt_obs, n_steps, inc,
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
                    "excursion_rate": teacher_sim["excursion_rate"],
                    "negativity_rate": teacher_sim["negativity_rate"]},
        "symbolic": {"failure_rate": sym_sim["failure_rate"],
                     "excursion_rate": sym_sim["excursion_rate"],
                     "negativity_rate": sym_sim["negativity_rate"]},
        "truth": {"failure_rate": truth_sim["failure_rate"],
                  "excursion_rate": truth_sim["excursion_rate"],
                  "negativity_rate": truth_sim["negativity_rate"]},
        "naive_thinned": {"failure_rate": naive_sim["failure_rate"],
                          "excursion_rate": naive_sim["excursion_rate"],
                          "negativity_rate": naive_sim["negativity_rate"]},
        "paired_rmse_median": float(np.median(paired_rmse)) if len(paired_rmse) else None,
        "paired_rmse_mean": float(np.mean(paired_rmse)) if len(paired_rmse) else None,
        "n_paired": int(len(paired_rmse)),
    }
    print("simulation: teacher fail %.4f exc %.4f neg %.4f; symbolic "
          "fail %.4f exc %.4f neg %.4f; naive neg %.4f; paired RMSE "
          "median %s" % (
              sim_table["teacher"]["failure_rate"],
              sim_table["teacher"]["excursion_rate"],
              sim_table["teacher"]["negativity_rate"],
              sim_table["symbolic"]["failure_rate"],
              sim_table["symbolic"]["excursion_rate"],
              sim_table["symbolic"]["negativity_rate"],
              sim_table["naive_thinned"]["negativity_rate"],
              sim_table["paired_rmse_median"]), flush=True)

    # --- H5(b): coarse-step negativity, GSDE-thinned vs naive-thinned
    rng_c = np.random.default_rng(int(sim_cfg["seed"]) + 1)
    inc_c = rng_c.standard_normal((n_te, 20, 2))
    gsde_c = simulate(gf, gg, x0_te, 0.5, 20, inc_c,
                      float(sim_cfg["failure_radius"]),
                      float(sim_cfg["domain_excursion_radius"]))
    naive_c = simulate(nf, ng, x0_te, 0.5, 20, inc_c,
                       float(sim_cfg["failure_radius"]),
                       float(sim_cfg["domain_excursion_radius"]))
    # Per-trajectory negativity flags are not returned by simulate, so
    # recompute them from the stored states (NaN after failure counts
    # as not-negative; failures are reported alongside).
    def neg_flags(sim):
        states = sim["states"]
        return np.array([bool(np.any(states[i][np.isfinite(states[i]).all(axis=1)] < 0))
                         for i in range(len(states))])
    neg_gsde_c = neg_flags(gsde_c)
    neg_naive_c = neg_flags(naive_c)
    d_neg = (neg_naive_c.astype(float) - neg_gsde_c.astype(float))
    h5b = {"negativity_gsde_coarse": float(np.mean(neg_gsde_c)),
           "negativity_naive_coarse": float(np.mean(neg_naive_c)),
           "difference": float(np.mean(d_neg)),
           "se": float(np.std(d_neg, ddof=1) / np.sqrt(len(d_neg))),
           "failure_gsde_coarse": gsde_c["failure_rate"],
           "failure_naive_coarse": naive_c["failure_rate"],
           "pass": bool(np.mean(d_neg) > 0
                        and np.mean(d_neg) > 2 * (np.std(d_neg, ddof=1)
                                                  / np.sqrt(len(d_neg))))}
    h5 = {"degradation": h5a, "coarse_negativity": h5b,
          "pass": bool(h5a["pass"] and h5b["pass"])}
    print("H5(b) coarse negativity: gsde %.4f naive %.4f diff %.4f SE %.4f" % (
        h5b["negativity_gsde_coarse"], h5b["negativity_naive_coarse"],
        h5b["difference"], h5b["se"]), flush=True)

    # --- H7 downstream: one-step X^1 MSE, symbolic vs teacher
    te_tr = splits["test"]["tr"]
    tea_pred = te_tr["cur"][:, [0]] + gsde_drift_predict(
        gsde, te_tr["t"], te_tr["cur"])[:, [0]] * dt_obs
    sym_pred = te_tr["cur"][:, [0]] + drift_field.predict(
        te_tr["cur"])[:, [0]] * dt_obs
    mse_tea = float(np.mean((tea_pred - te_tr["nxt"][:, [0]]) ** 2))
    mse_sym = float(np.mean((sym_pred - te_tr["nxt"][:, [0]]) ** 2))
    d = (sym_pred - te_tr["nxt"][:, [0]]) ** 2 - (tea_pred - te_tr["nxt"][:, [0]]) ** 2
    h7 = {"mse_teacher": mse_tea, "mse_symbolic": mse_sym,
          "abs_diff": abs(mse_sym - mse_tea),
          "paired_se": float(np.std(d.ravel(), ddof=1) / np.sqrt(d.size)),
          "established": bool(float(np.std(d.ravel(), ddof=1) / np.sqrt(d.size)) < 0.0025),
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
        "H3c_linear_coefficients": h3c["pass"],
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
            <= hyp["H4_simulation"]["domain_excursion_symbolic_minus_teacher_max"]
            and (sim_table["symbolic"]["negativity_rate"]
                 - sim_table["teacher"]["negativity_rate"])
            <= hyp["H4_simulation"]["negativity_symbolic_minus_teacher_max"]),
        "H5_paper_axis_gsde_vs_naive": h5["pass"],
        "H6_discretization_bias": h6["pass"],
        "H7_downstream_one_step": h7["pass"],
    }
    print("verdicts:", verdicts, flush=True)

    # --- TeX symbolic SDE (original coordinates)
    diff_exp = [expand_to_original(f)[0] for f in diff_fields]
    drift_tex = tex_equations(drift_exp, ("\\hat f_x", "\\hat f_y"), ("x", "y"))
    diff_tex_1 = tex_equations([diff_exp[0]], ("\\hat g^1",), ("t", "x"))
    diff_tex_2 = tex_equations([diff_exp[1]], ("\\hat g^2",), ("t", "y"))
    tex_doc = ("%% Day 7 symbolic SDE (Oh et al. 2024 Stable Neural SDEs, "
               "Geometric SDE; distilled GSDE teacher; x = X^1, y = X^2)\n"
               + drift_tex + "\n" + diff_tex_1 + "\n" + diff_tex_2 + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "symbolic_sde.tex").write_text(tex_doc)
    drift_field.save(OUT_DIR / "symbolic_drift.json")
    diff_fields[0].save(OUT_DIR / "symbolic_diffusion_g1.json")
    diff_fields[1].save(OUT_DIR / "symbolic_diffusion_g2.json")
    teacher_states = {
        "gsde_thinned": {"gamma": gsde["gamma"].state(),
                         "sigma": gsde["sigma"].state()},
        "naive_thinned": {"drift": naive["drift"].state(),
                          "diff": naive["diff"].state()},
        "gsde_dense": {"gamma": gsde_dense["gamma"].state(),
                       "sigma": gsde_dense["sigma"].state()},
        "naive_dense": {"drift": naive_dense["drift"].state(),
                        "diff": naive_dense["diff"].state()},
    }
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(teacher_states))
    results = {
        "day": 7, "date": cfg["date"], "paper": cfg["paper"],
        "split_sizes": {"train": int(len(idx_tr)), "validation": int(len(idx_va)),
                        "test": int(len(idx_te))},
        "state_stats_train": {
            "x1": {"mean": float(splits["train"]["q"]["cur"][:, 0].mean()),
                   "std": float(splits["train"]["q"]["cur"][:, 0].std())},
            "x2": {"mean": float(splits["train"]["q"]["cur"][:, 1].mean()),
                   "std": float(splits["train"]["q"]["cur"][:, 1].std())}},
        "collinearity": {
            "corr_x1_x2": corr,
            "vif_linear": vif_lin.tolist(),
            "vif_degree2": vif_deg2.tolist()},
        "teachers_vs_truth": teachers_table,
        "distillation": dist_table,
        "H3c": h3c,
        "grid": grid_diag,
        "teacher_t_dependence": t_diag,
        "H5": h5, "H6": h6, "H7": h7,
        "simulation": sim_table,
        "verdicts": verdicts,
        "runtime_seconds": round(time.time() - t_start, 1),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
