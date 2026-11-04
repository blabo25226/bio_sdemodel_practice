"""Day 10: Song et al. (2021) Score-Based Generative Modeling through SDEs
-> polynomial SINDy of the reverse-time SDE.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day10.json``):

1. Ground truth: a 1D bimodal target p0 = 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)
   under the paper's VP-SDE, dx = -0.5 beta(t) x dt + sqrt(beta(t)) dW with
   beta(t) = 0.1 + 19.9 t, t in [0, 1]. The perturbation kernel is analytic,
   so the perturbed score and the truth reverse drift
   f_rev(x, tau) = 0.5 beta(s) x + beta(s) score(x, s), s = 1 - tau,
   are known exactly. This is a declared cut: images are replaced by a 1D
   mixture whose scores are analytic, so teacher-vs-truth comparison is
   exact; nothing here speaks about the paper's image benchmarks.
2. Teacher: an epsilon-network score model trained by denoising score
   matching (DSM) on fresh (x0, t, z) draws, the paper's training
   objective in its VP-SDE form. The teacher reverse drift plugs the
   learned score into Anderson's reverse formula; the teacher diffusion
   is the paper's known schedule sqrt(beta(s)), not a learned function.
3. Distillation of the teacher reverse drift and diffusion into
   polynomial SINDy in (x, tau): direct function queries on
   sample-disjoint splits, validation-only selection, test once. A
   collinearity gate at the maximum selectable degree (3) precedes
   fitting (Day 1 / Day 7), with the constant column excluded (Day 8).
   No coefficient hypothesis is frozen against truth: the truth reverse
   drift is a rational mixture-score function, not a polynomial.
   A fixed (x, tau) grid is the Day 2 off-manifold diagnostic, reported
   only, never used for selection (Day 5), with the Day 9 three-way
   comparison (symbolic vs teacher, teacher vs truth, symbolic vs truth).
4. Validation by paired reverse sampling from the prior N(0, 1) with
   shared Brownian increments, retaining failed-sample mass and counting
   |X| excursions (> 8) as a domain measure separate from failure
   (Day 4). Paper axis: H5 compares reverse SDE sampling (teacher and
   symbolic) against a no-score baseline (plain OU reversal) and reports
   the probability-flow ODE (teacher score, deterministic) as a
   diagnostic. The Day 1 discretization diagnostic (H6) is the reverse
   solver step (0.002 vs 0.05, coarse increments exactly aggregating the
   fine increments), regime-split into near-linear (tau < 0.3) vs
   nonlinear (tau > 0.7) reverse time with per-step SNR quantified.

Run from the repository root:
``python -m src.daily_neural_sde.day10_song_score``

Outputs land in ``outputs/daily_neural_sde/day10/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, expand_to_original, fit_polynomial_field,
                     monomial_powers, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day10.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day10"

BETA_MIN, BETA_MAX = 0.1, 20.0
MU, SIG0 = 2.0, 0.3


# ---------------------------------------------------------------- truth

def beta(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return BETA_MIN + (BETA_MAX - BETA_MIN) * t


def alpha(t: np.ndarray) -> np.ndarray:
    """VP-SDE mean coefficient: exp(-0.5 * int_0^t beta)."""
    t = np.asarray(t, dtype=np.float64)
    return np.exp(-0.5 * (BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2))


def pert_var(t: np.ndarray) -> np.ndarray:
    """Variance of the perturbed mixture: 1 - (1 - SIG0^2) alpha^2."""
    a = alpha(t)
    return 1.0 - (1.0 - SIG0 ** 2) * a ** 2


def true_score(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Analytic score of the perturbed bimodal mixture at (x, t)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    a = alpha(t)
    v = pert_var(t)
    mu = MU * a
    dens = np.exp(-0.5 * (x - mu) ** 2 / v) + np.exp(-0.5 * (x + mu) ** 2 / v)
    numer = (np.exp(-0.5 * (x - mu) ** 2 / v) * (mu - x)
             + np.exp(-0.5 * (x + mu) ** 2 / v) * (-mu - x))
    return numer / (v * dens)


def true_reverse_drift(xtau: np.ndarray) -> np.ndarray:
    """Truth reverse drift f_rev(x, tau); input columns (x, tau)."""
    xtau = np.atleast_2d(np.asarray(xtau, dtype=np.float64))
    x, tau = xtau[:, 0], xtau[:, 1]
    s = 1.0 - tau
    b = beta(s)
    return (0.5 * b * x + b * true_score(x, s))[:, None]


def true_reverse_g(xtau: np.ndarray) -> np.ndarray:
    xtau = np.atleast_2d(np.asarray(xtau, dtype=np.float64))
    return np.sqrt(beta(1.0 - xtau[:, 1]))[:, None]


def forward_sample(rng: np.random.Generator, x0: np.ndarray,
                   t: np.ndarray) -> np.ndarray:
    """Draw x_t from the VP perturbation kernel given x0 and t."""
    a = alpha(t)
    return a * x0 + np.sqrt(1.0 - a ** 2) * rng.standard_normal(len(x0))


# ---------------------------------------------------------------- data

def generate_queries(cfg: dict) -> dict[str, Any]:
    """Split-disjoint (x_t, tau) query states from mixture samples."""
    rng = np.random.default_rng(int(cfg["seed"]))
    n_train = int(cfg["split"]["train"])
    n_val = int(cfg["split"]["validation"])
    n_test = int(cfg["split"]["test"])
    total = n_train + n_val + n_test
    signs = rng.choice([-1.0, 1.0], size=total)
    x0 = signs * MU + SIG0 * rng.standard_normal(total)
    k = int(cfg["data"]["queries_per_sample"])
    t_lo, t_hi = (float(v) for v in cfg["data"]["t_query_range"])
    out: dict[str, Any] = {"x0": x0}
    bounds = [("train", 0, n_train), ("validation", n_train, n_train + n_val),
              ("test", n_train + n_val, total)]
    for name, lo, hi in bounds:
        xs = np.repeat(x0[lo:hi], k)
        t = rng.uniform(t_lo, t_hi, size=len(xs))
        xt = forward_sample(rng, xs, t)
        out[name] = np.stack([xt, 1.0 - t], axis=1)
    return out


# ---------------------------------------------------------------- teacher

def train_score_teacher(cfg: dict, x0_train: np.ndarray) -> dict[str, Any]:
    """Train the epsilon-network by denoising score matching (torch)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(int(cfg["seed"]))
    device = "cpu"
    net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64),
                        nn.Tanh(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(int(cfg["seed"]) + 77)
    n = len(x0_train)
    steps, batch = 4000, 512
    t_lo, t_hi = (float(v) for v in cfg["teacher"]["t_train_range"])
    losses: list[float] = []
    for _ in range(steps):
        idx = rng.integers(0, n, size=batch)
        x0 = x0_train[idx]
        t = rng.uniform(t_lo, t_hi, size=batch)
        z = rng.standard_normal(batch)
        a = alpha(t)
        xt = a * x0 + np.sqrt(1.0 - a ** 2) * z
        inp = torch.tensor(np.stack([xt / 2.5, t], axis=1),
                           dtype=torch.float32)
        pred = net(inp)[:, 0]
        loss = torch.mean((pred - torch.tensor(z, dtype=torch.float32)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    weights = {k: v.detach().numpy().tolist()
               for k, v in net.state_dict().items()}
    return {"weights": weights, "final_dsm_loss": float(np.mean(losses[-200:])),
            "n_steps": steps}

    # NOTE: evaluation helpers below are module-level (numpy) so tests can
    # use them without torch.


_NET_CACHE: dict[str, Any] = {}


def _get_net(weights: dict):
    """Cache the torch score net keyed by the weights object's identity."""
    import torch
    import torch.nn as nn

    if _NET_CACHE.get("weights") is not weights:
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64),
                            nn.Tanh(), nn.Linear(64, 1))
        net.load_state_dict({k: torch.tensor(np.asarray(v, dtype=np.float32))
                             for k, v in weights.items()})
        net.eval()
        _NET_CACHE["weights"] = weights
        _NET_CACHE["net"] = net
    return _NET_CACHE["net"]


def teacher_score_from_weights(weights: dict, x: np.ndarray,
                               t: np.ndarray) -> np.ndarray:
    """Evaluate s_theta(x, t) = -eps_theta(x, t) / sqrt(1 - alpha^2)."""
    import torch

    net = _get_net(weights)
    with torch.no_grad():
        inp = torch.tensor(np.stack([np.asarray(x) / 2.5,
                                     np.asarray(t)], axis=1),
                           dtype=torch.float32)
        eps = net(inp)[:, 0].numpy()
    return -eps / np.sqrt(1.0 - alpha(t) ** 2)


def teacher_reverse_drift(weights: dict, xtau: np.ndarray) -> np.ndarray:
    xtau = np.atleast_2d(np.asarray(xtau, dtype=np.float64))
    x, tau = xtau[:, 0], xtau[:, 1]
    s = 1.0 - tau
    b = beta(s)
    return (0.5 * b * x + b * teacher_score_from_weights(weights, x, s))[:, None]


def teacher_reverse_g(xtau: np.ndarray) -> np.ndarray:
    return true_reverse_g(xtau)


# ---------------------------------------------------------------- SINDy

def fit_field_powers(x: np.ndarray, y: np.ndarray, degree: int,
                     threshold: float, ridge: float) -> PolynomialField:
    return fit_polynomial_field(x, y, degree, threshold, ridge)


def vif_max(x: np.ndarray, degree: int) -> float:
    """Max VIF of the full degree-`degree` library on states x, constant excluded."""
    from .common import evaluate_powers
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    powers = monomial_powers(x.shape[1], degree)
    theta = evaluate_powers(z, powers)
    keep = [j for j, p in enumerate(powers) if p.sum() > 0]
    theta = theta[:, keep]
    worst = 1.0
    for j in range(theta.shape[1]):
        others = np.delete(theta, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, theta[:, j], rcond=None)
        resid = theta[:, j] - others @ coef
        ss_tot = float(np.sum((theta[:, j] - theta[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-15)
        worst = max(worst, 1.0 / max(1.0 - r2, 1e-9))
    return float(worst)


def select_field(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray,
                 y_va: np.ndarray, degrees: list[int], thresholds: list[float],
                 ridge: float) -> tuple[PolynomialField, dict[str, Any]]:
    best: PolynomialField | None = None
    best_val = np.inf
    best_meta: dict[str, Any] = {}
    for degree in degrees:
        for thr in thresholds:
            field = fit_field_powers(x_tr, y_tr, degree, thr, ridge)
            val = nrmse(y_va, field.predict(x_va))
            if val < best_val:
                best_val, best, best_meta = val, field, {
                    "degree": degree, "threshold": thr,
                    "active_terms": int(np.count_nonzero(field.coefficients))}
    assert best is not None
    return best, best_meta


# ---------------------------------------------------------------- simulation

def simulate_reverse(drift_fn, g_fn, x0: np.ndarray, dt: float,
                     n_steps: int, increments: np.ndarray,
                     failure_radius: float) -> dict[str, Any]:
    """Euler-Maruyama reverse sampling with step-dependent drift.

    ``drift_fn(x, tau)`` and ``g_fn(x, tau)`` take states and the current
    reverse time; ``increments`` has shape (n, n_steps) of dW increments.
    Failures are latched and classified; failed samples keep their mass.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    n = len(x)
    failed = np.zeros(n, dtype=bool)
    excursion = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    for step in range(n_steps):
        tau = step * dt
        active = ~failed
        if active.any():
            xa = x[active][:, None]
            tau_arr = np.full(active.sum(), tau)
            drift = drift_fn(xa, tau_arr)[:, 0]
            g = g_fn(xa, tau_arr)[:, 0]
            x[active] = x[active] + drift * dt + g * increments[active, step]
        excursion = excursion | (np.abs(x) > 8.0)
        bad_nf = ~np.isfinite(x)
        bad_r = np.abs(x) > failure_radius
        newly = (~failed) & (bad_nf | bad_r)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nf[idx] else "outside_radius"
        failed = failed | newly
        x[failed] = np.nan
    return {"final": x, "failed": failed, "failure_class": failure_class,
            "failure_rate": float(np.mean(failed)),
            "excursion_rate": float(np.mean(excursion))}


def w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64)[np.isfinite(a)])
    b = np.sort(np.asarray(b, dtype=np.float64)[np.isfinite(b)])
    m = min(len(a), len(b))
    return float(np.mean(np.abs(a[:m] - b[:m])))


def _drift_at_tau(weights: dict, tau_scalar: float):
    def fn(xa: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
        xtau = np.stack([xa[:, 0], tau_arr], axis=1)
        return teacher_reverse_drift(weights, xtau)
    return fn


def _g_at_tau(tau_scalar: float):
    def fn(xa: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
        xtau = np.stack([xa[:, 0], tau_arr], axis=1)
        return teacher_reverse_g(xtau)
    return fn


def _field_drift_at_tau(field: PolynomialField):
    def fn(xa: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
        xtau = np.stack([xa[:, 0], tau_arr], axis=1)
        return field.predict(xtau)
    return fn


def _field_g_at_tau(field: PolynomialField):
    def fn(xa: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
        xtau = np.stack([xa[:, 0], tau_arr], axis=1)
        return field.predict(xtau)
    return fn


def _truth_drift_at_tau():
    def fn(xa: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
        xtau = np.stack([xa[:, 0], tau_arr], axis=1)
        return true_reverse_drift(xtau)
    return fn


def _noscore_drift_at_tau():
    def fn(xa: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
        s = 1.0 - tau_arr
        return (0.5 * beta(s) * xa[:, 0])[:, None]
    return fn


def simulate_pfode(weights: dict, x0: np.ndarray, dt: float,
                   n_steps: int) -> np.ndarray:
    """Deterministic probability-flow ODE with the teacher score."""
    x = np.array(x0, dtype=np.float64, copy=True)
    for step in range(n_steps):
        tau = step * dt
        s = 1.0 - tau
        b = beta(np.array([s]))[0]
        score = teacher_score_from_weights(weights, x, np.full(len(x), s))
        x = x + (0.5 * b * x + 0.5 * b * score) * dt
    return x


# ---------------------------------------------------------------- main

def main() -> dict[str, Any]:
    started = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    rng = np.random.default_rng(int(cfg["simulation"]["seed"]))
    queries = generate_queries(cfg)
    x_tr, x_va, x_te = queries["train"], queries["validation"], queries["test"]

    teacher = train_score_teacher(cfg, queries["x0"][: int(cfg["split"]["train"])])
    weights = teacher["weights"]

    drift_tr = teacher_reverse_drift(weights, x_tr)
    drift_va = teacher_reverse_drift(weights, x_va)
    drift_te = teacher_reverse_drift(weights, x_te)
    g_tr = teacher_reverse_g(x_tr)
    g_va = teacher_reverse_g(x_va)
    g_te = teacher_reverse_g(x_te)
    truth_drift_te = true_reverse_drift(x_te)
    truth_g_te = true_reverse_g(x_te)

    vif = vif_max(x_tr, 3)

    ridge = float(cfg["distillation"]["ridge"])
    thresholds = [float(v) for v in cfg["distillation"]["thresholds"]]
    drift_field, drift_meta = select_field(
        x_tr, drift_tr, x_va, drift_va,
        [int(d) for d in cfg["distillation"]["degrees"]["drift_full"]],
        thresholds, ridge)
    g_field, g_meta = select_field(
        x_tr, g_tr, x_va, g_va,
        [int(d) for d in cfg["distillation"]["degrees"]["diffusion_full"]],
        thresholds, ridge)

    # Oracle diagnostic: same procedure on truth field values.
    oracle_drift, _ = select_field(
        x_tr, true_reverse_drift(x_tr), x_va, true_reverse_drift(x_va),
        [int(d) for d in cfg["distillation"]["degrees"]["drift_full"]],
        thresholds, ridge)

    results: dict[str, Any] = {
        "teacher_dsm_loss_final200": teacher["final_dsm_loss"],
        "vif_max_degree3": vif,
        "drift_selection": drift_meta,
        "g_selection": g_meta,
        "drift_test_nrmse_vs_teacher": nrmse(drift_te, drift_field.predict(x_te)),
        "drift_test_nrmse_vs_truth": nrmse(truth_drift_te, drift_field.predict(x_te)),
        "teacher_drift_test_nrmse_vs_truth": nrmse(truth_drift_te, drift_te),
        "g_test_nrmse_vs_teacher": nrmse(g_te, g_field.predict(x_te)),
        "g_test_nrmse_vs_truth": nrmse(truth_g_te, g_field.predict(x_te)),
        "sigma_test_nrmse_vs_teacher": nrmse(g_te ** 2, g_field.predict(x_te) ** 2),
        "oracle_drift_test_nrmse_vs_truth": nrmse(
            truth_drift_te, oracle_drift.predict(x_te)),
        "query_ranges": {
            "x": [float(x_tr[:, 0].min()), float(x_tr[:, 0].max())],
            "tau": [float(x_tr[:, 1].min()), float(x_tr[:, 1].max())]},
    }

    # H2c: across-x spread of symbolic g at fixed tau.
    xs = np.linspace(-4.0, 4.0, 81)
    spread = {}
    for tau in (0.25, 0.5, 0.75):
        vals = g_field.predict(np.stack([xs, np.full(len(xs), tau)], axis=1))[:, 0]
        spread[str(tau)] = float(vals.std() / max(abs(vals.mean()), 1e-12))
    results["g_x_spread_by_tau"] = spread
    results["g_x_spread_max"] = max(spread.values())

    # Regime split of teacher drift error (H6 analogue) + per-step SNR.
    for name, mask in (("near_linear_tau_lt_0.3", x_te[:, 1] < 0.3),
                       ("nonlinear_tau_gt_0.7", x_te[:, 1] > 0.7)):
        results[f"teacher_drift_nrmse_vs_truth_{name}"] = nrmse(
            truth_drift_te[mask], drift_te[mask])
        f_true = np.abs(truth_drift_te[mask][:, 0])
        g_true = truth_g_te[mask][:, 0]
        results[f"per_step_snr_{name}"] = float(
            np.mean(f_true * 0.01 / (g_true * np.sqrt(0.01))))

    # Off-manifold grid (reporting only), three-way comparison.
    gx = np.linspace(-4.0, 4.0, 17)
    gtau = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    grid = np.array([[x, t] for t in gtau for x in gx])
    grid_res = {
        "symbolic_vs_teacher_drift": nrmse(teacher_reverse_drift(weights, grid),
                                          drift_field.predict(grid)),
        "teacher_vs_truth_drift": nrmse(true_reverse_drift(grid),
                                        teacher_reverse_drift(weights, grid)),
        "symbolic_vs_truth_drift": nrmse(true_reverse_drift(grid),
                                         drift_field.predict(grid)),
        "symbolic_vs_teacher_drift_rms": float(np.sqrt(np.mean(
            (teacher_reverse_drift(weights, grid)
             - drift_field.predict(grid)) ** 2))),
        "teacher_vs_truth_drift_rms": float(np.sqrt(np.mean(
            (true_reverse_drift(grid)
             - teacher_reverse_drift(weights, grid)) ** 2))),
        "symbolic_vs_teacher_g": nrmse(teacher_reverse_g(grid),
                                       g_field.predict(grid)),
        "teacher_vs_truth_g": nrmse(true_reverse_g(grid),
                                    teacher_reverse_g(grid)),
    }
    for tau in gtau:
        sel = grid[:, 1] == tau
        grid_res[f"tau_{tau}_symbolic_vs_teacher_drift"] = nrmse(
            teacher_reverse_drift(weights, grid[sel]), drift_field.predict(grid[sel]))
        grid_res[f"tau_{tau}_teacher_vs_truth_drift"] = nrmse(
            true_reverse_drift(grid[sel]), teacher_reverse_drift(weights, grid[sel]))
    results["grid"] = grid_res

    # ---- reverse sampling (H4 / H5) ----
    n_sim = int(cfg["simulation"]["n_samples"])
    dt = float(cfg["simulation"]["dt"])
    n_steps = int(round(1.0 / dt))
    prior = rng.standard_normal(n_sim)
    fine_dt = 0.002
    fine_steps = int(round(1.0 / fine_dt))
    dW_fine = rng.standard_normal((n_sim, fine_steps)) * np.sqrt(fine_dt)
    truth_samples = rng.choice([-1.0, 1.0], size=n_sim) * MU \
        + SIG0 * rng.standard_normal(n_sim)

    def run(drift_fn, g_fn, dW, step_dt):
        n_st = dW.shape[1]
        return simulate_reverse(drift_fn, g_fn, prior, step_dt, n_st, dW, 30.0)

    teacher_sim = run(_drift_at_tau(weights, 0.0), _g_at_tau(0.0),
                      dW_fine, fine_dt)
    symbolic_sim = run(_field_drift_at_tau(drift_field),
                       _field_g_at_tau(g_field), dW_fine, fine_dt)
    noscore_sim = run(_noscore_drift_at_tau(), _g_at_tau(0.0), dW_fine, fine_dt)
    truth_sim = run(_truth_drift_at_tau(), _g_at_tau(0.0), dW_fine, fine_dt)
    pfode = simulate_pfode(weights, prior, dt, n_steps)

    def dist_stats(sim: dict[str, Any]) -> dict[str, Any]:
        fin = sim["final"][np.isfinite(sim["final"])]
        return {"failure_rate": sim["failure_rate"],
                "excursion_rate": sim["excursion_rate"],
                "failure_class_counts": {
                    c: int(sum(1 for v in sim["failure_class"] if v == c))
                    for c in ("non_finite", "outside_radius")},
                "w1_vs_truth_samples": w1_1d(fin, truth_samples),
                "mode_mass_gt1": float(np.mean(np.abs(fin) > 1.0)),
                "mean": float(np.mean(fin)), "std": float(np.std(fin))}

    results["sampling"] = {
        "teacher": dist_stats(teacher_sim),
        "symbolic": dist_stats(symbolic_sim),
        "noscore": dist_stats(noscore_sim),
        "truth_sde": dist_stats(truth_sim),
        "pfode_teacher": {"w1_vs_truth_samples": w1_1d(pfode, truth_samples),
                          "mode_mass_gt1": float(np.mean(np.abs(pfode) > 1.0)),
                          "mean": float(np.mean(pfode)),
                          "std": float(np.std(pfode))},
        "truth_samples": {"w1_vs_truth_samples": 0.0,
                          "mode_mass_gt1": float(np.mean(np.abs(truth_samples) > 1.0)),
                          "mean": float(np.mean(truth_samples)),
                          "std": float(np.std(truth_samples))},
    }
    # Paired RMSE teacher vs symbolic on surviving pairs, per step-end only
    # (final states, the reported paired quantity).
    pair = np.isfinite(teacher_sim["final"]) & np.isfinite(symbolic_sim["final"])
    results["sampling"]["paired_final_rmse_symbolic_vs_teacher"] = float(
        np.sqrt(np.mean((symbolic_sim["final"][pair]
                         - teacher_sim["final"][pair]) ** 2))) if pair.any() else None

    # ---- H6: coarse vs fine reverse solver (paired increments) ----
    coarse_dt = 0.05
    block = int(round(coarse_dt / fine_dt))
    dW_coarse = dW_fine.reshape(n_sim, int(fine_steps / block), block).sum(axis=2)
    teacher_coarse = run(_drift_at_tau(weights, 0.0), _g_at_tau(0.0),
                         dW_coarse, coarse_dt)
    fin_coarse = teacher_coarse["final"][np.isfinite(teacher_coarse["final"])]
    results["h6"] = {
        "w1_fine": results["sampling"]["teacher"]["w1_vs_truth_samples"],
        "w1_coarse": w1_1d(fin_coarse, truth_samples),
        "coarse_failure_rate": teacher_coarse["failure_rate"],
    }
    results["h6"]["w1_degradation"] = (results["h6"]["w1_coarse"]
                                       - results["h6"]["w1_fine"])

    # ---- H7: one-step reverse prediction on test queries ----
    sub = x_te[:4000]
    dt1 = 0.01
    z1 = rng.standard_normal(len(sub))
    s_sub = 1.0 - sub[:, 1]
    nxt_truth = (sub[:, 0] + true_reverse_drift(sub)[:, 0] * dt1
                 + true_reverse_g(sub)[:, 0] * np.sqrt(dt1) * z1)
    pred_teacher = sub[:, 0] + teacher_reverse_drift(weights, sub)[:, 0] * dt1
    pred_sym = sub[:, 0] + drift_field.predict(sub)[:, 0] * dt1
    se_t = (nxt_truth - pred_teacher) ** 2
    se_s = (nxt_truth - pred_sym) ** 2
    diff = se_s - se_t
    results["h7"] = {
        "mse_teacher": float(np.mean(se_t)),
        "mse_symbolic": float(np.mean(se_s)),
        "mse_abs_diff": float(abs(np.mean(se_s) - np.mean(se_t))),
        "paired_se": float(np.std(diff, ddof=1) / np.sqrt(len(diff))),
    }

    # ---- save artifacts ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drift_field.save(OUT_DIR / "symbolic_f.json")
    g_field.save(OUT_DIR / "symbolic_g.json")
    (OUT_DIR / "teacher_weights.json").write_text(
        json.dumps({"weights": weights,
                    "dsm_loss_final200": teacher["final_dsm_loss"]}, indent=2))
    expanded_f = expand_to_original(drift_field)
    expanded_g = expand_to_original(g_field)
    tex = ("Symbolic reverse-time SDE in original coordinates "
           "(variables $x$, $\\\\tau = 1 - t$):\n\n$$\n"
           + tex_equations(expanded_f, ("f_{\\\\mathrm{rev}}(x, \\\\tau)",),
                           var_names=("x", "\\\\tau"))
           + "\n$$\n\n$$\n"
           + tex_equations(expanded_g, ("g_{\\\\mathrm{rev}}(x, \\\\tau)",),
                           var_names=("x", "\\\\tau"))
           + "\n$$\n\n"
           "with $dX_\\\\tau = f_{\\\\mathrm{rev}}(x, \\\\tau)\\\\,d\\\\tau "
           "+ g_{\\\\mathrm{rev}}(x, \\\\tau)\\\\,d\\\\bar{W}_\\\\tau$.\n")
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    (OUT_DIR / "symbolic_expanded.json").write_text(json.dumps({
        "drift": [{",".join(map(str, exp)): coef for exp, coef in poly.items()}
                  for poly in expanded_f],
        "diffusion": [{",".join(map(str, exp)): coef for exp, coef in poly.items()}
                      for poly in expanded_g]}, indent=2))
    results["elapsed_seconds"] = float(time.time() - started)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
