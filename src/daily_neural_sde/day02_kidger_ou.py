"""Day 2: Kidger et al. (2021) time-dependent Ornstein-Uhlenbeck process -> SINDy distillation.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day02.json``):

1. Regenerate the paper's synthetic example (Section 4.1): 8192
   trajectories of dz = (mu*t - theta*z) dt + sigma dW with
   mu = 0.02, theta = 0.1, sigma = 0.4, sampled at every integer on
   [0, 63]. The truth is affine, so data are drawn with the exact
   transition (not Euler-Maruyama).
2. Estimate a neural SDE *in the observed 1D coordinate* (declared
   deviation from the paper's Wasserstein-GAN generator/discriminator
   pair; see the config): separate MLPs of (t, z) for drift and
   diffusion, trained with four Gaussian transition likelihoods. The
   distilled teacher is the substep-moment estimator (v4), chosen a
   priori per the Day 1 reflection: when the absolute diffusion level
   is a primary question, use a simulation-based estimator with
   internal substeps from the start.
3. Distill the teacher into polynomial SINDy models in (t, z) by
   direct function queries on trajectory-disjoint splits, in the
   conventions of ``src/distill.py`` (validation-only selection,
   test once). A collinearity diagnostic on (t, z) is computed before
   fitting and gates the coefficient-level hypothesis, per the Day 1
   reflection.
4. Validate by paired Euler-Maruyama simulation (shared Brownian
   increments, time-aware drift/diffusion) against the neural SDE and
   the ground truth, retaining failed-trajectory mass.

Run from the repository root:  ``python -m src.daily_neural_sde.day02_kidger_ou``
Outputs land in ``outputs/daily_neural_sde/day02/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, evaluate_expanded, expand_to_original,
                     fit_polynomial_field, nrmse, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day02.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day02"

VAR_NAMES = ("t", "z")
TRUE_DRIFT_SUPPORT: list[dict[tuple[int, ...], float]] = [{(1, 0): 0.02, (0, 1): -0.1}]
N_SUBSTEPS = 8


def true_drift(tz: np.ndarray, cfg: dict) -> np.ndarray:
    """Ground-truth OU drift for (t, z) pairs -> (n, 1)."""
    gt = cfg["ground_truth_sde"]
    tz = np.atleast_2d(np.asarray(tz, dtype=np.float64))
    return (gt["mu"] * tz[:, [0]] - gt["theta"] * tz[:, [1]])


def true_diffusion(tz: np.ndarray, cfg: dict) -> np.ndarray:
    """Ground-truth constant diffusion for (t, z) pairs -> (n, 1)."""
    gt = cfg["ground_truth_sde"]
    n = len(np.atleast_2d(tz))
    return np.full((n, 1), gt["sigma"])


def exact_transition_mean_var(t: np.ndarray, z: np.ndarray, dt: float,
                              cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean/variance of the affine OU transition over one step.

    dz = (mu*t - theta*z) dt + sigma dW, with the time argument running
    from t to t + dt. The mean uses the integral of the ramp forcing
    against the OU kernel; the variance is sigma^2 (1 - exp(-2 theta dt))
    / (2 theta).
    """
    gt = cfg["ground_truth_sde"]
    mu, theta, sigma = gt["mu"], gt["theta"], gt["sigma"]
    decay = np.exp(-theta * dt)
    integral = ((t + dt) * (1.0 - decay) / theta
                - (1.0 - decay * (1.0 + theta * dt)) / theta ** 2)
    mean = decay * z + mu * integral
    var = sigma ** 2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
    return mean, np.full_like(mean, var)


def generate_data(cfg: dict) -> np.ndarray:
    """Simulate the paper's OU dataset with exact transitions -> (8192, 64)."""
    data_cfg = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_traj = data_cfg["n_trajectories"]
    n_steps = int(round(data_cfg["t_end"] / data_cfg["dt_obs"]))
    z = rng.standard_normal(n_traj)
    trajectories = np.empty((n_traj, n_steps + 1))
    trajectories[:, 0] = z
    for step in range(n_steps):
        t = float(step * data_cfg["dt_obs"])
        mean, var = exact_transition_mean_var(
            np.full(n_traj, t), z, data_cfg["dt_obs"], cfg)
        z = mean + np.sqrt(var) * rng.standard_normal(n_traj)
        trajectories[:, step + 1] = z
    return trajectories


def split_indices(cfg: dict) -> dict[str, np.ndarray]:
    """Trajectory-disjoint train / validation / test split (seeded)."""
    rng = np.random.default_rng(cfg["split"]["seed"])
    n = cfg["data"]["n_trajectories"]
    perm = rng.permutation(n)
    n_train = cfg["split"]["train"]
    n_val = cfg["split"]["validation"]
    return {"train": perm[:n_train],
            "validation": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:]}


def query_states(trajectories: np.ndarray, idx: np.ndarray,
                 dt_obs: float) -> np.ndarray:
    """(t, z) pairs for the given trajectories: one row per observation."""
    block = trajectories[idx]
    n_traj, n_obs = block.shape
    times = np.tile(np.arange(n_obs) * dt_obs, n_traj)
    return np.stack([times, block.reshape(-1)], axis=1)


def _softplus(v: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, v)


class NumpyMLP:
    """NumPy forward pass of a 2-hidden-layer Softplus MLP (exported weights)."""

    def __init__(self, layers: list[dict[str, Any]], output_transform: str) -> None:
        self.layers = [(np.asarray(l["weight"]), np.asarray(l["bias"])) for l in layers]
        self.output_transform = output_transform

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Standardized (t, z) inputs (n, 2) -> outputs (n, 1)."""
        h = np.asarray(u, dtype=np.float64)
        for idx, (weight, bias) in enumerate(self.layers):
            h = h @ weight.T + bias
            if idx < len(self.layers) - 1 or self.output_transform == "softplus_floor":
                h = _softplus(h)
        if self.output_transform == "softplus_floor":
            h = h + 0.01
        return h


class NumpyTeacher:
    """Neural SDE teacher evaluated from its exported JSON weights."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.mean = np.asarray(payload["input_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["input_scale"], dtype=np.float64)
        self.drift_net = NumpyMLP(payload["drift_net"], "identity")
        self.diffusion_net = NumpyMLP(payload["diffusion_net"], "softplus_floor")

    def drift(self, tz: np.ndarray) -> np.ndarray:
        """Teacher drift for (t, z) pairs -> (n, 1)."""
        tz = np.atleast_2d(tz)
        return self.drift_net((tz - self.mean) / self.scale)

    def diffusion(self, tz: np.ndarray) -> np.ndarray:
        """Teacher diagonal diffusion for (t, z) pairs -> (n, 1)."""
        tz = np.atleast_2d(tz)
        return self.diffusion_net((tz - self.mean) / self.scale)


def train_teacher(trajectories: np.ndarray, splits: dict[str, np.ndarray],
                  cfg: dict, estimator: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the neural SDE teacher with PyTorch; return (weights, log).

    Estimators (all Gaussian transition likelihoods, no observation
    noise; the Jacobian A = df/dz is a central finite difference in
    standardized coordinates, treated as a fixed local coefficient):

    - ``em``: mean z + f*dt, var g^2*dt.
    - ``local_linearization``: mean z + f*dt + 0.5*dt^2*A*f,
      var g^2*dt + A*g^2*dt^2.
    - ``local_linearization_exp``: mean z + Psi*f with
      Psi = (exp(A*dt) - 1)/A, var kappa*g^2 with
      kappa = (exp(2*A*dt) - 1)/(2*A).
    - ``substep_moments``: K = 8 internal substeps; per substep the mean
      is advanced by f at the current mean and the variance by
      (2*A*v + g^2)*h, with A clamped to [-5, 5]; the NLL is evaluated
      at the observed endpoint under the propagated (mean, variance).
    """
    import torch

    torch.manual_seed(cfg["seed"])
    teacher_cfg = cfg["teacher"]
    dt = cfg["data"]["dt_obs"]

    def transitions(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = splits[name]
        block = trajectories[idx]
        t_start = torch.tensor(np.tile(np.arange(block.shape[1] - 1) * dt,
                                       len(idx)), dtype=torch.float32)
        z_t = torch.tensor(block[:, :-1].reshape(-1), dtype=torch.float32)
        dx = torch.tensor((block[:, 1:] - block[:, :-1]).reshape(-1), dtype=torch.float32)
        return t_start, z_t, dx

    train_t, train_z, train_dx = transitions("train")
    val_t, val_z, val_dx = transitions("validation")
    train_u = torch.stack([train_t, train_z], dim=1)
    mean = train_u.mean(dim=0)
    std = train_u.std(dim=0)

    def make_net() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(2, 32), torch.nn.Softplus(),
            torch.nn.Linear(32, 32), torch.nn.Softplus(),
            torch.nn.Linear(32, 1))

    drift_net = make_net()
    diffusion_net = make_net()
    params = list(drift_net.parameters()) + list(diffusion_net.parameters())
    optimizer = torch.optim.Adam(params, lr=teacher_cfg["optimizer_lr"])
    eps = 1e-3

    def drift_at(u: torch.Tensor) -> torch.Tensor:
        return drift_net(u).squeeze(-1)

    def jac_phys(u: torch.Tensor) -> torch.Tensor:
        """df/dz (physical) via central finite differences, no_grad."""
        with torch.no_grad():
            shift = torch.zeros_like(u)
            shift[:, 1] = eps
            plus = drift_at(u + shift)
            minus = drift_at(u - shift)
        return (plus - minus) / (2.0 * eps * std[1])

    def nll(t_start: torch.Tensor, z_t: torch.Tensor,
            dx: torch.Tensor) -> torch.Tensor:
        u = torch.stack([t_start, z_t], dim=1)
        uz = (u - mean) / std
        drift = drift_at(uz)
        diffusion = torch.nn.functional.softplus(diffusion_net(uz)).squeeze(-1) + 0.01
        if estimator == "em":
            mean_step = drift * dt
            var = diffusion ** 2 * dt
        elif estimator == "local_linearization":
            a = jac_phys(uz)
            mean_step = drift * dt + 0.5 * dt ** 2 * a * drift
            var = diffusion ** 2 * dt + a * diffusion ** 2 * dt ** 2
        elif estimator == "local_linearization_exp":
            a = jac_phys(uz)
            psi = torch.where(torch.abs(a) > 1e-8, (torch.exp(a * dt) - 1.0) / a,
                              torch.full_like(a, dt))
            kappa = torch.where(torch.abs(a) > 1e-8,
                                (torch.exp(2 * a * dt) - 1.0) / (2 * a),
                                torch.full_like(a, dt))
            mean_step = psi * drift
            var = kappa * diffusion ** 2
        else:  # substep_moments
            h = dt / N_SUBSTEPS
            m = z_t.clone()
            v = torch.zeros_like(z_t)
            for k in range(N_SUBSTEPS):
                t_k = t_start + k * h
                u_k = (torch.stack([t_k, m], dim=1) - mean) / std
                f_k = drift_at(u_k)
                a_k = torch.clamp(jac_phys(u_k), -5.0, 5.0)
                g_k = torch.nn.functional.softplus(diffusion_net(u_k)).squeeze(-1) + 0.01
                m = m + f_k * h
                v = v + (2.0 * a_k * v + g_k ** 2) * h
            mean_step = m - z_t
            var = v
        var = torch.clamp(var, min=1e-8)
        resid = dx - mean_step
        return (0.5 * (torch.log(2 * np.pi * var) + resid ** 2 / var)).mean()

    epochs = teacher_cfg["epochs"][estimator]
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    batch = teacher_cfg["batch_transitions"]
    for epoch in range(epochs):
        drift_net.train()
        diffusion_net.train()
        perm = torch.randperm(len(train_z))
        for start in range(0, len(train_z), batch):
            sel = perm[start:start + batch]
            optimizer.zero_grad()
            loss = nll(train_t[sel], train_z[sel], train_dx[sel])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = float(nll(val_t, val_z, val_dx))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {"drift": {k: v.tolist() for k, v in drift_net.state_dict().items()},
                          "diffusion": {k: v.tolist() for k, v in diffusion_net.state_dict().items()}}
    assert best_state is not None

    def export(state: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
        layers = []
        for layer_idx in (0, 2, 4):
            layers.append({"weight": state[prefix][f"{layer_idx}.weight"],
                           "bias": state[prefix][f"{layer_idx}.bias"]})
        return layers

    payload = {"input_mean": mean.tolist(), "input_scale": std.tolist(),
               "drift_net": export(best_state, "drift"),
               "diffusion_net": export(best_state, "diffusion"),
               "architecture": teacher_cfg["architecture"],
               "estimator": estimator}
    with torch.no_grad():
        test_t, test_z, test_dx = transitions("test")
        drift_net.load_state_dict({k: torch.tensor(v) for k, v in best_state["drift"].items()})
        diffusion_net.load_state_dict({k: torch.tensor(v) for k, v in best_state["diffusion"].items()})
        log = {"estimator": estimator,
               "best_epoch": best_epoch, "best_validation_nll": best_val,
               "test_nll_at_best_checkpoint": float(nll(test_t, test_z, test_dx)),
               "epochs": epochs,
               "n_train_transitions": int(len(train_z)),
               "torch_version": torch.__version__}
    return payload, log


def _candidate_table(states: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                     degrees: list[int], thresholds: list[float],
                     ridge: float) -> tuple[list[dict[str, Any]], PolynomialField,
                                             dict[int, PolynomialField]]:
    """Fit all (degree, threshold) candidates; select on validation NRMSE.

    Also returns the best-validation model within each degree (for the
    pre-declared per-degree test summary; never used for selection).
    """
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[int, float], PolynomialField] = {}
    for degree in degrees:
        for threshold in thresholds:
            model = fit_polynomial_field(states["train"], targets["train"],
                                         degree, threshold, ridge)
            rows.append({"degree": degree, "threshold": threshold,
                         "active_terms": int(np.count_nonzero(model.coefficients)),
                         "library_size": int(len(model.powers)),
                         "validation_nrmse": nrmse(targets["validation"],
                                                   model.predict(states["validation"])),
                         "train_nrmse": nrmse(targets["train"],
                                              model.predict(states["train"]))})
            fitted[(degree, threshold)] = model
    best = min(rows, key=lambda r: (r["validation_nrmse"], r["active_terms"]))
    chosen = fitted[(best["degree"], best["threshold"])]
    chosen.metadata["selection"] = best
    per_degree: dict[int, PolynomialField] = {}
    for degree in degrees:
        row = min((r for r in rows if r["degree"] == degree),
                  key=lambda r: (r["validation_nrmse"], r["active_terms"]))
        per_degree[degree] = fitted[(degree, row["threshold"])]
    return rows, chosen, per_degree


def simulate_time_aware(drift_fn, diffusion_fn, z0: np.ndarray, dt: float,
                        n_steps: int, increments: np.ndarray,
                        failure_radius: float,
                        obs_every: int) -> dict[str, Any]:
    """Paired Euler-Maruyama simulation of a time-dependent 1D SDE.

    ``drift_fn`` / ``diffusion_fn`` take (t_array, z_array) -> (n,).
    ``increments`` has shape (n, n_steps). Failures are latched and
    classified as ``non_finite`` or ``outside_radius``; failed
    trajectories keep their mass in the returned counts and are NaN
    after failure.
    """
    z = np.array(z0, dtype=np.float64, copy=True)
    n = len(z)
    if increments.shape != (n, n_steps):
        raise ValueError("increments shape mismatch")
    failed = np.zeros(n, dtype=bool)
    failure_class: list[str | None] = [None] * n
    observed = [z.copy()]
    sqrt_dt = np.sqrt(dt)
    for step in range(n_steps):
        t = step * dt
        active = ~failed
        if active.any():
            t_arr = np.full(int(active.sum()), t)
            drift = drift_fn(t_arr, z[active])
            diffusion = diffusion_fn(t_arr, z[active])
            z[active] = z[active] + drift * dt + diffusion * sqrt_dt * increments[active, step]
        bad_nonfinite = ~np.isfinite(z)
        bad_radius = np.abs(z) > failure_radius
        newly = (~failed) & (bad_nonfinite | bad_radius)
        for idx in np.flatnonzero(newly):
            failure_class[idx] = "non_finite" if bad_nonfinite[idx] else "outside_radius"
        failed = failed | newly
        z[failed] = np.nan
        if (step + 1) % obs_every == 0:
            observed.append(z.copy())
    return {"states": np.stack(observed, axis=1), "failed": failed,
            "failure_class": failure_class,
            "failure_rate": float(np.mean(failed))}


def run() -> dict[str, Any]:
    """Execute the full Day 2 pipeline and write all artifacts."""
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"day": 2, "date": cfg["date"], "paper": cfg["paper"],
                               "config_path": "configs/daily_neural_sde_day02.json"}

    print("[1/7] generating data (paper Section 4.1 setting, exact transitions)")
    trajectories = generate_data(cfg)
    splits = split_indices(cfg)
    results["data"] = {
        "shape": list(trajectories.shape),
        "state_mean": float(trajectories.mean()),
        "state_std": float(trajectories.std()),
        "min_state": float(trajectories.min()),
        "max_state": float(trajectories.max()),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "note": cfg["data"]["note"],
        "initial_law": cfg["data"]["initial_law"]}

    states = {name: query_states(trajectories, idx, cfg["data"]["dt_obs"])
              for name, idx in splits.items()}
    truth_targets = {name: {"drift": true_drift(s, cfg), "diffusion": true_diffusion(s, cfg)}
                     for name, s in states.items()}

    print("[2/7] collinearity diagnostic on train query states (before any fit)")
    train_t, train_z = states["train"][:, 0], states["train"][:, 1]
    corr_tz = float(np.corrcoef(train_t, train_z)[0, 1])
    vif_tz = float(1.0 / max(1.0 - corr_tz ** 2, 1e-12))
    feats = np.stack([train_t, train_z, train_t ** 2, train_t * train_z, train_z ** 2], axis=1)
    feat_corr = np.corrcoef(feats, rowvar=False)
    off_diag = np.abs(feat_corr[np.triu_indices(5, k=1)])
    results["collinearity"] = {
        "corr_t_z_train": corr_tz,
        "vif_t_z_train": vif_tz,
        "max_pairwise_abs_corr_degree2_features_train": float(off_diag.max()),
        "coefficient_thresholds_apply": bool(vif_tz <= 10.0),
        "note": "corr(t, z) is high by construction: the ramp forcing mu*t drives the OU mean upward over the horizon, so t and z move together on trajectories. This gates H3 (see config)."}
    print(f"    corr(t,z)={corr_tz:.4f} VIF={vif_tz:.2f} "
          f"coefficient thresholds apply: {vif_tz <= 10.0}")

    print("[3/7] training neural SDE teachers (PyTorch, CPU): v1 EM, v2 LL, v3 LL-exp, v4 substep-moments")
    estimator_comparison: dict[str, Any] = {}
    payload = None
    teacher_log: dict[str, Any] = {}
    for estimator in ("em", "local_linearization", "local_linearization_exp", "substep_moments"):
        est_payload, est_log = train_teacher(trajectories, splits, cfg, estimator)
        est_teacher = NumpyTeacher(est_payload)
        g_val = est_teacher.diffusion(states["validation"])
        t_val = states["validation"][:, 0]
        tercile_means = {}
        for label, mask in (("t_lt_21", t_val < 21), ("t_21_42", (t_val >= 21) & (t_val <= 42)),
                            ("t_gt_42", t_val > 42)):
            tercile_means[label] = float(g_val[mask].mean())
        estimator_comparison[estimator] = {
            "training": est_log,
            "vs_truth_test": {
                field: nrmse(truth_targets["test"][field],
                             getattr(est_teacher, field)(states["test"]))
                for field in ("drift", "diffusion")},
            "diffusion_mean_on_validation_states": float(g_val.mean()),
            "diffusion_std_on_validation_states": float(g_val.std()),
            "diffusion_mean_by_t_tercile_validation": tercile_means}
        print(f"    {estimator}: drift NRMSE vs truth (test) = "
              f"{estimator_comparison[estimator]['vs_truth_test']['drift']:.4f}, "
              f"g mean = {float(g_val.mean()):.4f}")
        if estimator == "substep_moments":
            payload, teacher_log = est_payload, est_log
    assert payload is not None
    results["estimator_comparison"] = estimator_comparison
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(payload, indent=1))
    teacher = NumpyTeacher(payload)
    results["teacher_training"] = teacher_log
    results["teacher_deviation"] = cfg["teacher"]["deviation_from_paper"]

    teacher_targets = {name: {"drift": teacher.drift(s), "diffusion": teacher.diffusion(s)}
                       for name, s in states.items()}
    results["teacher_vs_truth_test"] = {
        field: nrmse(truth_targets["test"][field], teacher_targets["test"][field])
        for field in ("drift", "diffusion")}

    print("[4/7] polynomial SINDy distillation (validation-selected)")
    dist_cfg = cfg["distillation"]
    models: dict[str, PolynomialField] = {}
    results["candidates"] = {}
    results["per_degree_test_vs_teacher"] = {}
    results["symbolic_vs_teacher_test"] = {}
    results["symbolic_vs_truth_test"] = {}
    for field, degrees in (("drift", dist_cfg["drift_degrees"]),
                           ("diffusion", dist_cfg["diffusion_degrees"])):
        rows, chosen, per_degree = _candidate_table(
            states, {k: v[field] for k, v in teacher_targets.items()},
            degrees, dist_cfg["thresholds"], dist_cfg["ridge"])
        models[field] = chosen
        chosen.save(OUT_DIR / f"symbolic_{field}.json")
        results["candidates"][field] = rows
        results["per_degree_test_vs_teacher"][field] = {
            str(deg): nrmse(teacher_targets["test"][field], model.predict(states["test"]))
            for deg, model in per_degree.items()}
        results["symbolic_vs_teacher_test"][field] = nrmse(
            teacher_targets["test"][field], chosen.predict(states["test"]))
        results["symbolic_vs_truth_test"][field] = nrmse(
            truth_targets["test"][field], chosen.predict(states["test"]))

    print("[5/7] oracle diagnostic (SINDy fitted to ground-truth fields)")
    oracle: dict[str, Any] = {}
    for field, degrees in (("drift", dist_cfg["drift_degrees"]),
                           ("diffusion", dist_cfg["diffusion_degrees"])):
        rows, chosen, _ = _candidate_table(
            states, {k: v[field] for k, v in truth_targets.items()},
            degrees, dist_cfg["thresholds"], dist_cfg["ridge"])
        oracle[field] = {"selection": chosen.metadata["selection"],
                         "test_nrmse_vs_truth": nrmse(truth_targets["test"][field],
                                                      chosen.predict(states["test"]))}
    results["oracle_sindy"] = oracle

    print("[6/7] coefficient recovery, Stratonovich diagnostic, off-manifold grid")
    drift_expanded = expand_to_original(models["drift"])
    diffusion_expanded = expand_to_original(models["diffusion"])
    recovery = []
    for exp, truth in TRUE_DRIFT_SUPPORT[0].items():
        fitted = drift_expanded[0].get(exp, 0.0)
        recovery.append({"power": list(exp), "truth": truth, "fitted": fitted,
                         "relative_error": float(abs(fitted - truth) / abs(truth)),
                         "sign_match": bool(np.sign(fitted) == np.sign(truth))})
    spurious_part = [{exp: coef for exp, coef in poly.items()
                      if exp not in TRUE_DRIFT_SUPPORT[0]} for poly in drift_expanded]
    spurious_values = evaluate_expanded(spurious_part, states["test"])
    full_values = models["drift"].predict(states["test"])
    spurious_fraction = float(np.sqrt(np.mean(spurious_values ** 2)
                                      / np.mean(full_values ** 2)))
    nonconst_fraction = float(np.sqrt(np.mean(
        evaluate_expanded([{exp: coef for exp, coef in poly.items() if sum(exp) > 0}
                           for poly in diffusion_expanded], states["test"]) ** 2)
        / np.mean(models["diffusion"].predict(states["test"]) ** 2)))
    # Stratonovich correction 0.5 * g * dg/dz for the fitted teacher (Ito fit).
    test_tz = states["test"]
    h_step = 0.01
    g_plus = teacher.diffusion(test_tz + np.array([0.0, h_step]))
    g_minus = teacher.diffusion(test_tz - np.array([0.0, h_step]))
    dg_dz = (g_plus - g_minus) / (2.0 * h_step)
    strat_corr = 0.5 * teacher_targets["test"]["diffusion"] * dg_dz
    # Off-manifold grid diagnostic (supplementary only, never used for selection).
    grid_t = np.array([0.0, 16.0, 32.0, 48.0, 63.0])
    grid_z = np.linspace(-2.0, 12.0, 15)
    grid = np.array([[t, z] for t in grid_t for z in grid_z])
    results["recovery"] = {
        "monomials": recovery,
        "spurious_rms_fraction_test": spurious_fraction,
        "diffusion_nonconstant_rms_fraction_test": nonconst_fraction,
        "drift_expanded": [{str(k): v for k, v in poly.items()} for poly in drift_expanded],
        "diffusion_expanded": [{str(k): v for k, v in poly.items()} for poly in diffusion_expanded],
        "stratonovich_correction_mean_abs_over_g_mean_test": float(
            np.mean(np.abs(strat_corr)) / np.mean(teacher_targets["test"]["diffusion"])),
        "stratonovich_correction_max_abs_test": float(np.max(np.abs(strat_corr))),
        "off_manifold_grid": {
            "n_points": int(len(grid)),
            "symbolic_vs_teacher_drift_nrmse": nrmse(teacher.drift(grid),
                                                      models["drift"].predict(grid)),
            "symbolic_vs_truth_drift_nrmse": nrmse(true_drift(grid, cfg),
                                                    models["drift"].predict(grid)),
            "teacher_vs_truth_drift_nrmse": nrmse(true_drift(grid, cfg),
                                                  teacher.drift(grid)),
            "symbolic_vs_teacher_diffusion_nrmse": nrmse(teacher.diffusion(grid),
                                                          models["diffusion"].predict(grid)),
        }}

    print("[7/7] paired simulation validation")
    sim_cfg = cfg["simulation"]
    test_traj = trajectories[splits["test"]]
    z0 = test_traj[:, 0]
    n_steps = int(round(sim_cfg["t_end"] / sim_cfg["dt"]))
    obs_every = int(round(cfg["data"]["dt_obs"] / sim_cfg["dt"]))
    rng = np.random.default_rng(sim_cfg["seed"])
    increments = rng.standard_normal((len(z0), n_steps))

    def teacher_drift_fn(t_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        return teacher.drift(np.stack([t_arr, z_arr], axis=1)).squeeze(-1)

    def teacher_diff_fn(t_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        return teacher.diffusion(np.stack([t_arr, z_arr], axis=1)).squeeze(-1)

    def symbolic_drift_fn(t_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        return models["drift"].predict(np.stack([t_arr, z_arr], axis=1)).squeeze(-1)

    def symbolic_diff_fn(t_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        return models["diffusion"].predict(np.stack([t_arr, z_arr], axis=1)).squeeze(-1)

    def truth_drift_fn(t_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        return true_drift(np.stack([t_arr, z_arr], axis=1), cfg).squeeze(-1)

    def truth_diff_fn(t_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        return true_diffusion(np.stack([t_arr, z_arr], axis=1), cfg).squeeze(-1)

    sims = {
        "neural": simulate_time_aware(teacher_drift_fn, teacher_diff_fn, z0,
                                      sim_cfg["dt"], n_steps, increments,
                                      sim_cfg["failure_radius"], obs_every),
        "symbolic": simulate_time_aware(symbolic_drift_fn, symbolic_diff_fn, z0,
                                        sim_cfg["dt"], n_steps, increments,
                                        sim_cfg["failure_radius"], obs_every),
        "truth": simulate_time_aware(truth_drift_fn, truth_diff_fn, z0,
                                     sim_cfg["dt"], n_steps, increments,
                                     sim_cfg["failure_radius"], obs_every),
    }

    def paired_rmse(a: dict, b: dict) -> tuple[float, int]:
        both = (~a["failed"]) & (~b["failed"])
        if both.sum() == 0:
            return float("nan"), 0
        diff = a["states"][both] - b["states"][both]
        per_traj = np.sqrt(np.mean(diff ** 2, axis=1))
        return float(np.median(per_traj)), int(both.sum())

    sim_summary: dict[str, Any] = {}
    for name, sim in sims.items():
        final = sim["states"][:, -1]
        ok = ~sim["failed"]
        sim_summary[name] = {
            "failure_rate": sim["failure_rate"],
            "failure_classes": {c: sim["failure_class"].count(c)
                                for c in ("non_finite", "outside_radius")},
            "endpoint_mean_finishers": float(np.nanmean(final[ok])),
            "endpoint_std_finishers": float(np.nanstd(final[ok])),
            "n_finishers": int(ok.sum()),
            "marginals_finishers": {
                str(t): {"mean": float(np.nanmean(sim["states"][ok, t])),
                         "std": float(np.nanstd(sim["states"][ok, t]))}
                for t in sim_cfg["marginal_times"]}}
    rmse_sym_neural, n_pair = paired_rmse(sims["symbolic"], sims["neural"])
    rmse_sym_truth, _ = paired_rmse(sims["symbolic"], sims["truth"])
    rmse_neural_truth, _ = paired_rmse(sims["neural"], sims["truth"])
    sim_summary["paired_rmse_median"] = {
        "symbolic_vs_neural": rmse_sym_neural,
        "symbolic_vs_truth": rmse_sym_truth,
        "neural_vs_truth": rmse_neural_truth,
        "n_pairs_symbolic_neural": n_pair}
    results["simulation"] = sim_summary

    hyp = cfg["hypotheses"]
    drift_rows = results["candidates"]["drift"]
    best_val = {deg: min(r["validation_nrmse"] for r in drift_rows if r["degree"] == deg)
                for deg in dist_cfg["drift_degrees"]}
    best_degree = min(best_val, key=best_val.get)
    em = estimator_comparison["em"]
    v4 = estimator_comparison["substep_moments"]
    coef_ok = bool(all(r["sign_match"] and r["relative_error"]
                       <= hyp["H3_truth_recovery"]["theta_relative_error_max"]
                       if r["power"] == [0, 1] else
                       r["sign_match"] and r["relative_error"]
                       <= hyp["H3_truth_recovery"]["mu_relative_error_max"]
                       for r in recovery))
    h3_field = bool(results["symbolic_vs_truth_test"]["drift"]
                    <= hyp["H3_truth_recovery"]["field_symbolic_drift_test_nrmse_vs_truth_max"]
                    and spurious_fraction
                    <= hyp["H3_truth_recovery"]["spurious_contribution_rms_fraction_max"])
    verdicts = {
        "H1_drift_distillation": bool(
            results["symbolic_vs_teacher_test"]["drift"]
            <= hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"]
            and best_val[1] <= best_val[best_degree] * (1 + hyp["H1_drift_distillation"]
                                                        ["degree1_within_relative_of_best_validation_nrmse"])),
        "H2_diffusion_distillation": bool(
            results["symbolic_vs_teacher_test"]["diffusion"]
            <= hyp["H2_diffusion_distillation"]["symbolic_diffusion_test_nrmse_vs_teacher_max"]
            and nonconst_fraction
            <= hyp["H2_diffusion_distillation"]["nonconstant_contribution_rms_fraction_max"]),
        "H3_truth_recovery": bool(h3_field and (coef_ok if results["collinearity"]
                                                ["coefficient_thresholds_apply"] else True)),
        "H4_simulation": bool(
            sim_summary["symbolic"]["failure_rate"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_max"]
            and sim_summary["symbolic"]["failure_rate"] - sim_summary["neural"]["failure_rate"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_minus_neural_max"]
            and rmse_sym_neural <= hyp["H4_simulation"]["median_paired_rmse_vs_neural_max"]),
        "H5_estimator_bias": bool(
            abs(v4["diffusion_mean_on_validation_states"] - 0.4)
            < abs(em["diffusion_mean_on_validation_states"] - 0.4)
            and v4["vs_truth_test"]["drift"] < em["vs_truth_test"]["drift"]),
    }
    results["hypothesis_verdicts"] = verdicts
    results["H3_coefficient_clause_applied"] = bool(
        results["collinearity"]["coefficient_thresholds_apply"])
    results["best_validation_nrmse_by_drift_degree"] = {str(k): v for k, v in best_val.items()}

    tex = ("%% Symbolic SDE distilled on Day 2 (Kidger et al. 2021 OU teacher, "
           "variables (t, z))\n%% drift\n"
           + tex_equations(drift_expanded, ("\\hat f",), VAR_NAMES)
           + "\n%% diffusion\n"
           + tex_equations(diffusion_expanded, ("\\hat g",), VAR_NAMES) + "\n")
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    results["symbolic_sde_tex"] = tex
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({"teacher_vs_truth_test": results["teacher_vs_truth_test"],
                      "symbolic_vs_teacher_test": results["symbolic_vs_teacher_test"],
                      "symbolic_vs_truth_test": results["symbolic_vs_truth_test"],
                      "collinearity": results["collinearity"],
                      "verdicts": verdicts}, indent=2))
    return results


if __name__ == "__main__":
    run()
