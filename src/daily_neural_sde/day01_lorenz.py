"""Day 1: Li et al. (2020) stochastic Lorenz attractor -> SINDy distillation.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day01.json``):

1. Regenerate the paper's synthetic dataset (Appendix 9.10.2): 1024
   trajectories of the stochastic Lorenz SDE, observed every 0.025 on
   [0, 1], corrupted by Gaussian observation noise with std 0.01.
2. Estimate a neural SDE *in the observed 3D coordinates* (declared
   deviation from the paper's 4D latent SDE; see the config): separate
   MLPs for drift and diagonal diffusion, trained with the Gaussian
   Euler-Maruyama transition likelihood (observation-noise corrected).
3. Distill the teacher into polynomial SINDy models by direct function
   queries on trajectory-disjoint splits, in the conventions of
   ``src/distill.py`` (validation-only selection, test once).
4. Validate by paired Euler-Maruyama simulation (shared Brownian
   increments) against the neural SDE and the ground truth, retaining
   failed-trajectory mass.

Run from the repository root:  ``python -m src.daily_neural_sde.day01_lorenz``
Outputs land in ``outputs/daily_neural_sde/day01/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, evaluate_expanded, expand_to_original,
                     fit_polynomial_field, nrmse, simulate_em, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day01.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day01"

VAR_NAMES = ("x", "y", "z")
TRUE_DRIFT_SUPPORT: list[dict[tuple[int, ...], float]] = [
    {(1, 0, 0): -10.0, (0, 1, 0): 10.0},
    {(1, 0, 0): 28.0, (0, 1, 0): -1.0, (1, 0, 1): -1.0},
    {(1, 1, 0): 1.0, (0, 0, 1): -8.0 / 3.0},
]


def true_drift(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Ground-truth Lorenz drift for states (n, 3) -> (n, 3)."""
    gt = cfg["ground_truth_sde"]
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.stack([gt["sigma"] * (x[:, 1] - x[:, 0]),
                     x[:, 0] * (gt["rho"] - x[:, 2]) - x[:, 1],
                     x[:, 0] * x[:, 1] - gt["beta"] * x[:, 2]], axis=1)


def true_diffusion(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Ground-truth diagonal diffusion (constant) for states (n, 3)."""
    alpha = np.asarray(cfg["ground_truth_sde"]["alpha"], dtype=np.float64)
    return np.tile(alpha, (len(np.atleast_2d(x)), 1))


def generate_data(cfg: dict) -> np.ndarray:
    """Simulate the paper's dataset -> noisy trajectories (n_traj, 41, 3)."""
    data_cfg = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_traj = data_cfg["n_trajectories"]
    dt_sim = data_cfg["dt_sim"]
    steps_per_obs = int(round(data_cfg["dt_obs"] / dt_sim))
    n_obs_steps = int(round(data_cfg["t_end"] / data_cfg["dt_obs"]))
    x = rng.standard_normal((n_traj, 3))
    trajectories = np.empty((n_traj, n_obs_steps + 1, 3))
    trajectories[:, 0] = x
    sqrt_dt = np.sqrt(dt_sim)
    alpha = np.asarray(cfg["ground_truth_sde"]["alpha"])
    for obs in range(n_obs_steps):
        for _ in range(steps_per_obs):
            drift = true_drift(x, cfg)
            x = x + drift * dt_sim + alpha * sqrt_dt * rng.standard_normal((n_traj, 3))
        trajectories[:, obs + 1] = x
    noisy = trajectories + rng.normal(0.0, data_cfg["obs_noise_std"], trajectories.shape)
    return noisy


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


def _softplus(v: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, v)


class NumpyMLP:
    """NumPy forward pass of a 2-hidden-layer Softplus MLP (exported weights)."""

    def __init__(self, layers: list[dict[str, Any]], output_transform: str) -> None:
        self.layers = [(np.asarray(l["weight"]), np.asarray(l["bias"])) for l in layers]
        self.output_transform = output_transform

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Standardized states (n, 3) -> outputs (n, 3)."""
        h = np.asarray(z, dtype=np.float64)
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

    def drift(self, x: np.ndarray) -> np.ndarray:
        """Teacher drift for states (n, 3) -> (n, 3)."""
        return self.drift_net((np.atleast_2d(x) - self.mean) / self.scale)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """Teacher diagonal diffusion for states (n, 3) -> (n, 3)."""
        return self.diffusion_net((np.atleast_2d(x) - self.mean) / self.scale)


def train_teacher(trajectories: np.ndarray, splits: dict[str, np.ndarray],
                  cfg: dict, estimator: str = "local_linearization") -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the neural SDE teacher with PyTorch; return (weights, log).

    ``estimator`` is ``"em"`` (plain Gaussian Euler-Maruyama transition
    likelihood, retained as the v1 baseline) or ``"local_linearization"``
    (v2: the transition mean gains the 0.5 * dt^2 * (J f) local-linearization
    correction and the variance gains the (dt^3 / 3) * sum_j (df_i/dx_j)^2
    g_j^2 noise-induced drift-variation term; the Jacobian is treated as a
    fixed local coefficient within each step). The plain EM likelihood is
    biased at the paper's observation interval dt = 0.025 for the nonlinear
    Lorenz drift: it attributes drift curvature to the diffusion.
    """
    import torch

    torch.manual_seed(cfg["seed"])
    teacher_cfg = cfg["teacher"]
    dt = cfg["data"]["dt_obs"]
    obs_var = 2.0 * cfg["data"]["obs_noise_std"] ** 2

    def transitions(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        idx = splits[name]
        block = trajectories[idx]
        x_t = torch.tensor(block[:, :-1].reshape(-1, 3), dtype=torch.float32)
        dx = torch.tensor((block[:, 1:] - block[:, :-1]).reshape(-1, 3), dtype=torch.float32)
        return x_t, dx

    train_x, train_dx = transitions("train")
    val_x, val_dx = transitions("validation")
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)

    def make_net() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.Softplus(),
            torch.nn.Linear(64, 64), torch.nn.Softplus(),
            torch.nn.Linear(64, 3))

    drift_net = make_net()
    diffusion_net = make_net()
    params = list(drift_net.parameters()) + list(diffusion_net.parameters())
    optimizer = torch.optim.Adam(params, lr=teacher_cfg["optimizer_lr"])

    def nll(x_t: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        z = (x_t - mean) / std
        drift = drift_net(z)
        diffusion = torch.nn.functional.softplus(diffusion_net(z)) + 0.01
        if estimator == "em":
            var = diffusion ** 2 * dt + obs_var
            resid = dx - drift * dt
        elif estimator == "local_linearization":
            with torch.no_grad():
                jac_z = torch.func.vmap(torch.func.jacrev(lambda zz: drift_net(zz)))(z)
            jac_x = jac_z / std[None, None, :]
            mean_step = drift * dt + 0.5 * dt ** 2 * torch.einsum("nij,nj->ni", jac_x, drift)
            var = (diffusion ** 2 * dt + obs_var
                   + (dt ** 3 / 3.0) * torch.einsum("nij,nj->ni", jac_x ** 2, diffusion ** 2))
            resid = dx - mean_step
        else:  # local_linearization_exp (v3, the distilled teacher)
            with torch.no_grad():
                jac_z = torch.func.vmap(torch.func.jacrev(lambda zz: drift_net(zz)))(z)
                jac_x = jac_z / std[None, None, :]
                n_batch = z.shape[0]
                augmented = torch.zeros(n_batch, 6, 6, dtype=z.dtype)
                augmented[:, :3, :3] = jac_x * dt
                augmented[:, :3, 3:] = torch.eye(3, dtype=z.dtype) * dt
                psi = torch.linalg.matrix_exp(augmented)[:, :3, 3:]
                diag = torch.diagonal(jac_x, dim1=1, dim2=2)
                kappa = torch.where(torch.abs(diag) > 1e-8,
                                    (torch.exp(2 * diag * dt) - 1.0) / (2 * diag),
                                    torch.full_like(diag, dt))
                cross = (dt ** 3 / 3.0) * torch.einsum("nij,nj->ni", jac_x ** 2, diffusion.detach() ** 2)
                cross = cross - (dt ** 3 / 3.0) * diag ** 2 * diffusion.detach() ** 2
            mean_step = torch.einsum("nij,nj->ni", psi, drift)
            var = kappa * diffusion ** 2 + cross + obs_var
            resid = dx - mean_step
        return (0.5 * (torch.log(2 * np.pi * var) + resid ** 2 / var)).mean()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    batch = teacher_cfg["batch_transitions"]
    for epoch in range(teacher_cfg["epochs"]):
        drift_net.train()
        diffusion_net.train()
        perm = torch.randperm(len(train_x))
        for start in range(0, len(train_x), batch):
            sel = perm[start:start + batch]
            optimizer.zero_grad()
            loss = nll(train_x[sel], train_dx[sel])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = float(nll(val_x, val_dx))
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
        test_x, test_dx = transitions("test")
        log = {"estimator": estimator,
               "best_epoch": best_epoch, "best_validation_nll": best_val,
               "final_train_nll": float(nll(train_x, train_dx)),
               "test_nll_at_best_checkpoint": float(nll(test_x, test_dx)),
               "epochs": teacher_cfg["epochs"],
               "n_train_transitions": int(len(train_x)),
               "torch_version": torch.__version__}
        # Restore best weights for the reported test NLL.
        drift_net.load_state_dict({k: torch.tensor(v) for k, v in best_state["drift"].items()})
        diffusion_net.load_state_dict({k: torch.tensor(v) for k, v in best_state["diffusion"].items()})
        log["test_nll_at_best_checkpoint"] = float(nll(test_x, test_dx))
        log["final_train_nll"] = float(nll(train_x, train_dx))
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


def _contribution_fraction(expanded: list[dict[tuple[int, ...], float]],
                           keep: Any, x: np.ndarray, full: np.ndarray) -> float:
    """RMS fraction of the field carried by terms where ``keep(exp)`` is False."""
    rest = [{exp: coef for exp, coef in poly.items() if not keep(exp)} for poly in expanded]
    part = evaluate_expanded(rest, x)
    return float(np.sqrt(np.mean(part ** 2) / max(np.mean(full ** 2), 1e-15)))


def run() -> dict[str, Any]:
    """Execute the full Day 1 pipeline and write all artifacts."""
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"day": 1, "date": cfg["date"], "paper": cfg["paper"],
                               "config_path": "configs/daily_neural_sde_day01.json"}

    print("[1/6] generating data (paper Appendix 9.10.2 setting)")
    trajectories = generate_data(cfg)
    splits = split_indices(cfg)
    flat = trajectories.reshape(-1, 3)
    results["data"] = {
        "shape": list(trajectories.shape),
        "state_mean": flat.mean(axis=0).tolist(),
        "state_std": flat.std(axis=0).tolist(),
        "max_state_norm": float(np.max(np.linalg.norm(flat, axis=1))),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "note": cfg["data"]["note"]}

    print("[2/6] training neural SDE teachers (PyTorch, CPU): v1 EM, v2 local-linearization, v3 LL-exponential")
    states = {name: trajectories[idx].reshape(-1, 3) for name, idx in splits.items()}
    truth_for_compare = {name: {"drift": true_drift(s, cfg), "diffusion": true_diffusion(s, cfg)}
                         for name, s in states.items()}
    estimator_comparison: dict[str, Any] = {}
    payload = None
    teacher_log: dict[str, Any] = {}
    for estimator in ("em", "local_linearization", "local_linearization_exp"):
        est_payload, est_log = train_teacher(trajectories, splits, cfg, estimator)
        est_teacher = NumpyTeacher(est_payload)
        g_val = est_teacher.diffusion(states["validation"])
        estimator_comparison[estimator] = {
            "training": est_log,
            "vs_truth_test": {
                field: nrmse(truth_for_compare["test"][field],
                             getattr(est_teacher, field)(states["test"]))
                for field in ("drift", "diffusion")},
            "diffusion_mean_on_validation_states": g_val.mean(axis=0).tolist(),
            "diffusion_std_on_validation_states": g_val.std(axis=0).tolist()}
        if estimator == "local_linearization_exp":
            payload, teacher_log = est_payload, est_log
    assert payload is not None
    results["estimator_comparison"] = estimator_comparison
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(payload, indent=1))
    teacher = NumpyTeacher(payload)
    results["teacher_training"] = teacher_log
    results["teacher_deviation"] = cfg["teacher"]["deviation_from_paper"]

    teacher_targets = {name: {"drift": teacher.drift(s), "diffusion": teacher.diffusion(s)}
                       for name, s in states.items()}
    truth_targets = truth_for_compare
    results["teacher_vs_truth_test"] = {
        field: nrmse(truth_targets["test"][field], teacher_targets["test"][field])
        for field in ("drift", "diffusion")}

    print("[3/6] polynomial SINDy distillation (validation-selected)")
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

    print("[4/6] oracle diagnostic (SINDy fitted to ground-truth fields)")
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

    print("[5/6] coefficient recovery in original coordinates")
    drift_expanded = expand_to_original(models["drift"])
    diffusion_expanded = expand_to_original(models["diffusion"])
    recovery = []
    for i, support in enumerate(TRUE_DRIFT_SUPPORT):
        for exp, truth in support.items():
            fitted = drift_expanded[i].get(exp, 0.0)
            recovery.append({"output": i, "power": list(exp), "truth": truth,
                             "fitted": fitted,
                             "relative_error": float(abs(fitted - truth) / abs(truth)),
                             "sign_match": bool(np.sign(fitted) == np.sign(truth))})
    true_support_sets = [set(support) for support in TRUE_DRIFT_SUPPORT]
    spurious_part = [
        {exp: coef for exp, coef in poly.items() if exp not in true_support_sets[i]}
        for i, poly in enumerate(drift_expanded)]
    spurious_values = evaluate_expanded(spurious_part, states["test"])
    full_values = models["drift"].predict(states["test"])
    spurious_fraction = float(np.sqrt(np.mean(spurious_values ** 2)
                                      / np.mean(full_values ** 2)))
    nonconst_fraction = _contribution_fraction(
        diffusion_expanded, lambda exp: sum(exp) == 0, states["test"],
        models["diffusion"].predict(states["test"]))
    test_states = states["test"]
    results["recovery"] = {"monomials": recovery,
                           "feature_correlation_x_xz_test": float(np.corrcoef(
                               test_states[:, 0], test_states[:, 0] * test_states[:, 2])[0, 1]),
                           "test_state_mean": test_states.mean(axis=0).tolist(),
                           "test_state_std": test_states.std(axis=0).tolist(),
                           "spurious_rms_fraction_test": spurious_fraction,
                           "diffusion_nonconstant_rms_fraction_test": nonconst_fraction,
                           "drift_expanded": [{str(k): v for k, v in poly.items()}
                                              for poly in drift_expanded],
                           "diffusion_expanded": [{str(k): v for k, v in poly.items()}
                                                  for poly in diffusion_expanded]}

    print("[6/6] paired simulation validation")
    sim_cfg = cfg["simulation"]
    test_traj = trajectories[splits["test"]]
    x0 = test_traj[:, 0]
    n_steps = int(round(sim_cfg["t_end"] / sim_cfg["dt"]))
    obs_every = int(round(cfg["data"]["dt_obs"] / sim_cfg["dt"]))
    rng = np.random.default_rng(sim_cfg["seed"])
    increments = rng.standard_normal((len(x0), n_steps, 3))
    sims = {
        "neural": simulate_em(teacher.drift, teacher.diffusion, x0, sim_cfg["dt"],
                              n_steps, increments, sim_cfg["failure_radius"], obs_every),
        "symbolic": simulate_em(models["drift"].predict, models["diffusion"].predict,
                                x0, sim_cfg["dt"], n_steps, increments,
                                sim_cfg["failure_radius"], obs_every),
        "truth": simulate_em(lambda s: true_drift(s, cfg),
                             lambda s: true_diffusion(s, cfg),
                             x0, sim_cfg["dt"], n_steps, increments,
                             sim_cfg["failure_radius"], obs_every),
    }

    def paired_rmse(a: dict, b: dict) -> tuple[float, int]:
        both = (~a["failed"]) & (~b["failed"])
        if both.sum() == 0:
            return float("nan"), 0
        diff = a["states"][both] - b["states"][both]
        per_traj = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
        return float(np.median(per_traj)), int(both.sum())

    sim_summary: dict[str, Any] = {}
    for name, sim in sims.items():
        final = sim["states"][:, -1]
        ok = ~sim["failed"]
        sim_summary[name] = {
            "failure_rate": sim["failure_rate"],
            "failure_classes": {c: sim["failure_class"].count(c)
                                for c in ("non_finite", "outside_radius")},
            "endpoint_mean_finishers": np.nanmean(final[ok], axis=0).tolist(),
            "endpoint_std_finishers": np.nanstd(final[ok], axis=0).tolist(),
            "n_finishers": int(ok.sum())}
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
    verdicts = {
        "H1_drift_distillation": bool(
            results["symbolic_vs_teacher_test"]["drift"]
            <= hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"]
            and best_val[2] <= best_val[3] * (1 + hyp["H1_drift_distillation"]
                                              ["degree2_within_relative_of_degree3_validation_nrmse"])),
        "H2_diffusion_distillation": bool(
            results["symbolic_vs_teacher_test"]["diffusion"]
            <= hyp["H2_diffusion_distillation"]["symbolic_diffusion_test_nrmse_vs_teacher_max"]
            and nonconst_fraction
            <= hyp["H2_diffusion_distillation"]["nonconstant_contribution_rms_fraction_max"]),
        "H3_truth_recovery": bool(
            all(r["sign_match"] and r["relative_error"]
                <= hyp["H3_truth_recovery"]["true_monomial_relative_error_max"]
                for r in recovery)
            and spurious_fraction
            <= hyp["H3_truth_recovery"]["spurious_contribution_rms_fraction_max"]),
        "H4_simulation": bool(
            sim_summary["symbolic"]["failure_rate"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_max"]
            and sim_summary["symbolic"]["failure_rate"] - sim_summary["neural"]["failure_rate"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_minus_neural_max"]
            and rmse_sym_neural <= hyp["H4_simulation"]["median_paired_rmse_vs_neural_max"]),
    }
    results["hypothesis_verdicts"] = verdicts
    results["best_validation_nrmse_by_drift_degree"] = {str(k): v for k, v in best_val.items()}

    tex = ("%% Symbolic SDE distilled on Day 1 (Li et al. 2020 Lorenz teacher)\n"
           "%% drift\n" + tex_equations(drift_expanded, ("\\hat f_1", "\\hat f_2", "\\hat f_3"))
           + "\n%% diagonal diffusion\n"
           + tex_equations(diffusion_expanded, ("\\hat g_1", "\\hat g_2", "\\hat g_3")) + "\n")
    (OUT_DIR / "symbolic_sde.tex").write_text(tex)
    results["symbolic_sde_tex"] = tex
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({"teacher_vs_truth_test": results["teacher_vs_truth_test"],
                      "symbolic_vs_teacher_test": results["symbolic_vs_teacher_test"],
                      "symbolic_vs_truth_test": results["symbolic_vs_truth_test"],
                      "verdicts": verdicts}, indent=2))
    return results


if __name__ == "__main__":
    run()
