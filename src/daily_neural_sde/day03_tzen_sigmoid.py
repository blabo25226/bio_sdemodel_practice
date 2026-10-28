"""Day 3: Tzen & Raginsky (2019) sigmoid neural SDE -> SINDy distillation.

Pipeline (all settings frozen in ``configs/daily_neural_sde_day03.json``):

1. Regenerate the paper's Section 6 synthetic example, cut to d = 2:
   dX_t = sigmoid(A X_t) dt + dW_t on [0, 1], A drawn i.i.d. N(0, 1) with
   seed 0 (recorded in the config), X_0 ~ N(0, I_2). Data are generated
   by Euler-Maruyama at dt = 0.001 (the nonlinear drift has no exact
   transition) and observed every 0.05 (21 points per trajectory).
   The paper's experiment is d = 10, n = 1000; this study uses
   d = 2, n = 4096 trajectories (declared CPU cut).
2. Estimate the neural SDE in the paper's own parametric family
   (drift = sigmoid(A x), scalar diffusion g) with four Gaussian
   transition likelihoods (EM / local linearization /
   local-linearization-exponential / substep moments). The distilled
   teacher is the substep estimator (v4), fixed a priori per the Day 1
   reflection. The discretization-bias diagnostic is split by regime
   (near-linear vs saturated, defined by max|A x| at the transition
   start), the analogue of Day 1's linear-vs-nonlinear component split.
3. Separately, fit the paper's variational model: the same drift family
   plus a single shared constant Girsanov shift beta, optimized on
   train terminal observations Y = X_1 + N(0, I_2) by free-energy
   minimization (discretize-then-differentiate Euler simulation).
   A per-observation amortized encoder beta(y) and the paper's
   forward-mode pathwise differentiation are declared out of scope.
   The mesh study re-simulates the fitted variational model at
   h in {1/8, 1/16, 1/32, 1/64} (H6).
4. Distill the v4 teacher into polynomial SINDy models in (x1, x2)
   by direct function queries on trajectory-disjoint splits, in the
   conventions of ``src/distill.py`` (validation-only selection,
   test once). A collinearity diagnostic on train query states is
   computed before fitting (Day 2 reflection); an off-manifold grid
   diagnostic is standard (Day 2 reflection).
5. Validate by paired Euler-Maruyama simulation (shared Brownian
   increments) against the neural SDE and the ground truth,
   retaining failed-trajectory mass.

Run from the repository root:  ``python -m src.daily_neural_sde.day03_tzen_sigmoid``
Outputs land in ``outputs/daily_neural_sde/day03/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import (PolynomialField, evaluate_expanded, expand_to_original,
                     fit_polynomial_field, nrmse, simulate_em, tex_equations)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "daily_neural_sde_day03.json"
OUT_DIR = ROOT / "outputs" / "daily_neural_sde" / "day03"

VAR_NAMES = ("x1", "x2")
N_SUBSTEPS = 8


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Coordinatewise logistic sigmoid (numpy)."""
    return 1.0 / (1.0 + np.exp(-z))


def true_drift(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Ground-truth drift sigmoid(A x) for states (n, 2) -> (n, 2)."""
    a = np.asarray(cfg["ground_truth_sde"]["A"], dtype=np.float64)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return sigmoid(x @ a.T)


def true_diffusion(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Ground-truth constant diffusion (g = 1 per coordinate)."""
    n = len(np.atleast_2d(np.asarray(x, dtype=np.float64)))
    return np.ones((n, 2))


def generate_data(cfg: dict) -> np.ndarray:
    """Simulate the paper's synthetic data -> (n_traj, 21, 2)."""
    data_cfg = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_traj = data_cfg["n_trajectories"]
    dt_fine = data_cfg["dt_fine"]
    n_fine = int(round(data_cfg["t_end"] / dt_fine))
    n_obs = int(round(data_cfg["t_end"] / data_cfg["dt_obs"]))
    obs_every = int(round(data_cfg["dt_obs"] / dt_fine))
    x = rng.standard_normal((n_traj, 2))
    trajectories = np.empty((n_traj, n_obs + 1, 2))
    trajectories[:, 0] = x
    sqrt_dt = np.sqrt(dt_fine)
    for step in range(1, n_fine + 1):
        x = x + true_drift(x, cfg) * dt_fine + sqrt_dt * rng.standard_normal((n_traj, 2))
        if step % obs_every == 0:
            trajectories[:, step // obs_every] = x
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


def query_states(trajectories: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """All observed states for the given trajectories -> (n, 2)."""
    return trajectories[idx].reshape(-1, 2)


def transitions(trajectories: np.ndarray, idx: np.ndarray,
                dt_obs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-step observed transitions (start, end) for given trajectories."""
    block = trajectories[idx]
    return block[:, :-1].reshape(-1, 2), block[:, 1:].reshape(-1, 2)


def regime_of(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Regime label per state: 'near_linear', 'saturated', or 'intermediate'.

    Defined by max|A x| at the state (Day 1 reflection analogue for a
    sigmoid drift: the sigmoid is near-linear for small |A x| and
    saturated for large |A x|).
    """
    a = np.asarray(cfg["ground_truth_sde"]["A"], dtype=np.float64)
    u = np.abs(np.atleast_2d(x) @ a.T).max(axis=1)
    labels = np.full(len(u), "intermediate", dtype=object)
    labels[u < 0.5] = "near_linear"
    labels[u > 1.5] = "saturated"
    return labels


def _torch_transition_nll(estimator: str, x0: np.ndarray, x1: np.ndarray,
                          a_param, log_g_param, dt: float):
    """Gaussian transition NLL for one estimator (torch, J detached).

    ``a_param`` is the trainable (2, 2) matrix, ``log_g_param`` the
    trainable scalar log-diffusion. The Jacobian J = diag(sigmoid'(A x))
    A is treated as a fixed local coefficient (detached), as in Day 2.
    """
    import torch

    x0_t = torch.as_tensor(x0, dtype=torch.float64)
    x1_t = torch.as_tensor(x1, dtype=torch.float64)
    a_det = a_param.detach()
    u = x0_t @ a_det.T
    sig = torch.sigmoid(u)
    f = torch.sigmoid(x0_t @ a_param.T)
    sig_prime = sig * (1.0 - sig)
    jac = sig_prime[:, :, None] * a_det[None, :, :]  # (n, 2, 2), detached
    g = torch.exp(log_g_param)
    n = len(x0)
    eye = torch.eye(2, dtype=torch.float64)
    if estimator == "em":
        mean = x0_t + f * dt
        cov = (g ** 2 * dt) * eye[None].expand(n, 2, 2)
    elif estimator == "local_linearization":
        mean = x0_t + f * dt + 0.5 * dt ** 2 * torch.einsum("nij,nj->ni", jac, f)
        sym = 0.5 * (jac + jac.transpose(1, 2))
        cov = (g ** 2 * dt) * eye[None].expand(n, 2, 2) + (g ** 2 * dt ** 2) * sym
        eigval, eigvec = torch.linalg.eigh(cov)
        eigval = torch.clamp(eigval, min=1e-8)
        cov = (eigvec * eigval[:, None, :]) @ eigvec.transpose(1, 2)
    elif estimator == "local_linearization_exp":
        expm_jdt = torch.matrix_exp(jac * dt)
        psi = (expm_jdt - eye[None]) @ torch.linalg.pinv(jac)
        mean = x0_t + torch.einsum("nij,nj->ni", psi, f)
        nodes, weights = np.polynomial.legendre.leggauss(8)
        cov = torch.zeros((n, 2, 2), dtype=torch.float64)
        for node, weight in zip(nodes, weights):
            s = 0.5 * dt * (node + 1.0)
            e = torch.matrix_exp(jac * s)
            cov = cov + float(weight) * 0.5 * dt * (e @ e.transpose(1, 2))
        cov = (g ** 2) * cov
    elif estimator == "substep_moments":
        mean = x0_t
        cov = torch.zeros((n, 2, 2), dtype=torch.float64)
        h = dt / N_SUBSTEPS
        for _ in range(N_SUBSTEPS):
            f_m = torch.sigmoid(mean @ a_param.T)
            u_m = mean @ a_det.T
            sig_m = torch.sigmoid(u_m)
            jac_m = (sig_m * (1.0 - sig_m))[:, :, None] * a_det[None, :, :]
            cov = cov + (torch.einsum("nij,njk->nik", jac_m, cov)
                         + torch.einsum("nij,njk->nik", cov, jac_m.transpose(1, 2))
                         + (g ** 2) * eye[None]) * h
            mean = mean + f_m * h
    else:
        raise ValueError(f"unknown estimator {estimator}")
    diff = (x1_t - mean)[:, :, None]
    sign, logdet = torch.linalg.slogdet(cov)
    if bool((sign <= 0).any()):
        raise ValueError("non-positive-definite transition covariance")
    quad = torch.einsum("nij,njk,nk->n",
                        diff.transpose(1, 2), torch.linalg.inv(cov), diff[:, :, 0])
    return torch.mean(0.5 * (logdet + quad + 2.0 * np.log(2.0 * np.pi)))


def train_likelihood_teacher(estimator: str, train: tuple, val: tuple,
                             cfg: dict, epochs: int = 200) -> dict[str, Any]:
    """Fit one parametric likelihood teacher; checkpoint on val NLL."""
    import torch

    torch.manual_seed(123 + ["em", "local_linearization",
                             "local_linearization_exp",
                             "substep_moments"].index(estimator))
    dt = cfg["data"]["dt_obs"]
    a_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float64))
    log_g_param = torch.nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float64))
    opt = torch.optim.Adam([a_param, log_g_param], lr=0.01)
    best = {"nll": float("inf"), "A": None, "g": None}
    x0_tr, x1_tr = train
    x0_va, x1_va = val
    for epoch in range(epochs):
        opt.zero_grad()
        loss = _torch_transition_nll(estimator, x0_tr, x1_tr, a_param, log_g_param, dt)
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                val_nll = float(_torch_transition_nll(
                    estimator, x0_va, x1_va, a_param, log_g_param, dt))
            if val_nll < best["nll"]:
                best = {"nll": val_nll,
                        "A": a_param.detach().numpy().copy(),
                        "g": float(torch.exp(log_g_param).item())}
    return best


def teacher_drift(x: np.ndarray, teacher: dict) -> np.ndarray:
    """Parametric teacher drift sigmoid(A_hat x)."""
    a = np.asarray(teacher["A"], dtype=np.float64)
    return sigmoid(np.atleast_2d(x) @ a.T)


def teacher_diffusion(x: np.ndarray, teacher: dict) -> np.ndarray:
    """Parametric teacher diffusion (constant g_hat per coordinate)."""
    n = len(np.atleast_2d(x))
    return np.full((n, 2), teacher["g"])


def fit_variational(trajectories: np.ndarray, splits: dict,
                    cfg: dict, epochs: int = 300) -> dict[str, Any]:
    """Fit the shared-beta variational model on train terminal data.

    Free energy per observation (constant-excluded):
    F = E_paths[0.5 * ||beta||^2 + 0.5 * ||Y - X_1||^2], where Y is a
    train terminal observation (X_1 + N(0, I_2) noise) and paths follow
    the posterior SDE dX = (sigmoid(A X) + beta) dt + dW by Euler
    simulation at dt = 0.05. Checkpoint = min validation free energy
    on a fixed-seed simulation; the test free energy is reported once.
    """
    import torch

    rng = np.random.default_rng(5)
    noise = rng.standard_normal((cfg["data"]["n_trajectories"], 2))
    terminal = trajectories[:, -1] + noise  # Y per trajectory
    y_train = terminal[splits["train"]]
    y_val = terminal[splits["validation"]]
    y_test = terminal[splits["test"]]
    torch.manual_seed(0)
    a_init = torch.as_tensor(
        np.random.default_rng(1).standard_normal((2, 2)), dtype=torch.float64)
    a_param = torch.nn.Parameter(a_init.clone())
    beta_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = torch.optim.Adam([a_param, beta_param], lr=0.01)
    dt = cfg["teacher"]["variational"]["simulation_dt"]
    n_steps = int(round(1.0 / dt))

    def simulate_free_energy(y: np.ndarray, n_paths: int, seed: int,
                             a_t, beta_t, dt_sim: float):
        gen = torch.Generator().manual_seed(seed)
        steps = int(round(1.0 / dt_sim))
        x = torch.randn((n_paths, 2), dtype=torch.float64, generator=gen)
        inc = torch.randn((n_paths, steps, 2), dtype=torch.float64, generator=gen)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        idx = torch.randint(0, len(y), (n_paths,), generator=gen)
        for step in range(steps):
            x = x + (torch.sigmoid(x @ a_t.T) + beta_t) * dt_sim \
                + np.sqrt(dt_sim) * inc[:, step]
        resid = y_t[idx] - x
        per_path = 0.5 * (beta_t ** 2).sum() \
            + 0.5 * (resid ** 2).sum(dim=1)
        return per_path

    best = {"F_val": float("inf"), "A": None, "beta": None}
    for epoch in range(epochs):
        opt.zero_grad()
        per_path = simulate_free_energy(
            y_train, cfg["teacher"]["variational"]["path_batch"], 1000 + epoch,
            a_param, beta_param, dt)
        loss = per_path.mean()
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                f_val = simulate_free_energy(
                    y_val, 2048, 99, a_param, beta_param, dt).mean().item()
            if f_val < best["F_val"]:
                best = {"F_val": f_val,
                        "A": a_param.detach().numpy().copy(),
                        "beta": beta_param.detach().numpy().copy()}
    with torch.no_grad():
        a_t = torch.as_tensor(best["A"], dtype=torch.float64)
        beta_t = torch.as_tensor(best["beta"], dtype=torch.float64)
        f_train = simulate_free_energy(y_train, 4096, 7, a_t, beta_t, dt).mean().item()
        f_test = simulate_free_energy(y_test, 4096, 7, a_t, beta_t, dt).mean().item()
        mesh = {}
        for h in [1 / 8, 1 / 16, 1 / 32, 1 / 64]:
            vals = simulate_free_energy(y_val, 4096, 7, a_t, beta_t, h)
            mesh[str(h)] = {"mean": float(vals.mean()),
                            "se": float(vals.std() / np.sqrt(len(vals)))}
    const = float(2 / 2 * np.log(2.0 * np.pi))
    return {"A": best["A"], "beta": best["beta"],
            "F_val_excl": best["F_val"], "F_train_excl": f_train,
            "F_test_excl": f_test, "F_reported_const": const,
            "mesh": mesh,
            "y_noise_seed": 5}


def main() -> dict[str, Any]:
    """Run the Day 3 pipeline and write outputs + results.json."""
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories = generate_data(cfg)
    splits = split_indices(cfg)
    dt_obs = cfg["data"]["dt_obs"]

    # --- collinearity diagnostic on train query states (before fitting)
    train_q = query_states(trajectories, splits["train"])
    corr12 = float(np.corrcoef(train_q[:, 0], train_q[:, 1])[0, 1])
    vif = float(1.0 / max(1.0 - corr12 ** 2, 1e-12))
    from .common import monomial_powers, evaluate_powers
    powers2 = monomial_powers(2, 2)
    deg2_all = evaluate_powers(train_q, powers2)
    # Drop the constant feature (zero variance) before correlating.
    keep = np.any(powers2 != 0, axis=1)
    deg2 = deg2_all[:, keep]
    corr_mat = np.corrcoef(deg2.T)
    iu = np.triu_indices(len(corr_mat), k=1)
    max_deg2_corr = float(np.nanmax(np.abs(corr_mat[iu])))
    collinearity = {"corr_x1_x2_train": corr12, "vif_x1_x2_train": vif,
                    "max_pairwise_abs_corr_degree2_features_train": max_deg2_corr,
                    "coefficient_interpretation": "VIF > 10 bars symbolic "
                    "coefficient interpretation (Day 2 reflection)" if vif > 10
                    else "VIF <= 10; symbolic coefficients still carry no "
                    "A interpretation (polynomial != sigmoid)"}

    # --- likelihood teachers (v1-v4)
    train_tr = transitions(trajectories, splits["train"], dt_obs)
    val_tr = transitions(trajectories, splits["validation"], dt_obs)
    test_tr = transitions(trajectories, splits["test"], dt_obs)
    estimators: dict[str, Any] = {}
    for name in ["em", "local_linearization", "local_linearization_exp",
                 "substep_moments"]:
        fitted = train_likelihood_teacher(name, train_tr, val_tr, cfg)
        teacher = {"A": fitted["A"], "g": fitted["g"]}
        drift_nrmse_test = nrmse(true_drift(test_tr[0], cfg),
                                 teacher_drift(test_tr[0], teacher))
        drift_nrmse_test_by_regime = {}
        regimes = regime_of(test_tr[0], cfg)
        for regime in ["near_linear", "intermediate", "saturated"]:
            mask = regimes == regime
            if mask.sum() > 0:
                drift_nrmse_test_by_regime[regime] = {
                    "n": int(mask.sum()),
                    "drift_nrmse_vs_truth": nrmse(
                        true_drift(test_tr[0][mask], cfg),
                        teacher_drift(test_tr[0][mask], teacher))}
        estimators[name] = {
            "val_nll": fitted["nll"], "A": fitted["A"].tolist(),
            "g": fitted["g"],
            "g_absolute_error": abs(fitted["g"] - 1.0),
            "drift_nrmse_vs_truth_test": drift_nrmse_test,
            "drift_nrmse_vs_truth_test_by_regime": drift_nrmse_test_by_regime}
        print(f"[{name}] val NLL={fitted['nll']:.5f} g={fitted['g']:.4f} "
              f"drift NRMSE vs truth (test)={drift_nrmse_test:.4f}")
    teacher = {"A": np.asarray(estimators["substep_moments"]["A"]),
               "g": estimators["substep_moments"]["g"]}

    # --- variational fit (paper's inference family, shared beta)
    variational = fit_variational(trajectories, splits, cfg)
    a_var = np.asarray(variational["A"])
    a_true = np.asarray(cfg["ground_truth_sde"]["A"])
    variational["A_relative_frobenius_error"] = float(
        np.linalg.norm(a_var - a_true) / np.linalg.norm(a_true))
    variational["beta_norm"] = float(np.linalg.norm(variational["beta"]))
    print(f"[variational] A rel Frobenius error="
          f"{variational['A_relative_frobenius_error']:.4f} "
          f"|beta|={variational['beta_norm']:.4f}")

    # --- polynomial SINDy distillation of the v4 teacher
    queries = {split: query_states(trajectories, splits[split])
               for split in ["train", "validation", "test"]}
    teacher_drift_q = {s: teacher_drift(queries[s], teacher) for s in queries}
    teacher_diff_q = {s: teacher_diffusion(queries[s], teacher) for s in queries}
    truth_drift_q = {s: true_drift(queries[s], cfg) for s in queries}
    distilled: dict[str, Any] = {}
    for field, degrees, targets in [
            ("drift", cfg["distillation"]["drift_degrees"], teacher_drift_q),
            ("diffusion", cfg["distillation"]["diffusion_degrees"], teacher_diff_q)]:
        best_field = None
        per_degree = {}
        for degree in degrees:
            for threshold in cfg["distillation"]["thresholds"]:
                model = fit_polynomial_field(
                    queries["train"], targets["train"], degree, threshold,
                    ridge=cfg["distillation"]["ridge"])
                val_err = nrmse(targets["validation"], model.predict(queries["validation"]))
                if best_field is None or val_err < best_field["val_nrmse"]:
                    best_field = {"model": model, "degree": degree,
                                  "threshold": threshold, "val_nrmse": val_err}
            per_degree[str(degree)] = None
        model = best_field["model"]
        test_err = nrmse(targets["test"], model.predict(queries["test"]))
        truth_err = None
        if field == "drift":
            truth_err = nrmse(truth_drift_q["test"], model.predict(queries["test"]))
        expanded = expand_to_original(model)
        model.save(OUT_DIR / f"symbolic_{field}.json")
        if field == "drift":
            drift_expanded, drift_model = expanded, model
        else:
            diff_expanded, diff_model = expanded, model
        distilled[field] = {"degree": best_field["degree"],
                            "threshold": best_field["threshold"],
                            "active_terms": int(model.metadata["active_terms"]),
                            "val_nrmse_vs_teacher": best_field["val_nrmse"],
                            "test_nrmse_vs_teacher": test_err,
                            "test_nrmse_vs_truth": truth_err,
                            "expanded": [{str(k): v for k, v in poly.items()}
                                         for poly in expanded]}
        print(f"[{field}] degree={best_field['degree']} thr={best_field['threshold']} "
              f"test NRMSE vs teacher={test_err:.5f} truth={truth_err}")

    # diffusion nonconstant contribution on test queries
    diff_pred = diff_model.predict(queries["test"])
    const_part = np.full_like(diff_pred, teacher["g"])
    nonconst_rms = float(np.sqrt(np.mean((diff_pred - const_part) ** 2))
                         / max(np.sqrt(np.mean(diff_pred ** 2)), 1e-12))

    # oracle diagnostic: same SINDy procedure on ground-truth fields
    oracle = {}
    for field, degrees, targets in [
            ("drift", cfg["distillation"]["drift_degrees"], truth_drift_q),
            ("diffusion", cfg["distillation"]["diffusion_degrees"],
             {s: true_diffusion(queries[s], cfg) for s in queries})]:
        best_o = None
        for degree in degrees:
            for threshold in cfg["distillation"]["thresholds"]:
                model = fit_polynomial_field(
                    queries["train"], targets["train"], degree, threshold,
                    ridge=cfg["distillation"]["ridge"])
                val_err = nrmse(targets["validation"], model.predict(queries["validation"]))
                if best_o is None or val_err < best_o[0]:
                    best_o = (val_err, degree, threshold, model)
        oracle[field] = {"degree": best_o[1], "threshold": best_o[2],
                         "test_nrmse_vs_truth": nrmse(
                             targets["test"], best_o[3].predict(queries["test"]))}

    # --- off-manifold grid (Day 2 reflection: standard procedure)
    grid_vals = [-3.0, -1.5, 0.0, 1.5, 3.0]
    grid = np.array([[a, b] for a in grid_vals for b in grid_vals])
    off = {"drift_nrmse_symbolic_vs_teacher": nrmse(
               teacher_drift(grid, teacher), drift_model.predict(grid)),
           "drift_nrmse_symbolic_vs_truth": nrmse(
               true_drift(grid, cfg), drift_model.predict(grid)),
           "drift_nrmse_teacher_vs_truth": nrmse(
               true_drift(grid, cfg), teacher_drift(grid, teacher))}

    # --- paired simulation validation (common.simulate_em)
    sim_cfg = cfg["simulation"]
    rng = np.random.default_rng(sim_cfg["seed"])
    n_test = len(splits["test"])
    n_steps = int(round(sim_cfg["t_end"] / sim_cfg["dt"]))
    increments = rng.standard_normal((n_test, n_steps, 2))
    x0 = trajectories[splits["test"], 0]
    sims = {}
    for name, drift_fn, diff_fn in [
            ("neural", lambda x: teacher_drift(x, teacher),
             lambda x: teacher_diffusion(x, teacher)),
            ("symbolic", lambda x: drift_model.predict(x),
             lambda x: diff_model.predict(x)),
            ("truth", lambda x: true_drift(x, cfg),
             lambda x: true_diffusion(x, cfg))]:
        sims[name] = simulate_em(drift_fn, diff_fn, x0, sim_cfg["dt"],
                                 n_steps, increments,
                                 sim_cfg["failure_radius"])
    def paired_rmse(a_states, b_states):
        both_ok = ~(np.isnan(a_states[:, -1]).any(axis=1)
                    | np.isnan(b_states[:, -1]).any(axis=1))
        diffs = a_states[both_ok, -1] - b_states[both_ok, -1]
        per_path = np.sqrt(np.mean(diffs ** 2, axis=1))
        return float(np.median(per_path)), int(both_ok.sum())
    sim_med, n_paired = paired_rmse(sims["symbolic"]["states"], sims["neural"]["states"])
    simulation = {
        "failure_rate": {k: v["failure_rate"] for k, v in sims.items()},
        "failure_class_counts": {
            k: {cls: int(sum(1 for c in v["failure_class"] if c == cls))
                for cls in ["non_finite", "outside_radius"]}
            for k, v in sims.items()},
        "terminal_mean": {k: [float(v) for v in np.nanmean(
            sims[k]["states"][:, -1], axis=0)] for k in sims},
        "terminal_std": {k: [float(v) for v in np.nanstd(
            sims[k]["states"][:, -1], axis=0)] for k in sims},
        "paired_rmse_median_symbolic_vs_neural": sim_med,
        "n_paired": n_paired}

    # --- verdicts (frozen thresholds in the config)
    hyp = cfg["hypotheses"]
    mesh = variational["mesh"]
    mesh_change = abs(mesh["0.015625"]["mean"] - mesh["0.03125"]["mean"])
    verdicts = {
        "H1_drift_distillation": bool(
            distilled["drift"]["test_nrmse_vs_teacher"]
            <= hyp["H1_drift_distillation"]["symbolic_drift_test_nrmse_vs_teacher_max"]),
        "H2_diffusion_distillation": bool(
            distilled["diffusion"]["test_nrmse_vs_teacher"]
            <= hyp["H2_diffusion_distillation"]["symbolic_diffusion_test_nrmse_vs_teacher_max"]
            and nonconst_rms
            <= hyp["H2_diffusion_distillation"]["nonconstant_contribution_rms_fraction_max"]),
        "H3_truth_recovery": bool(
            distilled["drift"]["test_nrmse_vs_truth"]
            <= hyp["H3_truth_recovery"]["field_symbolic_drift_test_nrmse_vs_truth_max"]
            and float(np.linalg.norm(
                np.asarray(estimators["substep_moments"]["A"]) - a_true)
                / np.linalg.norm(a_true))
            <= hyp["H3_truth_recovery"]["teacher_A_relative_frobenius_error_max"]),
        "H4_simulation": bool(
            simulation["failure_rate"]["symbolic"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_max"]
            and simulation["failure_rate"]["symbolic"]
            - simulation["failure_rate"]["neural"]
            <= hyp["H4_simulation"]["symbolic_failure_rate_minus_neural_max"]
            and sim_med <= hyp["H4_simulation"]["median_paired_rmse_vs_neural_max"]),
        "H5_estimator_bias": bool(
            estimators["substep_moments"]["g_absolute_error"]
            < estimators["em"]["g_absolute_error"]
            and estimators["substep_moments"]["drift_nrmse_vs_truth_test"]
            < estimators["em"]["drift_nrmse_vs_truth_test"]),
        "H6_variational_mesh": bool(
            mesh_change <= hyp["H6_variational_mesh"]["free_energy_absolute_change_max"])}
    print("verdicts:", verdicts)

    tex = tex_equations(drift_expanded, ("\\hat f_1", "\\hat f_2"), VAR_NAMES) \
        + "\n" + tex_equations(diff_expanded, ("\\hat g_1", "\\hat g_2"), VAR_NAMES)
    (OUT_DIR / "symbolic_sde.tex").write_text(tex + "\n")

    results = {"teacher_vs_truth_test": {
                   "drift": estimators["substep_moments"]["drift_nrmse_vs_truth_test"],
                   "diffusion": abs(estimators["substep_moments"]["g"] - 1.0)},
               "symbolic_vs_teacher_test": {
                   "drift": distilled["drift"]["test_nrmse_vs_teacher"],
                   "diffusion": distilled["diffusion"]["test_nrmse_vs_teacher"]},
               "symbolic_vs_truth_test": {
                   "drift": distilled["drift"]["test_nrmse_vs_truth"],
                   "diffusion": None},
               "estimators": estimators,
               "variational": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in variational.items()},
               "distillation": distilled,
               "oracle": oracle,
               "collinearity": collinearity,
               "diffusion_nonconstant_rms_fraction": nonconst_rms,
               "off_manifold_grid": off,
               "simulation": simulation,
               "mesh_change_1_32_to_1_64": mesh_change,
               "verdicts": verdicts}
    (OUT_DIR / "teacher_weights.json").write_text(json.dumps(
        {"likelihood_teachers": {k: {"A": v["A"], "g": v["g"]}
                                 for k, v in estimators.items()},
         "variational": {"A": np.asarray(variational["A"]).tolist(),
                         "beta": np.asarray(variational["beta"]).tolist()}},
        indent=2))
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
