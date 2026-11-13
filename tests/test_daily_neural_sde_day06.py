"""Day 6 tests: Eggen & Midtfjord (2023) Delay-SDE-Net SDDE (numpy only).

These tests do not train the MLP teachers; they check the
ground-truth field identities, the data generator, the query/lag
construction, and the SDDE simulator that the Day 6 pipeline relies on.
"""
import json
from pathlib import Path

import numpy as np

from src.daily_neural_sde.day06_eggen_delay import (
    build_queries, build_transitions, generate_data, history_values,
    simulate_sdde, true_drift_from_u, true_g_from_u0, true_tanh_args)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "daily_neural_sde_day06.json").read_text())


def _small_cfg():
    small = dict(CFG)
    small["data"] = dict(CFG["data"], n_trajectories=16)
    return small


def test_true_drift_and_diffusion_at_known_inputs():
    # u = [t, X1 lags, X2 lags]; all lags zero, t = 0 -> drift zero,
    # g^1 = 4 * sigmoid(0) = 2, g^2 = (1/8) * sigmoid(1).
    u = np.zeros((1, 9))
    assert np.allclose(true_drift_from_u(u), 0.0)
    g = true_g_from_u0(u)
    assert np.allclose(g[0, 0], 2.0)
    assert np.allclose(g[0, 1], 0.125 / (1.0 + np.exp(-1.0)))
    # tanh args are zero at the zero input.
    assert np.allclose(true_tanh_args(u), 0.0)


def test_generate_data_shape_and_initial_condition():
    data = generate_data(_small_cfg())
    assert data["obs"].shape == (16, 161, 2)
    assert np.isfinite(data["obs"]).all()
    assert (data["obs"][:, 0, 0] == 0.0).all()
    assert (data["obs"][:, 0, 1] == 1.0).all()
    assert data["init_lag"].shape == (16, 8)


def test_query_lags_use_history_before_zero():
    data = generate_data(_small_cfg())
    q = build_queries(data["obs"], data["z"], data["init_lag"], 0.05)
    # Query rows are time-major: row 0 is (t = 0, path 0). Its lag
    # block must equal the analytic history at -4..-1.
    hist = history_values(data["z"][:1], np.array([-4.0, -3.0, -2.0, -1.0]))[0]
    assert np.allclose(q["aug"][0, 2:6], hist[:, 0])
    assert np.allclose(q["aug"][0, 6:10], hist[:, 1])
    # Initial lag vector must match the same history values.
    assert np.allclose(data["init_lag"][0, :4], hist[:, 0])
    assert np.allclose(data["init_lag"][0, 4:], hist[:, 1])


def test_transition_alignment():
    data = generate_data(_small_cfg())
    q = build_queries(data["obs"], data["z"], data["init_lag"], 0.05)
    tr = build_transitions(q, data["obs"], 0.05)
    n_paths = 16
    assert len(tr["cur"]) == n_paths * 160
    assert np.allclose(tr["cur"][:n_paths], data["obs"][:, 0, :])
    assert np.allclose(tr["nxt"][:n_paths], data["obs"][:, 1, :])


def test_simulate_sdde_truth_stays_finite_and_classifies_failures():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((8, 2))
    inc = rng.standard_normal((8, 160, 2))
    sim = simulate_sdde(lambda u: true_drift_from_u(u),
                        lambda u: true_g_from_u0(u),
                        z, 0.05, 160, inc, 100.0, 50.0)
    assert sim["failure_rate"] == 0.0
    assert np.isfinite(sim["states"][:, 0, :]).all()
    # A tiny failure radius forces outside_radius failures.
    sim2 = simulate_sdde(lambda u: true_drift_from_u(u),
                         lambda u: true_g_from_u0(u),
                         z, 0.05, 160, inc, 0.5, 0.25)
    assert sim2["failure_rate"] > 0.0
    assert set(c for c in sim2["failure_class"] if c) <= {"outside_radius"}
