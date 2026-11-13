"""Day 21 tests: Wang et al. (ICLR 2024) SCOTCH utilities.

These tests do not train the teacher. They check the cubic chain, the graph
gate, failure accounting, shapes, and the frozen hypothesis gates.
"""
import math

import numpy as np
import torch

from src.daily_neural_sde.common import monomial_powers, nrmse
from src.daily_neural_sde.day21_wang_scotch import (
    GraphGatedField, ScotchTeacher, decide, field_np, load_config, partial_tensor,
    signature_partials, simulate_em_diag, truth_diffusion, truth_drift, truth_graph, vif_max)

CFG = load_config()


def _row(nrmse_value: float = 0.1, floor: float = 0.4) -> dict:
    return {"nrmse": nrmse_value, "floor": floor}


def _metrics(**overrides) -> dict:
    base = {
        "drift": [_row(), _row(), _row()],
        "diffusion": [_row(), _row(), _row()],
        "diffusion_nonpositive_fraction": 0.0,
        "vif": 4.0,
        "fail_teacher": 0.0,
        "fail_symbolic": 0.0,
        "path_rmse": 0.1,
        "law_floor": 0.05,
        "law_symbolic": 0.08,
        "law_fail_teacher": 0.0,
        "law_fail_symbolic": 0.0,
        "law_fail_floor": 0.0,
        "grid_nrmse": 0.2,
        "teacher_nonparent_rms": 0.0,
        "teacher_parent_rms": 0.4,
        "symbolic_relative_nonparent": 0.05,
        "parent_library_nrmse": 0.12,
        "full_library_nrmse": 0.10,
        "teacher_vs_truth_drift": 0.2,
        "symbolic_vs_truth_drift": 0.22,
        "teacher_vs_truth_diffusion": 0.05,
        "symbolic_vs_truth_diffusion": 0.06,
    }
    base.update(overrides)
    return base


def test_frozen_thresholds_match_the_pre_registration():
    assert CFG["paper"]["venue"] == "ICLR"
    assert CFG["paper"]["year"] == 2024
    assert CFG["truth"]["cubic_x"] == -0.2
    assert CFG["truth"]["cubic_z"] == -0.15
    assert CFG["truth"]["sigma"] == [0.4, 0.4, 0.4]
    assert CFG["truth"]["graph"] == [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
    assert CFG["data"]["horizon"] == 4.0
    assert CFG["data"]["dt_obs"] == 0.2
    assert CFG["data"]["failure_radius"] == 8.0
    assert CFG["model"]["graph_fixed_at_truth"] is True
    assert CFG["model"]["diffusion_head"] == "sigmoid"
    assert CFG["hypotheses"]["H1_drift"]["symbolic_vs_teacher_test_nrmse_max"] == 0.25
    assert CFG["hypotheses"]["H3_collinearity_gate"]["vif_max"] == 10.0
    assert CFG["hypotheses"]["H7_signature"]["teacher_nonparent_partial_rms_max"] == 1e-6
    assert CFG["hypotheses"]["H8_parent_library"]["gap_max"] == 0.05
    assert CFG["hypotheses"]["H9_truth"]["drift_nrmse_max"] == 0.45
    assert CFG["interpretation"]["drift_teacher_is_neural"] is True
    assert CFG["interpretation"]["diffusion_teacher_is_neural"] is True
    assert CFG["interpretation"]["coefficients_are_closed_form"] is False
    assert CFG["interpretation"]["distillation_discovers_true_sde"] is False
    assert CFG["interpretation"]["coordinate_is_gene_regulation"] is False
    seeds = CFG["seeds"]
    assert len({seeds["law_a"], seeds["law_b"], seeds["law_symbolic"], seeds["paired"]}) == 4


def test_cubic_chain_respects_the_signature_graph():
    graph = truth_graph(CFG)
    rng = np.random.default_rng(0)
    states = rng.normal(size=(30, 3))
    grads = partial_tensor(lambda z: truth_drift(z, CFG), states, 1e-5)
    parts = signature_partials(grads, graph)
    assert parts["nonparent_rms"] < 1e-8
    assert parts["parent_rms"] > 0.1
    # f1 has no z, f2 has no x, f3 has no y.
    assert np.max(np.abs(grads[:, 0, 2])) < 1e-8
    assert np.max(np.abs(grads[:, 1, 0])) < 1e-8
    assert np.max(np.abs(grads[:, 2, 1])) < 1e-8
    diffusion = truth_diffusion(states, CFG)
    assert diffusion.shape == (30, 3)
    assert np.allclose(diffusion, 0.4)


def test_gated_network_shapes_and_blocked_partials():
    torch.manual_seed(0)
    graph = torch.as_tensor(truth_graph(CFG), dtype=torch.float64)
    field = GraphGatedField(3, 8, 8, graph, sigmoid_head=False).to(dtype=torch.float64)
    diffusion = GraphGatedField(3, 8, 8, graph.clone(), sigmoid_head=True).to(dtype=torch.float64)
    states = np.linspace(-0.5, 0.5, 12).reshape(4, 3)
    drift = field_np(field, states)
    gate = field_np(diffusion, states)
    assert drift.shape == (4, 3)
    assert gate.shape == (4, 3)
    assert np.isfinite(drift).all()
    assert np.isfinite(gate).all()
    assert np.all(gate > 0.0) and np.all(gate < 1.0)
    grads = partial_tensor(lambda z: field_np(field, z), states, 1e-5)
    parts = signature_partials(grads, truth_graph(CFG))
    assert parts["nonparent_rms"] < 1e-8


def test_library_size_and_vif_excludes_the_constant():
    powers = monomial_powers(3, 3)
    assert len(powers) == 20
    rng = np.random.default_rng(1)
    states = rng.normal(size=(80, 3))
    value = vif_max(states, 3)
    assert math.isfinite(value)
    assert 1.0 <= value < 1e6


def test_simulation_is_reproducible_and_latches_nonpositive_diffusion():
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(6, 3))
    increments = rng.normal(size=(6, 5, 3))

    def drift(z):
        return -0.2 * z

    def diffusion(z):
        return np.full_like(z, 0.3)

    first = simulate_em_diag(drift, diffusion, x0, 0.05, increments, 8.0)
    second = simulate_em_diag(drift, diffusion, x0, 0.05, increments, 8.0)
    assert np.allclose(first["states"], second["states"], equal_nan=True)
    assert first["failure_rate"] == 0.0
    assert np.isfinite(first["states"]).all()

    def negative(z):
        out = np.full_like(z, 0.3)
        out[:, 0] = -0.1
        return out

    failed = simulate_em_diag(drift, negative, x0, 0.05, increments, 8.0)
    assert failed["failure_rate"] == 1.0
    assert set(failed["failure_class"]) == {"nonpositive_diffusion"}
    assert np.isnan(failed["states"][:, 1:]).all()


def test_decide_gates_and_indeterminate_vif():
    assert decide(CFG, _metrics()) == {key: "pass" for key in
                                       ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")}
    high = decide(CFG, _metrics(vif=40.0))
    assert high["H3"] == "indeterminate"
    assert high["H1"] == "pass"
    blown = decide(CFG, _metrics(fail_teacher=0.2, path_rmse=0.01))
    assert blown["H4"] == "indeterminate"
    floor = decide(CFG, _metrics(law_floor=0.4, law_symbolic=0.01))
    assert floor["H5"] == "indeterminate"
    missed = decide(CFG, _metrics(drift=[_row(0.4, 0.8), _row(), _row()]))
    assert missed["H1"] == "fail"
    assert nrmse(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
