"""Clinically meaningful episode metrics."""

import pytest

from lumen.data import Episode, EpisodeMeta, Outcome, Step
from lumen.data.capture import EpisodeRecorder
from lumen.data.metrics import compute_clinical_metrics


def _episode():
    return Episode(
        meta=EpisodeMeta(
            device={"guidewire": {"radius": 0.2}, "catheter": {"radius": 0.65}},
            labels={"target_branch": "m2_left"},
            notes={
                "target_s": 100.0,
                "success_tol": 2.5,
                "perforation_force_threshold": 9.0,
                "perforation_penetration_threshold": 0.5,
                "fragmentation_damage_threshold": 0.8,
                "flow_restoration_threshold": 0.7,
                "support_gap_threshold": 4.0,
            },
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={
                     "tip_s": 70.0,
                     "edge": "trunk",
                     "wall_force_max": 3.0,
                     "max_penetration": 0.0,
                     "flow_baseline_Q": 4.0,
                     "flow_downstream_Q": 0.8,
                     "clot_damage_max": 0.1,
                     "clot_occlusion_max": 1.2,
                     "retrieval_status": "none",
                     "catheter_tip_s": 68.0,
                 }),
            Step(t=0.1, action={"insertion": -1.0, "aspiration": 0.6},
                 kinematics={
                     "tip_s": 99.0,
                     "edge": "m2_left",
                     "wall_force_max": 12.0,
                     "max_penetration": 0.6,
                     "flow_baseline_Q": 4.0,
                     "flow_downstream_Q": 3.2,
                     "clot_damage_max": 0.9,
                     "clot_occlusion_max": 0.5,
                     "retrieval_status": "fragment",
                     "catheter_tip_s": 96.0,
                 }),
        ],
        outcome=Outcome(success=False, final_dist=99.0, steps=2, retrieval="slip",
                        label="thrombectomy"),
    )


def test_compute_clinical_metrics_names_each_required_endpoint():
    m = compute_clinical_metrics(_episode())

    assert m["tip_target"]["success"] is True
    assert m["tip_target"]["final_dist"] == pytest.approx(1.0)
    assert m["branch_choice"] == {"target": "m2_left", "final": "m2_left", "correct": True}
    assert m["wall_safety"]["perforation_risk"] is True
    assert m["wall_safety"]["max_wall_force"] == pytest.approx(12.0)
    assert m["clot"]["retrieval"] == "fragment"
    assert m["clot"]["fragmentation"] is True
    assert m["clot"]["distal_emboli_proxy"] > 0.0
    assert m["flow"]["restoration"] == pytest.approx(0.8)
    assert m["flow"]["restored"] is True
    assert m["catheter_support"]["final_gap"] == pytest.approx(3.0)
    assert m["catheter_support"]["supported"] is True


def test_compute_clinical_metrics_degrades_when_a_signal_is_absent():
    ep = Episode(meta=EpisodeMeta(notes={"target_s": 10.0}),
                 steps=[Step(t=0.0, action={}, kinematics={"tip_s": 1.0})],
                 outcome=Outcome(success=False, final_dist=9.0, steps=1))

    m = compute_clinical_metrics(ep)

    assert m["tip_target"]["success"] is False
    assert m["branch_choice"]["correct"] is None
    assert m["wall_safety"]["perforation_risk"] is False
    assert m["clot"]["retrieval"] == "none"
    assert m["flow"]["restoration"] is None
    assert m["catheter_support"]["supported"] is None


def test_recorder_samples_flow_clot_retrieval_and_catheter_support_signals():
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    import numpy as np

    from lumen.newton.devices import Stentriever
    from lumen.newton.flow import FlowField, FlowFieldParams
    from lumen.newton.sim import NewtonGuidewireSim

    vessel = np.stack([np.zeros(60), np.zeros(60), np.linspace(0, 120.0, 60)], axis=1)
    wire = np.stack([np.full(11, 0.3), np.zeros(11), 40.0 + np.arange(11) * 2.0], axis=1)
    cath = np.stack([np.full(13, 0.3), np.zeros(13), 34.0 + np.arange(13) * 2.0], axis=1)
    sim = NewtonGuidewireSim(vessel, 2.0, wire, radius=0.2,
                             catheter_points=cath, catheter_radius=0.65,
                             catheter_inner_radius=0.5,
                             flow=FlowField(FlowFieldParams(P_pulse=0.0)),
                             clot_segment=(55.0, 70.0), clot_height=1.2,
                             stentriever=Stentriever(deployed_center=62.0),
                             device="cpu")
    meta = EpisodeMeta(
        notes={"target_s": 60.0, "support_gap_threshold": 4.0,
               "flow_restoration_threshold": 0.5},
        labels={"target_branch": "e0"},
        device={"guidewire": {"radius": 0.2}, "catheter": {"radius": 0.65}},
    )
    rec = EpisodeRecorder(sim, modality="none", meta=meta, substeps=2)

    rec.record_step({"insertion": -1.0, "aspiration": 0.6})
    ep = rec.episode(Outcome(success=False, final_dist=0.0, steps=1))
    ep.outcome.metrics = compute_clinical_metrics(ep)

    kin = ep.steps[0].kinematics
    assert {"flow_downstream_Q", "flow_baseline_Q", "clot_damage_max",
            "clot_occlusion_max", "support_gap"}.issubset(kin)
    assert ep.outcome.metrics["flow"]["final_Q"] is not None
    assert ep.outcome.metrics["catheter_support"]["final_gap"] is not None
    assert ep.outcome.metrics["clot"]["retrieval"] in {"none", "retrieve", "slip", "fragment"}
