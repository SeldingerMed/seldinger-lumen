"""Clinical endpoint metrics for replayed/captured cases.

These are episode-level summaries, not rewards. They name the things a CV/endo
reviewer cares about: target reach, branch choice, safety, thrombectomy result,
flow, and coaxial support.
"""

from __future__ import annotations

import math

import numpy as np

from lumen.data.schema import Episode


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _step_values(ep: Episode, *keys):
    vals = []
    for step in ep.steps:
        kin = step.kinematics if isinstance(step.kinematics, dict) else {}
        for key in keys:
            if key in kin:
                v = _finite(kin[key])
                if v is not None:
                    vals.append(v)
                break
    return vals


def _last_value(ep: Episode, *keys):
    vals = _step_values(ep, *keys)
    return vals[-1] if vals else None


def _max_value(ep: Episode, *keys):
    vals = _step_values(ep, *keys)
    return max(vals) if vals else None


def _last_text(ep: Episode, *keys):
    for step in reversed(ep.steps):
        kin = step.kinematics if isinstance(step.kinematics, dict) else {}
        for key in keys:
            if kin.get(key) is not None:
                return str(kin[key])
    return None


def _threshold(notes, key, default):
    v = _finite(notes.get(key)) if isinstance(notes, dict) else None
    return default if v is None else v


def _tip_target(ep: Episode, notes: dict) -> dict:
    target_s = _finite(notes.get("target_s")) if isinstance(notes, dict) else None
    tol = _threshold(notes, "success_tol", 2.5)
    final_tip_s = _last_value(ep, "tip_s", "route_s")
    if target_s is not None and final_tip_s is not None:
        final_dist = abs(final_tip_s - target_s)
        success = bool(final_dist <= tol)
    else:
        final_dist = _finite(ep.outcome.final_dist)
        success = bool(ep.outcome.success)
    return {"success": success, "final_dist": final_dist, "target_s": target_s,
            "success_tol": tol}


def _branch_choice(ep: Episode, notes: dict) -> dict:
    labels = ep.meta.labels if isinstance(ep.meta.labels, dict) else {}
    target = (labels.get("target_branch") or labels.get("target_edge")
              or notes.get("target_branch") or notes.get("target_edge")
              or notes.get("target_node"))
    final = _last_text(ep, "branch", "edge", "tip_edge")
    correct = None if not target or not final else final == str(target)
    return {"target": None if target is None else str(target), "final": final, "correct": correct}


def _wall_safety(ep: Episode, notes: dict) -> dict:
    max_force = _max_value(ep, "wall_force_max", "wall_force", "wall_load_max")
    max_pen = _max_value(ep, "max_penetration", "max_pen")
    force_thr = _threshold(notes, "perforation_force_threshold", math.inf)
    pen_thr = _threshold(notes, "perforation_penetration_threshold", math.inf)
    force_score = 0.0 if max_force is None or not math.isfinite(force_thr) else max_force / force_thr
    pen_score = 0.0 if max_pen is None or not math.isfinite(pen_thr) else max_pen / pen_thr
    risk_score = max(force_score, pen_score)
    return {"max_wall_force": max_force, "max_penetration": max_pen,
            "force_threshold": None if not math.isfinite(force_thr) else force_thr,
            "penetration_threshold": None if not math.isfinite(pen_thr) else pen_thr,
            "risk_score": float(risk_score), "perforation_risk": bool(risk_score >= 1.0)}


def _clot(ep: Episode, notes: dict) -> dict:
    status = _last_text(ep, "retrieval_status") or ep.outcome.retrieval or "none"
    max_damage = _max_value(ep, "clot_damage_max", "clot_damage")
    residual = _last_value(ep, "clot_occlusion_max", "clot_occlusion")
    frag_thr = _threshold(notes, "fragmentation_damage_threshold", 0.8)
    explicit_emboli = _max_value(ep, "distal_emboli_proxy")
    fragmentation = status == "fragment" or (max_damage is not None and max_damage >= frag_thr)
    if explicit_emboli is not None:
        emboli = explicit_emboli
    elif fragmentation and max_damage is not None:
        emboli = max_damage * max(residual or 0.0, 0.0)
    else:
        emboli = 0.0
    return {"retrieval": status, "fragmentation": bool(fragmentation),
            "max_damage": max_damage, "residual_occlusion": residual,
            "distal_emboli_proxy": float(emboli)}


def _flow(ep: Episode, notes: dict) -> dict:
    baseline = _last_value(ep, "flow_baseline_Q", "flow_pre_Q")
    final = _last_value(ep, "flow_downstream_Q", "flow_final_Q", "downstream_Q")
    threshold = _threshold(notes, "flow_restoration_threshold", 0.7)
    restoration = None
    restored = None
    if baseline is not None and abs(baseline) > 1e-12 and final is not None:
        restoration = float(np.clip(final / baseline, 0.0, np.inf))
        restored = bool(restoration >= threshold)
    return {"baseline_Q": baseline, "final_Q": final, "restoration": restoration,
            "restoration_threshold": threshold, "restored": restored}


def _catheter_support(ep: Episode, notes: dict) -> dict:
    gaps = _step_values(ep, "support_gap", "catheter_gap")
    if not gaps:
        for step in ep.steps:
            kin = step.kinematics if isinstance(step.kinematics, dict) else {}
            tip_s = _finite(kin.get("tip_s"))
            cath_s = _finite(kin.get("catheter_tip_s"))
            if tip_s is not None and cath_s is not None:
                gaps.append(tip_s - cath_s)
    threshold = _threshold(notes, "support_gap_threshold", 4.0)
    if not gaps:
        return {"final_gap": None, "min_gap": None, "max_gap": None,
                "support_gap_threshold": threshold, "unsupported_lead": None,
                "supported": None}
    final_gap, min_gap, max_gap = gaps[-1], min(gaps), max(gaps)
    unsupported = max(0.0, final_gap - threshold)
    return {"final_gap": final_gap, "min_gap": min_gap, "max_gap": max_gap,
            "support_gap_threshold": threshold, "unsupported_lead": unsupported,
            "supported": bool(final_gap <= threshold)}


def compute_clinical_metrics(ep: Episode) -> dict:
    """Return clinically named endpoint metrics for an episode."""
    notes = ep.meta.notes if isinstance(ep.meta.notes, dict) else {}
    return {"tip_target": _tip_target(ep, notes),
            "branch_choice": _branch_choice(ep, notes),
            "wall_safety": _wall_safety(ep, notes),
            "clot": _clot(ep, notes),
            "flow": _flow(ep, notes),
            "catheter_support": _catheter_support(ep, notes)}
