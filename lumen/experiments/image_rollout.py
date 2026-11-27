"""One decision at a fork: select an insertion pulse from a selective DRR.

The same left-oriented rest shape and proximal rail are used for either goal.
The selected pulse changes the actual Newton dynamics, never the rail or tip.
This is a scripted, open-loop-after-selection experiment, not learned steering.
"""
from __future__ import annotations

import base64

import numpy as np

from lumen.assets.procedural import bifurcation
from lumen.envs.tree_nav import TreeNavEnv
from lumen.sensors import FluoroSensor
from lumen.sensors.carm import CArm

# Requested daughter, bifurcation angle, projection direction, insertion gain.
_REGISTRY = {"case-a": ("right", 25, (0, 1, 0), 2),
             "case-b": ("left", 35, (0, 1, 0), 12),
             "case-c": ("right", 25, (1, 0, 0), 2)}
CASES = tuple(_REGISTRY)
GOAL = "Enter the selectively contrasted daughter at the fork."
CONTACT_LIMIT_MM = 0.3


def choose(image):
    """Only input: displayed detector bytes. Look for a distal lateral blush."""
    h, w = image.shape
    distal = image[2*h//3:]
    left = float(distal[:, :w//2-3].sum())
    right = float(distal[:, w//2+3:].sum())
    if max(left, right) < 128 or min(left, right) > .25 * max(left, right):
        return "stop"
    return "left" if left > right else "right"


def run_case(case_id):
    target, angle, axis, gain = _REGISTRY[case_id]
    asset = bifurcation(trunk=35, branch=40, angle_deg=angle)
    # The insertions are open-loop and force-blind on purpose (case-b is the fast one), so the
    # feed limit that pauses a push against the wall is off, with the wall stiffness this
    # experiment was built on. A push that takes the wire over CONTACT_LIMIT_MM into the wall
    # ends the insertion.
    env = TreeNavEnv(asset, target_node="left_out", max_insertion=gain, device="cpu",
                     feed_stop_pen_mm=None, wall_stiffness=3.0e3)
    sensor = FluoroSensor(res=48, n_samples=96, nu=64, nv=64)
    camera = CArm.looking_at((0, 0, 36), axis=axis, width=168, height=168, nu=64, nv=64)
    branch = next(e for e in asset.edges if e.id == target)
    image, _ = sensor.render(env.sim.body_positions(), carm=camera,
                             contrast_nodes=branch.centerline_mm, contrast_radius=1.8)
    pixels = np.rint(255 * image / max(float(image.max()), 1e-9)).astype(np.uint8)
    # Copy across the observation boundary. No masks, camera, seed, or info dict.
    action = choose(pixels.copy())
    steps = []
    tips = [env.sim.body_positions()[-1].tolist()]
    outcome = "abstain" if action == "stop" else "timeout"
    reached = False
    if action != "stop":
        pulse = .5 if action == "left" else 1.
        for _ in range(45):
            _, _, terminated, truncated, info = env.step([pulse, 0.])
            if info.get("diverged"):
                raise RuntimeError("Newton diverged")
            tip = env.sim.body_positions()[-1]
            tips.append(tip.tolist())
            proj = env.tree.project(tip)
            reached = bool(proj.edge_id == target and proj.s >= 10.)
            steps.append({"edge": proj.edge_id, "edge_s_mm": float(proj.s),
                          "penetration_mm": float(info["max_pen"]),
                          "load_sim_units": float(info["wall_load_max"]),
                          "insertion_action": pulse, "twist_action": 0.})
            if info["max_pen"] > CONTACT_LIMIT_MM:
                outcome = "contact"
                break
            if reached:
                outcome = "reach"
                break
            if terminated or truncated:
                break
    pen = max((s["penetration_mm"] for s in steps), default=0.)
    load = max((s["load_sim_units"] for s in steps), default=0.)
    reasons = {"reach": f"The tip entered {target} and passed the 10 mm branch station.",
               "contact": f"The shaft or tip crossed the simulated wall by {pen:.3f} mm, above the {CONTACT_LIMIT_MM} mm experiment stop. The insertions were open-loop after the image decision.",
               "abstain": "The projection does not separate a contrasted daughter from the central column. No insertion was issued.",
               "timeout": "The pulse ended without reaching the target or triggering the wall stop."}
    return {"case_id": case_id, "goal": GOAL, "action": action,
            "frames": [{"shape": list(pixels.shape), "data": base64.b64encode(pixels.tobytes()).decode()}],
            "hidden": {"tip_xyz_mm": tips, "step_evidence": steps, "outcome": outcome,
                       "target": target, "angle_deg": angle, "insertion_gain": gain,
                       "reason": reasons[outcome]},
            "scores": {"target_reached": reached, "max_penetration_mm": pen,
                       "peak_wall_load_sim_units": load, "abstained": action == "stop"}}
