"""Record a run for replay: the wire's node positions at every step, in the anatomy's frame.

``lumen view ANATOMY --run`` drives the wire to a target with :class:`FollowPolicy`, a simple
scripted operator, and saves what happened as a ``lumen-replay/0`` JSON file. Any other
policy (an RL agent, say) can be recorded the same way with :func:`record`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPLAY_VERSION = "lumen-replay/0"
ENV_DEFAULTS = {"max_insertion": 2.0}


class FollowPolicy:
    """Push; when the tip leaves the route or stops gaining, pull back until it is on the route
    again, rotate the shaped tip about a third of a turn, and push again. Crude, but it is how
    a person probes a branch point, and it gives replays that show real navigation."""

    def __init__(self, stall_steps: int = 10, max_back: int = 15, turn_steps: int = 2):
        self.stall_steps, self.max_back, self.turn_steps = stall_steps, max_back, turn_steps
        self.reset()

    def reset(self):
        self.phase, self.count, self.best, self.since = "push", 0, np.inf, 0

    def __call__(self, info: dict) -> np.ndarray:
        if info:
            self._update(info)
        return {"push": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0]),
                "turn": np.array([0.0, 1.0])}[self.phase].astype(np.float32)

    def _update(self, info):
        if self.phase == "push":
            d = float(info.get("dist", np.inf))
            if d < self.best - 0.3:
                self.best, self.since = d, 0
            else:
                self.since += 1
            if info.get("off_route") or self.since >= self.stall_steps:
                self.phase, self.count = "back", 0
        elif self.phase == "back":
            self.count += 1
            if (not info.get("off_route") and self.count >= 3) or self.count >= self.max_back:
                self.phase, self.count = "turn", 0
        else:
            self.count += 1
            if self.count >= self.turn_steps:
                self.phase, self.since, self.best = "push", 0, np.inf


def record(env, policy=None, steps: int = 400, *, anatomy: str = "", target: str = "") -> dict:
    """Run ``env`` (a TreeNavEnv) under ``policy(info) -> action`` and return a replay dict."""
    policy = policy or FollowPolicy()
    if hasattr(policy, "reset"):
        policy.reset()
    env.reset()
    frames = [_frame(env, 0, np.zeros(2), {})]
    info: dict = {}
    outcome = {"reached": False, "steps": 0}
    for k in range(1, steps + 1):
        action = np.asarray(policy(info), np.float32)
        _, _, term, trunc, info = env.step(action)
        frames.append(_frame(env, k, action, info))
        outcome = {"reached": bool(info.get("success")), "steps": k,
                   "left_mm": round(float(info.get("dist", 0.0)), 2)}
        if term or trunc:
            break
    pens = [f["overlap_mm"] for f in frames]
    outcome["max_overlap_mm"] = round(max(pens), 3)
    return {"version": REPLAY_VERSION, "anatomy": anatomy, "target": target,
            "target_node": getattr(env, "target_node", None),
            "dt_s": 5e-3 * getattr(env, "substeps", 4),
            "device_radius_mm": float(getattr(env, "device_radius_mm", 0.2)),
            "route": np.round(np.asarray(env.route_frame.points, float), 2).tolist(),
            "frames": frames, "outcome": outcome}


def _frame(env, k: int, action, info: dict) -> dict:
    pos = np.asarray(env.sim.body_positions(), float)
    return {"step": k, "nodes": np.round(pos, 2).tolist(),
            "action": [round(float(a), 3) for a in np.asarray(action).ravel()[:2]],
            "left_mm": round(float(info.get("dist", env.L)), 2),
            "overlap_mm": round(float(info.get("max_pen", 0.0)), 3),
            "off_route": bool(info.get("off_route", False))}


def save(replay: dict, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(replay, separators=(",", ":")))
    return p


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("version") != REPLAY_VERSION:
        raise ValueError(f"{path} is not a Lumen replay ({REPLAY_VERSION})")
    for f in data.get("frames", []):
        nodes = np.asarray(f.get("nodes"), float)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
            raise ValueError(f"{path}: frame {f.get('step')} has no usable wire positions")
    return data
