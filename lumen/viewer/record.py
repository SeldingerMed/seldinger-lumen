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
    """How a person probes a branch point: push; when the tip leaves the route or stops gaining,
    pull back until it is on the route a few millimetres short of the branch point it stuck at
    (or of where it stuck, away from a branch), turn the shaped tip a little (about 30
    degrees), and push again. Successive tries sweep the tip round the full circle from the
    same approach, so a branch that opens only at one roll is found.

    It reads the env's ``tip_mm`` (the tip's place along the route, tracked inside the vessels)
    and ``branch_mm`` (the roadmap's branch points along the route) when present; otherwise it
    falls back to the distance to the target."""

    def __init__(self, stall_steps: int = 6, max_back: int = 15, turn_steps: int = 2,
                 back_mm: float = 3.0, approach_mm: float = 5.0):
        self.stall_steps, self.max_back, self.turn_steps = stall_steps, max_back, turn_steps
        self.back_mm, self.approach_mm = float(back_mm), float(approach_mm)
        self.reset()

    def reset(self):
        self.phase, self.count, self.best, self.since = "push", 0, np.inf, 0
        self.back_to = None

    def _pullback_target(self, info) -> float:
        """Where to pull back to: ``approach_mm`` short of the branch point the tip is stuck at
        (the nearest on the roadmap, ``branch_mm``), and at least ``approach_mm`` behind the
        tip; else ``back_mm`` short of where it stuck."""
        tip = float(info["tip_mm"])
        near = [b for b in info.get("branch_mm", ())
                if b - 2.0 * self.approach_mm <= tip <= b + self.approach_mm]
        if near:
            b = min(near, key=lambda b: abs(b - tip))
            return min(b - self.approach_mm, tip - self.approach_mm)
        return tip - self.back_mm

    def __call__(self, info: dict) -> np.ndarray:
        if info:
            self._update(info)
        return {"push": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0]),
                "turn": np.array([0.0, 1.0])}[self.phase].astype(np.float32)

    def _update(self, info):
        tracked = "tip_mm" in info
        if self.phase == "push":
            # progress of this push, from where it began (not the furthest point ever reached,
            # which a push after a pull-back must first re-cover)
            d = -float(info["tip_mm"]) if tracked else float(info.get("dist", np.inf))
            if d < self.best - 0.3:
                self.best, self.since = d, 0
            else:
                self.since += 1
            if info.get("off_route") or self.since >= self.stall_steps:
                self.phase, self.count = "back", 0
                self.back_to = self._pullback_target(info) if tracked else None
        elif self.phase == "back":
            self.count += 1
            if self.back_to is not None:
                short = float(info.get("tip_mm", self.back_to)) <= self.back_to
            else:
                short = self.count >= 3
            if (not info.get("off_route") and short) or self.count >= self.max_back:
                self.phase, self.count = "turn", 0
                self.best, self.since = np.inf, 0
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
    try:                                     # the env's own distance, as every step reports it
        start = {"dist": abs(float(env._features()["s"]) - float(env.target_s))}
    except AttributeError:
        start = {}                           # a stand-in env: _frame falls back to the route length
    first = _frame(env, 0, np.zeros(2), start)
    if first is None:
        raise RuntimeError("the wire's starting position is not finite; nothing to record")
    frames = [first]
    info: dict = {}
    outcome = {"reached": False, "steps": 0, "left_mm": frames[0]["left_mm"]}
    for k in range(1, steps + 1):
        action = np.asarray(policy(info), np.float32)
        _, _, term, trunc, info = env.step(action)
        frame = _frame(env, k, action, info)
        if info.get("diverged") or frame is None:
            # The solver blew up: its positions may be NaN. End the replay on the last good frame.
            outcome = {**outcome, "diverged": True}
            break
        frames.append(frame)
        outcome = {"reached": bool(info.get("success")), "steps": k,
                   "left_mm": round(float(info.get("dist", 0.0)), 2)}
        if term or trunc:
            break
    pens = [f["overlap_mm"] for f in frames]
    outcome["max_overlap_mm"] = round(max(pens), 3)
    from lumen.viewer.scene import fingerprint

    return {"version": REPLAY_VERSION, "anatomy": Path(anatomy).name if anatomy else "",
            "anatomy_id": fingerprint(env.asset), "target": target,
            "target_node": getattr(env, "target_node", None),
            "dt_s": 5e-3 * getattr(env, "substeps", 4),
            "device_radius_mm": float(getattr(env, "device_radius_mm", 0.2)),
            "route": np.round(np.asarray(env.route_frame.points, float), 2).tolist(),
            "frames": frames, "outcome": outcome}


def _frame(env, k: int, action, info: dict) -> dict | None:
    pos = np.asarray(env.sim.body_positions(), float)
    if not np.isfinite(pos).all():
        return None
    return {"step": k, "nodes": np.round(pos, 2).tolist(),
            "action": [round(float(a), 3) for a in np.asarray(action).ravel()[:2]],
            "left_mm": round(float(info.get("dist", env.L)), 2),
            "overlap_mm": round(float(info.get("max_pen", 0.0)), 3),
            "off_route": bool(info.get("off_route", False))}


def save(replay: dict, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(replay, separators=(",", ":"), allow_nan=False))
    return p


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("version") != REPLAY_VERSION:
        raise ValueError(f"{path} is not a Lumen replay ({REPLAY_VERSION})")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path}: the replay has no frames")
    for f in frames:
        if not isinstance(f, dict):
            raise ValueError(f"{path}: a frame is not an object")
        try:
            nodes = np.asarray(f.get("nodes"), float)
        except (TypeError, ValueError):
            nodes = np.zeros((0,))
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2 or not np.isfinite(nodes).all():
            raise ValueError(f"{path}: frame {f.get('step')} has no usable wire positions")
        for key in ("left_mm", "overlap_mm"):
            v = f.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValueError(f"{path}: frame {f.get('step')} has no {key}")
    dt = data.get("dt_s")
    if dt is not None and (isinstance(dt, bool) or not isinstance(dt, (int, float))
                           or not np.isfinite(dt) or dt <= 0):
        raise ValueError(f"{path}: dt_s must be a positive number of seconds")
    return data
