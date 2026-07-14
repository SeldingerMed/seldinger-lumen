"""Navigation env over the Newton Layer-0 solver (doc M5: Gym integration).

Drives the guidewire tip to a target arc-length in a procedural tube, backed by
``lumen.newton.NewtonGuidewireSim`` (Newton VBD on CPU or CUDA — picked by
``lumen.hardware.detect_device``). Follows the Gym/Gymnasium reset/step
convention without hard-depending on gymnasium.

Action  (2-D): insertion + twist in [-1, 1]; legacy scalar actions mean zero twist.
Obs     (5-D): [tip_s/L, tip_r/R, sin(theta), cos(theta), (target - tip_s)/L].
Reward       : progress toward the target − wall-contact penalty − time penalty.

Requires `newton` + `warp`.
"""

from __future__ import annotations

import numpy as np

from lumen.hardware import detect_device

try:
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:                       # pragma: no cover - gym optional
    _HAS_GYM = False


class NavEnv:
    def __init__(self, asset=None, target_frac=0.7, max_insertion=2.0,
                 max_twist=1.0, substeps=4, max_steps=40, success_tol=2.5,
                 safety_max_pen=0.3, terminate_on_unsafe: bool = False, device=None):
        from lumen.assets import procedural
        from lumen.core.frame import CenterlineFrame
        self.asset = asset or procedural.straight_tube(length=80.0, radius=2.0)
        pts, lumen = self.asset.edge_arrays(self.asset.edges[0])
        self.vessel = np.asarray(pts)
        self.lumen = lumen                       # true R(s,θ) — passed to contact (not averaged)
        self.R = float(np.asarray(lumen.R).mean())   # representative R, for obs normalisation only
        self.frame = CenterlineFrame(self.vessel)
        self.L = float(self.frame.length)
        self.target_frac = target_frac
        self.target_s = target_frac * self.L
        self.rng = np.random.default_rng()
        self.max_insertion, self.max_twist = max_insertion, max_twist
        self.substeps, self.max_steps = substeps, max_steps
        self.success_tol = success_tol
        self.safety_max_pen = float(safety_max_pen)
        self.terminate_on_unsafe = bool(terminate_on_unsafe)
        self.device = device or detect_device()
        if _HAS_GYM:
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)
        self.reset()

    def _device_points(self, n=10, sp=2.0):
        # short guidewire seated just inside the wall at the vessel entrance
        p0, t0 = self.frame.points[0], self.frame.tangents[0]
        m1 = self.frame.m1[0]
        return (p0 + 0.5 * self.R * m1)[None, :] + np.arange(n)[:, None] * sp * t0[None, :]

    def _tip(self):
        pos = self.sim.body_positions()[-1]
        pr = self.frame.project(pos)
        return pr.s, pr.r, pr.theta, float(self.sim.node_radii().max())

    def _contact_features(self):
        if not (hasattr(self.sim, "body_positions") and hasattr(self, "frame") and hasattr(self, "lumen")):
            _, _, _, rmax = self._tip()
            return float(rmax), max(0.0, float(rmax) - self.R)
        projs = [self.frame.project(p) for p in self.sim.body_positions()]
        max_r = max((float(pr.r) for pr in projs), default=0.0)
        max_pen = max(
            (max(0.0, float(pr.r) - float(self.lumen.eval(pr.s, pr.theta))) for pr in projs),
            default=0.0,
        )
        return max_r, max_pen

    def _tip_roll(self):
        x, y, z, w = self.sim.body_quaternions()[-1]
        return float(2.0 * np.arctan2(z, w))

    def _parse_action(self, action):
        act = np.asarray(action, dtype=float).reshape(-1)
        if len(act) < 1:
            raise ValueError("action must contain at least an insertion command")
        insertion = float(np.clip(act[0], -1.0, 1.0))
        twist = float(np.clip(act[1] if len(act) > 1 else 0.0, -1.0, 1.0))
        return insertion, twist

    def _obs(self):
        s, r, th, _ = self._tip()
        return np.array([s / self.L, r / self.R, np.sin(th), np.cos(th),
                         (self.target_s - s) / self.L], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        from lumen.newton.sim import NewtonGuidewireSim
        # #13 — honour the seed: reproducible-but-varied episodes via a jittered
        # target (cheap — no model rebuild, unlike varying the start geometry).
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.target_s = float(np.clip(self.target_frac + self.rng.uniform(-0.1, 0.1),
                                      0.25, 0.9)) * self.L
        if getattr(self, "sim", None) is None:
            self.sim = NewtonGuidewireSim(self.vessel, self.R, self._device_points(),
                                          radius=0.2, kappa=3e3, d_hat=0.3,
                                          lumen_field=self.lumen,
                                          vbd_iterations=8, device=self.device)
        else:
            self.sim.reset()                 # cheap state restore, no rebuild
        self.steps = 0
        self._prev_dist = abs(self._tip()[0] - self.target_s)
        return self._obs(), {}

    def step(self, action):
        a, twist = self._parse_action(action)
        self.sim.step(dt=5e-3 * self.substeps, substeps=self.substeps,
                      insertion=a * self.max_insertion, twist=twist * self.max_twist)
        self.steps += 1
        s, r, th, rmax = self._tip()
        obs = self._obs()
        # #14 — NaN guard: a diverged sim must not emit NaN obs/reward (invalid JSON,
        # broken comparisons). End the episode with a finite penalty instead.
        if not (np.isfinite(obs).all() and np.isfinite([s, rmax]).all()):
            zeros = np.zeros(5, dtype=np.float32)
            return zeros, -100.0, True, False, {
                "tip_s": 0.0, "dist": 1e6, "max_r": 0.0,        # L1: finite (JSON-safe)
                "max_pen": 0.0, "success": False, "diverged": True}
        dist = abs(s - self.target_s)
        max_r, max_pen = self._contact_features()
        contact_pen = max_pen
        reward = (self._prev_dist - dist) - 0.5 * contact_pen - 0.01
        self._prev_dist = dist
        unsafe = bool(max_pen > getattr(self, "safety_max_pen", 0.3))
        success = bool(dist <= self.success_tol)
        safe_success = bool(success and not unsafe)
        terminated = bool(success or (unsafe and getattr(self, "terminate_on_unsafe", False)))
        truncated = self.steps >= self.max_steps
        if success:
            reward += 10.0
        if unsafe:
            reward -= 10.0
        info = {"tip_s": s, "dist": dist, "max_r": max_r, "max_pen": max_pen,
                "success": success, "safe_success": safe_success, "unsafe": unsafe,
                "twist": twist, "tip_roll": self._tip_roll()}
        return obs, float(reward), terminated, truncated, info
