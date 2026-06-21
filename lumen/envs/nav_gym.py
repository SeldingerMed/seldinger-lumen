"""Navigation env over the Newton Layer-0 solver (doc M5: Gym integration).

Drives the guidewire tip to a target arc-length in a procedural tube, backed by
``lumen.newton.NewtonGuidewireSim`` (Newton VBD on CPU or CUDA — picked by
``lumen.hardware.detect_device``). Follows the Gym/Gymnasium reset/step
convention without hard-depending on gymnasium.

Action  (1-D): insertion rate in [-1, 1] (advance/retract the proximal end).
Obs     (5-D): [tip_s/L, tip_r/R, sin(theta), cos(theta), (target - tip_s)/L].
Reward       : progress toward the target − wall-contact penalty − time penalty.

Requires `newton` + `warp`.
"""

from __future__ import annotations

import numpy as np

from lumen.hardware import detect_device

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:                       # pragma: no cover - gym optional
    _HAS_GYM = False


class NavEnv:
    def __init__(self, asset=None, target_frac=0.7, max_insertion=2.0,
                 substeps=4, max_steps=40, success_tol=2.5, device=None):
        from lumen.assets import procedural
        from lumen.core.frame import CenterlineFrame
        self.asset = asset or procedural.straight_tube(length=80.0, radius=2.0)
        pts, lumen = self.asset.edge_arrays(self.asset.edges[0])
        self.vessel = np.asarray(pts)
        self.R = float(np.asarray(lumen.R).mean())
        self.frame = CenterlineFrame(self.vessel)
        self.L = float(self.frame.length)
        self.target_s = target_frac * self.L
        self.max_insertion, self.substeps, self.max_steps = max_insertion, substeps, max_steps
        self.success_tol = success_tol
        self.device = device or detect_device()
        if _HAS_GYM:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
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

    def _obs(self):
        s, r, th, _ = self._tip()
        return np.array([s / self.L, r / self.R, np.sin(th), np.cos(th),
                         (self.target_s - s) / self.L], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        from lumen.newton.sim import NewtonGuidewireSim
        self.sim = NewtonGuidewireSim(self.vessel, self.R, self._device_points(),
                                      radius=0.2, kappa=3e3, d_hat=0.3,
                                      vbd_iterations=8, device=self.device)
        self.steps = 0
        self._prev_dist = abs(self._tip()[0] - self.target_s)
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.sim.step(dt=5e-3, substeps=self.substeps, insertion=a * self.max_insertion)
        self.steps += 1
        s, r, th, rmax = self._tip()
        dist = abs(s - self.target_s)
        contact_pen = max(0.0, rmax - self.R)
        reward = (self._prev_dist - dist) - 0.5 * contact_pen - 0.01
        self._prev_dist = dist
        terminated = dist < self.success_tol
        truncated = self.steps >= self.max_steps
        if terminated:
            reward += 10.0
        info = {"tip_s": s, "dist": dist, "max_r": rmax, "success": terminated}
        return self._obs(), float(reward), terminated, truncated, info
