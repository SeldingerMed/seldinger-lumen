"""Navigation env: drive the device tip to a target arc-length in a procedural tube.

Action  (1-D): axial push in [-1, 1], scaled to a force (translate/insert -- the
                proximal-end actuation of doc §1.2).
Obs     (5-D): [tip_s/L, tip_r/R, sin(theta), cos(theta), (target - tip_s)/L].
Reward       : progress toward the target, minus a wall-contact penalty and a
                small time penalty.

Built on the rigid-tube fast tier for throughput; pass a wall/flow/occlusion to
the underlying solver to make it harder. Procedural anatomy only -- no patient
data -- so it is fully shippable in the open repo.
"""

from __future__ import annotations

import numpy as np
import torch

from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod, RodParams
from lumen.physics.solver import SimConfig, Solver

try:                                # optional: expose gym spaces if available
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:                   # pragma: no cover - gym is optional
    _HAS_GYM = False


class NavEnv:
    def __init__(self, asset=None, target_frac=0.7, max_insertion=2.0,
                 substeps=8, max_steps=40, success_tol=2.0, seed=0,
                 dtype=torch.float64):
        from lumen.assets import procedural
        self.asset = asset or procedural.straight_tube(length=80.0, radius=2.0)
        pts, lumen = self.asset.edge_arrays(self.asset.edges[0])
        self.geom = ContactGeometry(pts, lumen, dtype=dtype)
        self.L = float(self.geom.cum_s[-1])
        self.R = float(self.geom.R_grid.mean())
        self.target_s = target_frac * self.L
        self.max_insertion = max_insertion
        self.substeps, self.max_steps = substeps, max_steps
        self.success_tol = success_tol
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.cp = ContactParams(kappa=1.5e3, d_hat=0.25, mu=0.2)
        if _HAS_GYM:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)
        self.reset()

    def _make_rod(self):
        n, sp = 12, 2.0
        p0 = self.geom.P[0].cpu().numpy()
        t0 = self.geom.T[0].cpu().numpy()
        x0 = p0[None, :] + np.arange(n)[:, None] * sp * t0[None, :]
        # seat the rod slightly off-axis so contact is exercised
        x0 = x0 + 0.3 * np.array([1.0, 0.0, 0.0])
        return Rod(torch.tensor(x0, dtype=self.dtype).unsqueeze(0),
                   RodParams(k_stretch=3e2, k_bend=3.0, damping=2e2))

    def _tip(self):
        proj = self.geom.project(self.rod.x)
        return (float(proj["s"][0, -1]), float(proj["r"][0, -1]),
                float(proj["theta"][0, -1]), float(proj["r"].max()))

    def _obs(self):
        s, r, th, _ = self._tip()
        return np.array([s / self.L, r / self.R, np.sin(th), np.cos(th),
                         (self.target_s - s) / self.L], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.rod = self._make_rod()
        self.steps = 0
        self._prev_dist = abs(self._tip()[0] - self.target_s)
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        delta = a * self.max_insertion          # mm to feed the device this step
        with torch.no_grad():
            # kinematic insertion: feed the whole device forward along its axis,
            # then relax contact/elastics so it conforms to the lumen
            x = self.rod.x
            tang = torch.zeros_like(x)
            tang[:, :-1] = x[:, 1:] - x[:, :-1]
            tang[:, -1] = x[:, -1] - x[:, -2]
            tang = tang / torch.linalg.norm(tang, dim=-1, keepdim=True).clamp_min(1e-12)
            self.rod.x = x + delta * tang
            cfg = SimConfig(dt=8e-3, steps=self.substeps, anchor_base=False)
            self.rod = Solver(self.geom, contact=self.cp, cfg=cfg).rollout(self.rod)
        self.steps += 1
        s, r, th, rmax = self._tip()
        dist = abs(s - self.target_s)
        contact_pen = max(0.0, rmax - (self.R - self.cp.d_hat))
        reward = (self._prev_dist - dist) - 0.5 * contact_pen - 0.01
        self._prev_dist = dist
        terminated = dist < self.success_tol
        truncated = self.steps >= self.max_steps
        if terminated:
            reward += 10.0
        info = {"tip_s": s, "dist": dist, "max_r": rmax, "success": terminated}
        return self._obs(), float(reward), terminated, truncated, info
