"""L0d.1c — navigation on a vascular TREE: drive the tip down a chosen branch.

Extends the single-tube NavEnv to a graph: a target is a node in a specific branch,
and progress is arc-length along the ROUTE (the root→target edge path). The route is
flattened to one polyline so progress/θ come from a plain CenterlineFrame, while the
contact radius is measured against the true nearest edge (junction-aware) via the
VascularTree.

Honest scope: this makes the tree-navigation TASK expressible and the obs/reward
pipeline correct. It does NOT claim a small policy solves autonomous branch SELECTION
(steering into the right vessel at a junction is hard — same feasibility caveat as the
image-nav work); seed/steer to validate, leave learned selection to a harder task.

Requires `newton` + `warp` (it runs the Layer-0 tree sim).
"""

from __future__ import annotations

import numpy as np

from lumen.core.frame import CenterlineFrame
from lumen.core.tree import VascularTree
from lumen.hardware import detect_device


def _route_polyline(tree: VascularTree, route: list[int], start_node: str):
    """Concatenate the route edges into one oriented polyline (each edge flipped as
    needed for continuity, shared junction points de-duplicated)."""
    pts_all, node = [], start_node
    for k, ei in enumerate(route):
        e = tree.edges[ei]
        p = np.asarray(e.frame.points, float)
        if e.node_a == node:
            nxt = e.node_b
        else:                                   # traversed b->a: flip so it runs start->end
            p, nxt = p[::-1], e.node_a
        pts_all.append(p if k == 0 else p[1:])  # drop the shared junction point
        node = nxt
    return np.concatenate(pts_all, axis=0)


class TreeNavEnv:
    def __init__(self, asset, target_node=None, target_frac=1.0, max_insertion=2.0,
                 substeps=4, max_steps=60, success_tol=2.5, blend_len=4.0, device=None):
        self.asset = asset
        self.tree = VascularTree(asset, blend_len=blend_len)
        self.start_node = asset.device_spawn.node_id
        # default target = a leaf (degree-1 node) that isn't the start
        if target_node is None:
            leaves = [n.id for n in asset.nodes
                      if not self.tree.is_junction(n.id) and n.id != self.start_node]
            target_node = leaves[-1]
        self.target_node = target_node
        self.route = self.tree.route(target_node, self.start_node)
        route_pts = _route_polyline(self.tree, self.route, self.start_node)
        self.route_frame = CenterlineFrame(route_pts)
        self.L = float(self.route_frame.length)
        self.R = float(np.asarray(self.tree.edges[self.route[0]].lf.R).mean())
        self.target_s = float(np.clip(target_frac, 0.0, 1.0)) * self.L
        self.max_insertion, self.substeps, self.max_steps = max_insertion, substeps, max_steps
        self.success_tol = success_tol
        self.device = device or detect_device()
        self.reset()

    def _device_points(self, n=10, sp=2.0):
        f = self.route_frame
        p0, t0, m1 = f.points[0], f.tangents[0], f.m1[0]
        return (p0 + 0.5 * self.R * m1)[None, :] + np.arange(n)[:, None] * sp * t0[None, :]

    def _tip(self):
        pos = self.sim.body_positions()[-1]
        rs = self.route_frame.project(pos)               # progress + θ along the route
        r = self.tree.project(pos).r                     # radius vs the true nearest edge
        return rs.s, r, rs.theta

    def _obs(self):
        s, r, th = self._tip()
        return np.array([s / self.L, r / self.R, np.sin(th), np.cos(th),
                         (self.target_s - s) / self.L], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        from lumen.newton.sim import NewtonGuidewireSim
        trunk_pts = np.asarray(self.tree.edges[self.route[0]].frame.points)
        if getattr(self, "sim", None) is None:
            self.sim = NewtonGuidewireSim(trunk_pts, self.R, self._device_points(),
                                          radius=0.2, kappa=3e3, d_hat=0.3,
                                          vbd_iterations=8, device=self.device, tree=self.tree)
        else:
            self.sim.reset()
        self.steps = 0
        self._prev = abs(self._tip()[0] - self.target_s)
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.sim.step(dt=5e-3 * self.substeps, substeps=self.substeps,
                      insertion=a * self.max_insertion)
        self.steps += 1
        s, r, _ = self._tip()
        obs = self._obs()
        if not (np.isfinite(obs).all() and np.isfinite([s, r]).all()):
            return np.zeros(5, np.float32), -100.0, True, False, {
                "route_s": 0.0, "dist": 1e6, "max_r": 0.0, "success": False, "diverged": True}
        dist = abs(s - self.target_s)
        contact_pen = max(0.0, float(self.sim.node_radii().max()) - self.R)
        reward = (self._prev - dist) - 0.5 * contact_pen - 0.01
        self._prev = dist
        terminated = bool(dist < self.success_tol)
        if terminated:
            reward += 10.0
        info = {"route_s": s, "dist": dist, "max_r": float(self.sim.node_radii().max()),
                "success": terminated, "edge": self.tree.project(self.sim.body_positions()[-1]).edge_id}
        return obs, float(reward), terminated, self.steps >= self.max_steps, info
