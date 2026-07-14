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

try:
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:                       # pragma: no cover - gym optional
    _HAS_GYM = False


def _route_polyline(tree: VascularTree, route: list[int], start_node: str):
    """Concatenate the route edges into one oriented polyline (each edge flipped as
    needed for continuity, shared junction points de-duplicated)."""
    if not route:                               # start == target: a single-point "route"
        return tree._node_pos[start_node][None, :].copy()
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
                 max_twist=1.0, substeps=4, max_steps=60, success_tol=2.5,
                 safety_max_pen=0.3, blend_len=4.0, terminate_on_unsafe: bool = False,
                 device=None):
        self.asset = asset
        self.tree = VascularTree(asset, blend_len=blend_len)
        self.start_node = asset.device_spawn.node_id
        # default target = a leaf (degree-1 node) that isn't the start
        if target_node is None:
            leaves = [n.id for n in asset.nodes
                      if not self.tree.is_junction(n.id) and n.id != self.start_node]
            target_node = leaves[-1]
        if target_node == self.start_node:
            raise ValueError("target_node must differ from start_node (empty route)")
        self.target_node = target_node
        self.route = self.tree.route(target_node, self.start_node)
        route_pts = _route_polyline(self.tree, self.route, self.start_node)
        self.route_frame = CenterlineFrame(route_pts)
        self.L = float(self.route_frame.length)
        self.R = float(np.asarray(self.tree.edges[self.route[0]].lf.R).mean())  # entry-edge ref scale
        self.target_s = float(np.clip(target_frac, 0.0, 1.0)) * self.L
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

    def _route_seed(self, s_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate route centerline points and frame normals at seed arc-lengths."""
        f = self.route_frame
        pts = np.stack([np.interp(s_nodes, f.cum_s, f.points[:, ax]) for ax in range(3)],
                       axis=1)
        # Nearest frame sample is enough for an initial rest shape; contact immediately
        # resolves small discontinuities, and this avoids over-smoothing normals through
        # a branch kink.
        idx = np.searchsorted(f.cum_s, s_nodes, side="right") - 1
        idx = np.clip(idx, 0, len(f.points) - 1)
        return pts, f.m1[idx]

    def _clamp_seed_inside_lumen(self, pts: np.ndarray, margin_mm: float = 0.25) -> np.ndarray:
        """Keep the initial seed inside the local tree lumen.

        Curved/tree routes should not start with a node already beyond the wall; that
        reports as a physics failure before navigation begins. Clamping only adjusts the
        initial rest pose into the visible lumen, leaving subsequent contact dynamics to
        the Newton solver. ``tree.project`` returns the local lumen radius (``R``) and
        radial unit vector (``e_r``), used here to move only the excess radial distance.
        """
        out = np.asarray(pts, float).copy()
        for i, p in enumerate(out):
            pr = self.tree.project(p)
            limit = max(0.0, float(pr.R) - float(margin_mm))
            if pr.r > limit:
                out[i] = p - pr.e_r * (pr.r - limit)
        return out

    def _device_points(self, n=10, sp=2.0, tip_bend_nodes=3, tip_bend_mm=0.7):
        """Shaped-tip guidewire seed that navigates the junction into the target branch.

        A straight, centred wire jams in the crotch of the fork and is then dragged across
        the septum (unphysical wall penetration). Two things fix it, and both matter:

        1. **Symmetric seed** — offset the shaft *out of the bifurcation plane* (not toward
           a branch), so neither branch is favoured by the initial pose.
        2. **Pre-shaped tip** — a distal bend (a rest shape, so it persists as the wire
           advances) that makes the tip *enter* the target branch by contact. Empirically
           the wire enters the branch *opposite* the bend (a lever off the crotch), so we
           bend away from the target; the wire then agrees with the route rail and avoids
           unsafe septum crossing on the canonical branch benchmark.

        The shaft follows the route centerline at reset time, so tortuous entries start
        inside the vessel instead of cutting through the first curve."""
        f = self.route_frame
        t0 = f.tangents[0]
        s_nodes = np.minimum(np.arange(n, dtype=float) * float(sp), f.length)
        route_pts, route_m1 = self._route_seed(s_nodes)
        branch_lat = f.tangents[-1] - np.dot(f.tangents[-1], t0) * t0   # branch heading vs trunk
        bl = float(np.linalg.norm(branch_lat))
        if bl < 1e-6 or tip_bend_nodes <= 0:                            # straight route
            return self._clamp_seed_inside_lumen(route_pts + 0.5 * self.R * route_m1)
        branch_lat = branch_lat / bl
        oop = np.cross(t0, branch_lat)                                  # out of the fork plane
        oop_n = float(np.linalg.norm(oop))
        oop = oop / oop_n if oop_n > 1e-6 else f.m1[0]
        # Re-orthogonalise the constant out-of-plane preference against each local
        # route tangent; otherwise it can push through the wall on curved entries.
        oop_local = []
        for tangent, fallback in zip(f.tangents[np.searchsorted(f.cum_s, s_nodes, side="right") - 1],
                                     route_m1):
            v = oop - np.dot(oop, tangent) * tangent
            vn = float(np.linalg.norm(v))
            oop_local.append(v / vn if vn > 1e-6 else fallback)
        seed = route_pts + 0.3 * self.R * np.asarray(oop_local)
        bend = -branch_lat                                             # bend away from the branch
        for k in range(1, min(int(tip_bend_nodes), n - 1) + 1):
            seed[-k] = seed[-k] + bend * tip_bend_mm * (tip_bend_nodes - k + 1)
        return self._clamp_seed_inside_lumen(seed)

    def _features(self):
        """One nearest-edge projection pass over all device nodes → the tip's route
        progress (s, θ) + LOCAL blended radius/edge, plus the deepest penetration and
        max radius across nodes (local-R-aware, so a narrower branch isn't underreported)."""
        pos = self.sim.body_positions()
        projs = [self.tree.project(p) for p in pos]
        rs = self.route_frame.project(pos[-1])           # progress + θ along the route polyline
        tip = projs[-1]
        max_r = max(pr.r for pr in projs)
        max_pen = max(0.0, max(pr.r - pr.R for pr in projs))   # deepest penetration (vs LOCAL R)
        # on_route from PROXIMITY to the route polyline (rs.r), not the nearest-edge id:
        # min-r ownership can flip across the junction band (the documented project()
        # ceiling), but the tip's radial distance from the route path is stable — a tip in
        # the wrong (sibling) branch is many radii from the route, so it reads off-route.
        on_route = rs.r < 2.0 * self.R
        return {"s": rs.s, "r": tip.r, "theta": rs.theta, "R_loc": tip.R,
                "edge": tip.edge_id, "max_r": max_r, "max_pen": max_pen, "on_route": on_route}

    def _obs(self, f):
        Rn = f["R_loc"] if f["R_loc"] > 1e-6 else self.R   # normalize r by the LOCAL branch radius
        return np.array([f["s"] / self.L, f["r"] / Rn, np.sin(f["theta"]), np.cos(f["theta"]),
                         (self.target_s - f["s"]) / self.L], dtype=np.float32)

    def _tip_roll(self):
        # Newton rods store twist around the cable-local z axis; this matches the torsion test helper.
        x, y, z, w = self.sim.body_quaternions()[-1]
        return float(2.0 * np.arctan2(z, w))

    def _parse_action(self, action):
        act = np.asarray(action, dtype=float).reshape(-1)
        if len(act) < 1:
            raise ValueError("action must contain at least an insertion command")
        insertion = float(np.clip(act[0], -1.0, 1.0))
        # Backward compatibility: scalar actions from old policies mean no commanded twist.
        twist = float(np.clip(act[1] if len(act) > 1 else 0.0, -1.0, 1.0))
        return insertion, twist

    def reset(self, *, seed=None, options=None):           # seed accepted for the Gym contract
        from lumen.newton.sim import NewtonGuidewireSim
        if getattr(self, "sim", None) is None:
            # the base follows the WHOLE route polyline so insertion can push past the
            # junction into the target branch (not just up the trunk).
            self.sim = NewtonGuidewireSim(np.asarray(self.route_frame.points), self.R,
                                          self._device_points(), radius=0.2, kappa=3e3, d_hat=0.3,
                                          vbd_iterations=8, device=self.device, tree=self.tree,
                                          route_centerline=np.asarray(self.route_frame.points))
        else:
            self.sim.reset()
        self.steps = 0
        self._prev = abs(self._features()["s"] - self.target_s)
        return self._obs(self._features()), {}

    def step(self, action):
        a, twist = self._parse_action(action)
        self.sim.step(dt=5e-3 * self.substeps, substeps=self.substeps,
                      insertion=a * self.max_insertion, twist=twist * self.max_twist)
        self.steps += 1
        f = self._features()
        obs = self._obs(f)
        if not (np.isfinite(obs).all() and np.isfinite([f["s"], f["r"], f["max_r"]]).all()):
            return np.zeros(5, np.float32), -100.0, True, False, {
                "route_s": 0.0, "dist": 1e6, "max_r": 0.0, "success": False, "diverged": True}
        dist = abs(f["s"] - self.target_s)
        progress = self._prev - dist
        # gate progress when the tip is OFF the route (don't reward advancing the wrong
        # branch — route_s still projects onto the polyline there); small off-route penalty.
        reward = (progress if f["on_route"] else -0.1) - 0.5 * f["max_pen"] - 0.01
        self._prev = dist
        unsafe = bool(f["max_pen"] > getattr(self, "safety_max_pen", 0.3))
        success = bool(dist <= self.success_tol and f["on_route"])
        safe_success = bool(success and not unsafe)
        terminated = bool(success or (unsafe and getattr(self, "terminate_on_unsafe", False)))
        if success:
            reward += 10.0
        if unsafe:
            reward -= 10.0
        info = {"route_s": f["s"], "dist": dist, "max_r": f["max_r"],
                "max_pen": f["max_pen"], "success": success, "safe_success": safe_success,
                "unsafe": unsafe,
                "edge": f["edge"], "off_route": not f["on_route"],
                "twist": twist, "tip_roll": self._tip_roll()}
        return obs, float(reward), terminated, self.steps >= self.max_steps, info
