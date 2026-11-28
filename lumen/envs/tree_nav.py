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
from lumen.envs.nav_gym import (
    parse_nav_action,
    validate_action_scale,
    validate_boolean,
)
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


WIRE_SEG_MM = 2.0            # the simulated wire's segment length
AIM_MIN_TURN_DEG = 15.0      # a route join turning less than this is the vessel running on


def guide_start(tree: VascularTree, route: list[int], start_node: str, guide_radius_mm: float):
    """Where a guide catheter parked in the route leaves the wire: the far end of the last route
    vessel at least ``guide_radius_mm`` wide (mean lumen radius). A microwire is delivered
    through a guide catheter across the large vessels (iliac access, aorta, arch, a wide
    common carotid); it is navigated on its own only beyond them. A narrow access vessel before
    the aorta (a femoral or iliac inlet) does not count: the catheter crosses it too. Returns
    None when the route ends in a wide vessel (a catheter-only route); returns ``start_node``
    when the whole route is narrower than the guide."""
    node, start = start_node, start_node
    for ei in route:
        e = tree.edges[ei]
        node = e.node_b if e.node_a == node else e.node_a
        if float(np.mean(e.lf.R)) >= guide_radius_mm:
            start = node
    if route and float(np.mean(tree.edges[route[-1]].lf.R)) >= guide_radius_mm:
        return None
    return start


# A tip pressed into the wall by up to the contact margin is still in the vessel.
WALL_TOL_MM = 0.5

# Guidewires: radius in mm and bending stiffness per 2 mm joint. The rod solver's bending
# stiffness is per joint and in its own units, so the two are calibrated against each other,
# not to newton-metres: a wire's bending stiffness scales with its diameter to the fourth
# power, and a 0.035 inch wire (with its tapered core) is about twenty times as stiff as a
# 0.016 inch microwire. In a vessel many times its width a microwire bows under a push and
# its tip slides along the wall; an operator crosses the aorta, the iliacs and the arch
# vessels on a 0.035 inch wire and keeps the microwire for the arteries beyond.
WIRES = {"micro": (0.2, 50.0), "0.035": (0.45, 900.0)}
# ``wire="auto"`` takes the 0.035 inch wire when every vessel of the route is at least this
# wide (mean lumen radius, as ``guide_start`` measures it), else the microwire.
WIDE_ROUTE_MIN_R_MM = 3.0


def pick_wire(tree: VascularTree, route: list[int]) -> str:
    """The wire an operator would navigate ``route`` on: ``"0.035"`` when every route vessel
    is at least ``WIDE_ROUTE_MIN_R_MM`` wide, else ``"micro"``."""
    if route and min(float(np.mean(tree.edges[i].lf.R)) for i in route) >= WIDE_ROUTE_MIN_R_MM:
        return "0.035"
    return "micro"


class TreeNavEnv:
    def __init__(self, asset, target_node=None, target_frac=1.0, max_insertion=2.0,
                 max_twist=1.0, substeps=4, max_steps=60, success_tol=2.5,
                 safety_max_pen=0.3, blend_len=4.0, terminate_on_unsafe: bool = False,
                 device=None, device_radius_mm: float | None = None, density: float | None = None,
                 feed_stop_pen_mm: float | None = 0.2, wall_stiffness: float = 6.0e3,
                 wire_nodes: int | None = None, guide_radius_mm: float | None = None,
                 substep_ms: float = 2.5, bend_damping: float = 0.05, tip_deg: float = 45.0,
                 support_gap_mm: float = 20.0, contain_tol_mm: float | None = 0.3,
                 settle_steps: int = 10, quasi_static: bool = True,
                 vbd_iterations: int | None = None, aim_tip: bool = False,
                 start_node: str | None = None, wire: str = "micro",
                 bend_stiffness: float | None = None):
        """``density`` is the wire's mass per cubic millimetre in the solver's units (None
        gives 1e-2, see below). ``feed_stop_pen_mm`` stops feeding, for the rest of a step,
        once any part of the wire presses that far into the wall. The pushed end is driven
        kinematically, so without this a jammed wire is pushed with unlimited force and goes
        through the wall; with it, more push against a jam only loads the wire, as a hand on
        a real wire would feel. None turns the limit off, and with it the wire-tracking
        actuation, the catheter support limit and wall containment described below. ``wall_stiffness`` is the barrier
        stiffness of the vessel wall. Too soft (the old 3e3) and a wire pulled taut through a
        tight bend of a small artery cuts across it; much stiffer (1e4 and up) and, with the
        solver's few iterations per step, the wire sticks at junctions.

        ``wire_nodes`` sets the simulated distal wire (2 mm segments). ``guide_radius_mm`` parks
        a guide catheter in the route: the wire starts beyond the last route vessel at least
        this wide (see ``guide_start``); a route that ends in such a vessel raises ValueError.
        None starts the wire at ``start_node``, by default the asset's spawn point; a scan with
        several cut inflow arteries enters through the one nearest the target (see
        ``lumen.scan.pipeline.access_nodes``).

        ``substep_ms`` is the solver's time step. The rod solver runs a fixed number of
        iterations per step; with 5 ms steps a wire laid into a tight real bend (a carotid
        siphon) does not converge and its joints fold (kinks past 140 degrees, the tip sliding
        back with no input). 2.5 ms keeps the wire's length and shape and passes 0.99 of a push
        through a carotid-siphon S-bend to the tip. ``bend_damping`` is
        the bending relaxation time in seconds (Newton's coefficient is ``bend_damping *
        bend_stiffness``); the shaped tip takes a few steps to swing round after a turn.
        ``tip_deg`` is the shaped tip's rest angle; the rest of the wire rests straight.
        ``density`` defaults to 1e-2 (nitinol is about 6.5e-3 g/mm^3); the solver's default
        of 1 makes the wire so heavy that its inertia stretches and kinks it.

        ``quasi_static`` stops the wire when the hand stops or changes direction (velocities
        cleared at the end of a step with no push or turn, and before a step whose feed
        direction differs from the last). A guidewire in blood is overdamped: it moves with
        the hand. The rod solver has no velocity damping, so without this a wire pulled back
        keeps coasting after the pull ends, folds round its held end, and runs back against
        the next push. While the hand keeps moving one way the velocity is kept: it carries
        the push down the wire between steps.

        ``vbd_iterations`` is the rod solver's iterations per time step (default twice
        ``wire_nodes``). A push reaches about one node further down the wire per iteration, so
        with fewer the pushed end runs ahead of the tip and the wire bows in a wide vessel
        (16 nodes at 8 iterations pass a third of a push to the tip, at 32 about 0.8).

        ``aim_tip`` turns the shaped tip at the start toward the branch the route turns into at
        the next junction (``_tip_aim``), as an operator does before reaching it; a wire that
        is only pushed then enters it. Off (the default), the tip starts in one fixed direction
        whatever the target, so a policy or an experiment has to choose the branch by turning
        the wire. On the arch the probing ``FollowPolicy`` turns its tip into the left subclavian
        from the fixed start and not from the aimed one, so aiming is no help to a policy that
        turns (arriving at the end of that branch is another matter: it turns on microns, see
        ``test_wire_stays_in_vessel``).

        ``support_gap_mm`` is how close the microcatheter behind the wire may come to the
        wire's tip, measured along the route: the pushed end stops there, so a tip that
        is caught and falls back is not fed further into a coil. ``contain_tol_mm`` puts back
        any node that went that far through the wall (None: off). ``settle_steps`` lets the
        wire relax into the vessel, with no input, before the first episode.

        ``wire`` is the guidewire, a key of ``WIRES`` (``"micro"``, the default, or
        ``"0.035"``) or ``"auto"`` for the one an operator would use on the route
        (``pick_wire``). ``device_radius_mm`` and ``bend_stiffness`` override the wire's
        radius and bending stiffness."""
        terminate_on_unsafe = validate_boolean(
            terminate_on_unsafe,
            "terminate_on_unsafe",
        )
        self.asset = asset
        # the wire enters at start_node (default: the asset's spawn); that end is open
        self.start_node = asset.device_spawn.node_id if start_node is None else str(start_node)
        self.tree = VascularTree(asset, blend_len=blend_len, entry_node=self.start_node)
        # default target = a leaf (degree-1 node) that isn't the start
        if target_node is None:
            leaves = [n.id for n in asset.nodes
                      if not self.tree.is_junction(n.id) and n.id != self.start_node]
            target_node = leaves[-1]
        if target_node == self.start_node:
            raise ValueError("target_node must differ from start_node (empty route)")
        self.target_node = target_node
        self.route = self.tree.route(target_node, self.start_node)
        if guide_radius_mm is not None:
            start = guide_start(self.tree, self.route, self.start_node, float(guide_radius_mm))
            if start is None:
                raise ValueError(f"route to {target_node!r} never narrows below the guide "
                                 f"radius {guide_radius_mm} mm (a catheter-only route)")
            if start == target_node:
                raise ValueError(f"the guide catheter reaches {target_node!r} itself")
            self.start_node = start
            self.route = self.tree.route(target_node, self.start_node)
        route_pts = _route_polyline(self.tree, self.route, self.start_node)
        self.route_frame = CenterlineFrame(route_pts)
        self.L = float(self.route_frame.length)
        ends = np.minimum(np.cumsum([self.tree.edges[i].frame.length for i in self.route]), self.L)
        self._route_ranges = list(zip(np.r_[0.0, ends[:-1]], ends))
        # nodes where the route passes from one of its vessels into the next
        self._route_joins = set()
        for i, j in zip(self.route[:-1], self.route[1:]):
            ei, ej = self.tree.edges[i], self.tree.edges[j]
            self._route_joins |= {ei.node_a, ei.node_b} & {ej.node_a, ej.node_b}
        # where the route passes from one vessel into the next (what a roadmap shows)
        self._branch_mm = tuple(float(a) for a, _ in self._route_ranges[1:])
        self.R = float(np.asarray(self.tree.edges[self.route[0]].lf.R).mean())  # entry-edge ref scale
        self.target_s = float(np.clip(target_frac, 0.0, 1.0)) * self.L
        self.max_insertion = validate_action_scale(max_insertion, "max_insertion")
        # How far along the route the tip can honestly move in one step: a few insertion
        # increments. A larger jump went through a wall to a later stretch of the route.
        self._progress_cap = max(10.0, 4.0 * self.max_insertion)
        self.max_twist = validate_action_scale(max_twist, "max_twist")
        self.substeps, self.max_steps = substeps, max_steps
        self.success_tol = success_tol
        self.safety_max_pen = float(safety_max_pen)
        self.terminate_on_unsafe = terminate_on_unsafe
        self.device = device or detect_device()
        if wire == "auto":
            wire = pick_wire(self.tree, self.route)
        if wire not in WIRES:
            raise ValueError(f"wire must be 'auto' or one of {sorted(WIRES)}, not {wire!r}")
        self.wire = wire
        self.device_radius_mm = float(WIRES[wire][0] if device_radius_mm is None
                                      else device_radius_mm)
        self.bend_stiffness = float(WIRES[wire][1] if bend_stiffness is None else bend_stiffness)
        if not (np.isfinite(self.bend_stiffness) and self.bend_stiffness > 0):
            raise ValueError("bend_stiffness must be finite and positive")
        self.density = 1.0e-2 if density is None else float(density)
        if not (np.isfinite(self.density) and self.density > 0):
            raise ValueError("density must be finite and positive")
        self.feed_stop_pen_mm = None if feed_stop_pen_mm is None else float(feed_stop_pen_mm)
        # The wire is laid along the route at reset, so it must fit: 16 nodes (30 mm), or as
        # many 2 mm segments as fit a shorter route.
        fit = int((self.L - 1.0) // WIRE_SEG_MM) + 1
        if wire_nodes is None:
            wire_nodes = min(16, fit)
        self.wire_nodes = int(wire_nodes)
        if self.wire_nodes < 5:
            raise ValueError("wire_nodes must be at least 5 (a shaft and a shaped tip); a route "
                             f"of {self.L:.1f} mm holds {fit}")
        if self.wire_nodes > fit:
            raise ValueError(f"a {self.wire_nodes}-node wire ({(self.wire_nodes - 1) * WIRE_SEG_MM:.0f}"
                             f" mm) does not fit a {self.L:.1f} mm route; use at most {fit}")
        self.substep_ms = float(substep_ms)
        self.bend_damping = float(bend_damping)
        self.tip_deg = float(tip_deg)
        if not (np.isfinite(self.tip_deg) and 0.0 <= self.tip_deg <= 90.0):
            raise ValueError("tip_deg must be between 0 and 90")
        self.support_gap_mm = float(support_gap_mm)
        self.contain_tol_mm = None if contain_tol_mm is None else float(contain_tol_mm)
        if self.contain_tol_mm is not None and not (np.isfinite(self.contain_tol_mm)
                                                     and self.contain_tol_mm >= 0):
            raise ValueError("contain_tol_mm must be finite and non-negative, or None")
        self.settle_steps = int(settle_steps)
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        self.quasi_static = validate_boolean(quasi_static, "quasi_static")
        self.aim_tip = validate_boolean(aim_tip, "aim_tip")
        # a push reaches about one node further down the wire per solver iteration
        self.vbd_iterations = 2 * self.wire_nodes if vbd_iterations is None else int(vbd_iterations)
        if self.vbd_iterations < 1:
            raise ValueError("vbd_iterations must be at least 1")
        for name in ("substep_ms", "bend_damping", "support_gap_mm"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0) or (name == "substep_ms" and v == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'substep_ms' else 'non-negative'}")
        # a wire shorter than the gap could never be fed: the gap is held to the wire's span
        # less two segments (20 mm of a 30 mm wire, as given)
        span = (self.wire_nodes - 1) * WIRE_SEG_MM
        self._support_gap = min(self.support_gap_mm, max(0.0, span - 2 * WIRE_SEG_MM))
        self.wall_stiffness = float(wall_stiffness)
        if not (np.isfinite(self.wall_stiffness) and self.wall_stiffness > 0):
            raise ValueError("wall_stiffness must be finite and positive")
        if not np.isfinite(self.device_radius_mm) or self.device_radius_mm <= 0.0:
            raise ValueError("device_radius_mm must be finite and positive")
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
            limit = max(0.0, float(pr.R) - float(margin_mm) - self.device_radius_mm)
            if pr.r > limit:
                out[i] = p - pr.e_r * (pr.r - limit)
        return out

    def _device_points(self, n=10, sp=2.0, tip_bend_nodes=3, tip_bend_mm=0.7):
        """Shaped-tip guidewire seed that navigates the junction into the target branch.

        A straight, centred wire jams in the crotch of the fork and is then dragged across
        the septum (unphysical wall penetration). Two things fix it, and both matter:

        1. **Symmetric seed** — offset the shaft *out of the bifurcation plane* (not toward
           a branch), so neither branch is favoured by the initial pose.
        2. **Pre-shaped tip** — a distal bend (the rest shape, ``_rest_points``, so it
           persists as the wire advances) turned toward the branch the route takes, as an
           operator turns the tip before reaching a junction. The seed offsets its tip the
           other way, so the shaft starts against the far wall as the tip swings round: on
           the benchmark bifurcation a seed bent toward the branch jams in the crotch, this
           one enters the branch with no wall penetration.

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
        aim = self._tip_aim(seed)
        bend = -(branch_lat if aim is None else aim)                   # away (see 2. above)
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
        max_pen = max(0.0, max(pr.r + self.device_radius_mm - pr.R for pr in projs))
        return {"s": rs.s, "r": tip.r, "theta": rs.theta, "R_loc": tip.R,
                "edge": tip.edge_id, "max_r": max_r, "max_pen": max_pen}

    def _obs(self, f):
        Rn = f["R_loc"] if f["R_loc"] > 1e-6 else self.R   # normalize r by the LOCAL branch radius
        return np.array([f["s"] / self.L, f["r"] / Rn, np.sin(f["theta"]), np.cos(f["theta"]),
                         (self.target_s - f["s"]) / self.L], dtype=np.float32)

    def _tip_roll(self):
        # Newton rods store twist around the cable-local z axis; this matches the torsion test helper.
        quats = self.sim.body_quaternions()
        if len(quats) < 1:
            raise ValueError("body_quaternions must contain at least one guidewire body")
        x, y, z, w = quats[-1]
        return float(2.0 * np.arctan2(z, w))

    def _tip_aim(self, seed):
        """Where the shaped tip is set to point at the start: into the vessel the route turns
        into at the next junction ahead of the tip (by at least ``AIM_MIN_TURN_DEG``), as an
        operator turns the tip toward a branch before reaching it. None when the route runs on
        straight there: a side branch leaves at that junction, and a tip turned toward a later
        branch would take this one."""
        if not self.aim_tip:
            return None
        f = self.route_frame
        s_tip = float(f.project(np.asarray(seed[-1], float)).s)

        def tangent(s):
            return f.tangents[min(int(np.searchsorted(f.cum_s, s, side="right")) - 1,
                                  len(f.tangents) - 1)]
        ahead = [b for b in self._branch_mm if b > s_tip]
        if not ahead:
            return None
        s_b = ahead[0]
        t_in, t_out = tangent(max(s_b - 3.0, 0.0)), tangent(min(s_b + 5.0, float(f.length)))
        lat = t_out - float(t_out @ t_in) * t_in
        n = float(np.linalg.norm(lat))
        return lat / n if n >= np.sin(np.radians(AIM_MIN_TURN_DEG)) else None

    def _rest_points(self, seed=None, sp=2.0, tip_bend_nodes=3):
        """The wire's rest shape: straight, with an angled distal tip (the shaped tip). Given
        the starting pose (``seed``), the tip is bent toward ``_tip_aim``: the rest shape sets
        each segment's bend relative to the one before it, and the wire is laid in along the
        seed's parallel-transport frames, so the bend is turned to the aim in the frame of the
        segment the bend starts from."""
        n = self.wire_nodes
        deg = self.tip_deg
        k = min(int(tip_bend_nodes), n - 1)
        phi = 0.0
        aim = None if seed is None else self._tip_aim(seed)
        if aim is not None and n - k - 2 >= 0:
            import newton
            import warp as wp

            pts = [wp.vec3(*map(float, p)) for p in np.asarray(seed, float)]
            x, y, z, w = (float(c) for c in newton.utils.create_parallel_transport_cable_quaternions(
                pts)[n - k - 2])
            rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            local = rot.T @ aim
            phi = float(np.arctan2(local[1], local[0]))
        pts = np.zeros((n, 3))
        pts[:, 2] = np.arange(n) * sp
        for j in range(1, k + 1):                    # the last k nodes leave the axis at `deg`
            off = j * sp * np.sin(np.radians(deg))
            pts[n - k - 1 + j, 0] = off * np.cos(phi)
            pts[n - k - 1 + j, 1] = off * np.sin(phi)
            pts[n - k - 1 + j, 2] = pts[n - k - 1, 2] + j * sp * np.cos(np.radians(deg))
        return pts

    def _contain(self) -> float:
        """Deepest wall penetration of the wire, after putting back any node that went more than
        `contain_tol_mm` through the wall. The wall barrier is a soft penalty with a finite
        step: a hard push (a caught, compressed wire) can carry a node through it, and once a
        node is on the far side the barrier no longer holds it. A vessel wall does not let a
        wire through, so such a node is moved back to the wall and stopped there."""
        sim = self.sim
        pos = np.asarray(sim.body_positions(), float)
        if not np.isfinite(pos).all():          # blown up: step() ends the episode as diverged
            return 0.0
        worst, moved = 0.0, {}
        for i, p in enumerate(pos):
            pr = self.tree.project(p)
            pen = float(pr.r + self.device_radius_mm - pr.R)
            if not np.isfinite(pen):
                continue
            worst = max(worst, pen)
            if i > 0 and self.contain_tol_mm is not None and pen > self.contain_tol_mm:
                moved[i] = (pen, np.asarray(pr.e_r, float))
        if moved:
            ids = [sim.bodies[i] for i in moved]
            q = sim.state_0.body_q.numpy()
            prev = getattr(sim.solver, "body_q_prev", None)    # absent before the first step
            qp = None if prev is None else prev.numpy()
            qd = sim.state_0.body_qd.numpy()
            for b, (pen, n) in zip(ids, moved.values()):
                # back onto the wall, without the outward part of its motion; sliding along the
                # wall is kept (the wall is frictionless here)
                q[b, :3] = q[b, :3] - pen * n
                if qp is not None:
                    v = q[b, :3] + pen * n - qp[b, :3]
                    vn = max(0.0, float(v @ n))
                    qp[b, :3] = q[b, :3] - (v - vn * n)
                w = float(qd[b, :3] @ n) if qd.shape[1] >= 3 else 0.0
                if w > 0:
                    qd[b, :3] = qd[b, :3] - w * n
            sim.state_0.body_q.assign(q)
            if qp is not None:
                prev.assign(qp)
            sim.state_0.body_qd.assign(qd)
        return max(0.0, worst)

    def _parse_action(self, action):
        return parse_nav_action(action)

    def reset(self, *, seed=None, options=None):           # seed accepted for the Gym contract
        from lumen.newton.sim import NewtonGuidewireSim
        if getattr(self, "sim", None) is None:
            seed = self._device_points(n=self.wire_nodes)
            # the base follows the WHOLE route polyline so insertion can push past the
            # junction into the target branch (not just up the trunk).
            self.sim = NewtonGuidewireSim(np.asarray(self.route_frame.points), self.R,
                                          seed,
                                          radius=self.device_radius_mm,
                                          kappa=self.wall_stiffness, d_hat=0.3,
                                          vbd_iterations=self.vbd_iterations,
                                          rest_points=self._rest_points(seed),
                                          bend_damping=self.bend_damping,
                                          device=self.device, tree=self.tree,
                                          route_centerline=np.asarray(self.route_frame.points),
                                          density=self.density,
                                          bend_stiffness=self.bend_stiffness)
            self._settle()
        self.sim.reset()
        self.steps = 0
        self._wall_load_impulse = 0.0
        # where the pushed end has been: it can be pulled back along this path
        b0 = np.asarray(self.sim.body_positions()[0], float)
        e0 = np.asarray(self.route_frame.points[0], float)
        self._base_path = [e0, b0] if np.linalg.norm(b0 - e0) > 0.25 else [b0]
        tip = np.asarray(self.sim.body_positions()[-1], float)
        s0 = self._route_s(tip, near=0.0)
        self._tip_best = self._tip_s = self.route_frame.project(tip).s if s0 is None else s0
        self._tip_s_age = 0
        self._step_pen = 0.0
        self._feed = 0.0
        self._prev = abs(self._tip_s - self.target_s)
        return self._obs(self._features()), {}

    def _solver_substeps(self, dt: float) -> int:
        return max(1, int(round(dt * 1e3 / self.substep_ms)))

    def _settle(self) -> None:
        """Let the wire, laid along the centreline, relax into the vessel with no input, and make
        that the state every episode starts from. A straight-resting wire laid into a curved
        vessel is not at rest there; without this each episode would begin with it springing."""
        dt = 5e-3 * self.substeps
        for _ in range(max(0, self.settle_steps)):
            self.sim.step(dt=dt, substeps=self._solver_substeps(dt), insertion=0.0, twist=0.0)
            if self.feed_stop_pen_mm is not None:
                self._contain()
        if self.settle_steps > 0:
            self.sim._init_body_q = self.sim.state_0.body_q.numpy().copy()

    def _set_rail_along_wire(self) -> None:
        """Move the pushed end along the wire itself, as a catheter tracks over a wire: the rail
        is the path the end has come along, then the wire's own current shape to (2 mm short
        of) the tip. A push then drives the wire along its length; it cannot carry the wire
        into a vessel the tip did not enter, nor past a tip that is caught (that buckles the
        wire instead, as it would). Pulling back retraces the path."""
        pos = np.asarray(self.sim.body_positions(), float)
        path = [p for p in self._base_path]
        if np.linalg.norm(path[-1] - pos[0]) > 1e-3:
            path.append(pos[0])
        # only the part of the wire that keeps heading on along the route: past a point where it
        # turns back on itself, pushing would walk the pushed end backwards (a shaft cannot
        # kink back at the catheter tip); the push stops there until a pull-back straightens it
        s_prev, ahead = self.route_frame.project(pos[0]).s - 2.0, []
        for p in pos[1:]:
            s = self.route_frame.project(p).s
            if s < s_prev - 2.0:
                break
            ahead.append(p)
            s_prev = max(s_prev, s)
        rail = np.asarray(path + ahead, float)
        # a kinematic end cannot be corrected by contact: keep its rail inside the lumen
        margin = float(self.sim.device_radius) + 0.1
        for i in range(len(rail)):
            tp = self.tree.project(rail[i])
            over = tp.r - (tp.R - margin)
            if over > 0.0 and np.isfinite(over):
                rail[i] = rail[i] - over * np.asarray(tp.e_r, float)
        keep = np.r_[True, np.linalg.norm(np.diff(rail, axis=0), axis=1) > 1e-4]
        rail = rail[keep]
        n_path = int(np.count_nonzero(keep[:len(path)]))   # the path's points left after merging
        seg = np.linalg.norm(np.diff(rail, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        end = max(cum[n_path - 1], cum[-1] - 2.0)          # stop short of the tip
        k = int(np.searchsorted(cum, end))
        if k < len(rail) and cum[k] > end:
            u = (end - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-9)
            rail = np.vstack([rail[:k], rail[k - 1] + u * (rail[k] - rail[k - 1])])
        if len(rail) >= 2:
            self.sim.solver.set_actuation_rail(rail)

    def _route_s(self, p, near: float | None = None):
        """Arc length along the route of a point inside a route vessel's lumen, else None.

        Ownership in ``tree.project`` goes to the edge with the most room, so where a short side
        vessel or a skeleton spur opens inside the route's own lumen a point still in the route
        vessel can be owned by the side vessel; progress must not stop there. The arc length is
        measured on the containing vessel's own stretch of the route. Where two stretches of a
        coiled route overlap, the one nearest where the tip last was (``near``, default the
        tracked tip position) wins."""
        p = np.asarray(p, float)
        f = self.route_frame
        cands = []
        for i, (a, b) in zip(self.route, self._route_ranges):
            e = self.tree.edges[i]
            pr = e.frame.project(p)
            end = e.node_a if pr.s <= 1e-6 else e.node_b if pr.s >= e.frame.length - 1e-6 else None
            # past an end the projection stops at the end point: the point's distance is to
            # that point, not to the centreline carried on beyond it
            r = pr.r if end is None else float(np.linalg.norm(
                p - np.asarray(e.frame.points[0 if pr.s <= 1e-6 else -1], float)))
            if not r < self.tree.blended_R(i, pr.s, pr.theta) + WALL_TOL_MM:
                continue
            idx = np.flatnonzero((f.cum_s >= a - 1e-6) & (f.cum_s <= b + 1e-6))
            if len(idx) < 2:
                continue
            sub = CenterlineFrame(f.points[idx[0]:idx[-1] + 1])
            # Past an end where the route goes on into its next vessel, the point belongs to
            # that vessel: a projection clamped there only counts if no vessel holds the point
            # inside its own length (the outside of a sharp corner, past both ends).
            weak = end is not None and end in self._route_joins
            cands.append((weak, float(f.cum_s[idx[0]] + sub.project(p).s)))
        if not cands:
            return None
        strong = [s for weak, s in cands if not weak]
        ref = getattr(self, "_tip_s", self._tip_best) if near is None else near
        return min(strong or [s for _, s in cands], key=lambda s: abs(s - ref))

    def _wire_lies_along_route(self, s_new: float) -> bool:
        """Whether the wire itself shows the way from the last tracked place to ``s_new``.
        A reading further on than one step can carry the tip is believed only if, walking back
        from the tip along the wire, every node is inside a route vessel and each is a segment
        or so behind the one before, down to one near the last tracked place. A tip that went
        through a wall into a later stretch of the route, or that is read there while the wire
        stays where it was, fails; waiting there does not make it pass."""
        prev = float(s_new)
        for p in np.asarray(self.sim.body_positions(), float)[-2::-1]:
            s = self._route_s(p, near=prev)
            if s is None or abs(s - prev) > 1.5 * WIRE_SEG_MM + WALL_TOL_MM:
                return False
            if abs(s - self._tip_s) <= self._progress_cap:
                return True
            prev = s
        return False

    def _record_base_path(self) -> None:
        """Trim the path at the pushed end's place on it (after a pull-back) and extend it."""
        p = np.asarray(self.sim.body_positions()[0], float)
        if len(self._base_path) >= 2:
            fr = CenterlineFrame(np.asarray(self._base_path, float))
            s = fr.project(p).s
            keep = [q for q, cs in zip(self._base_path, fr.cum_s) if cs <= s + 1e-6]
            self._base_path = (keep or [self._base_path[0]])
        if np.linalg.norm(self._base_path[-1] - p) > 0.25:
            self._base_path.append(p)

    def _diverged(self):
        """The terminal step of an episode whose simulation blew up: off the route, no reach."""
        return np.zeros(5, np.float32), -100.0, True, False, {
            "route_s": 0.0, "tip_mm": float(self._tip_s), "furthest_mm": float(self._tip_best),
            "dist": 1e6, "max_r": 0.0, "max_pen": 0.0,
            "wall_load_max": 0.0, "wall_load_sum": 0.0,
            "wall_pressure_max": 0.0, "wall_load_impulse": 0.0,
            "wall_load_units": "sim_units",
            "success": False, "diverged": True, "off_route": True}

    def step(self, action):
        a, twist = self._parse_action(action)
        insertion, turn = a * self.max_insertion, twist * self.max_twist
        split = None
        # deepest wall penetration seen in this step, before containment puts nodes back
        self._step_pen = 0.0
        if insertion > 0.0 and self.feed_stop_pen_mm is not None:
            # the route ends where the scan's vessel ends: a tip already there gets no more
            # push, and the pushed end never runs past the end of the route. The stop sits
            # inside the goal band, so a target at the very end can still be reached.
            stop = self.L - min(1.0, 0.5 * float(self.success_tol))
            pos = self.sim.body_positions()
            base_s = self.route_frame.project(pos[0]).s
            if self._tip_s >= stop:
                insertion = 0.0
            insertion = float(np.clip(stop - base_s, 0.0, insertion))
        if self.feed_stop_pen_mm is not None:
            self._set_rail_along_wire()
            if insertion > 0.0:
                # the microcatheter behind the wire stays support_gap_mm behind the tip's
                # place on the route: a caught tip that falls back stops the feed
                tip = np.asarray(self.sim.body_positions()[-1], float)
                ts = self._route_s(tip)
                # off the route the tip's place is where it was last tracked: the catheter is
                # held behind that, so a tip that left the route (into a side vessel or
                # through a wall) is not followed further, and its return is fed from there
                ts = self._tip_s if ts is None else ts
                insertion = float(np.clip(ts - self._support_gap - base_s, 0.0, insertion))
        feed = float(np.sign(insertion))
        if self.quasi_static and feed != self._feed:
            # the hand changed direction (or started or stopped feeding): the wire's motion
            # from the last step does not carry on into this one
            self.sim.state_0.body_qd.zero_()
        self._feed = feed
        if (insertion > 0.0 or turn != 0.0) and self.feed_stop_pen_mm is not None:
            # push and turn a piece per substep, only while the wire is not pressing into the
            # wall (turning a shaped tip that is already in the wall swings it through). Each
            # sim.step reports its own peaks and impulse, so they are combined over the pieces.
            n = max(1, int(self.substeps))
            split = {"load": 0.0, "pressure": 0.0, "impulse": 0.0, "sum": 0.0}
            for _ in range(n):
                pen = self._contain()
                self._step_pen = max(self._step_pen, pen)
                free = pen <= self.feed_stop_pen_mm
                piece = insertion / n if free or insertion < 0 else 0.0
                dt = 5e-3 * self.substeps / n
                self.sim.step(dt=dt, substeps=self._solver_substeps(dt),
                              insertion=piece, twist=(turn / n if free else 0.0))
                if hasattr(self.sim, "wall_load_max"):
                    split["load"] = max(split["load"], float(self.sim.wall_load_max()))
                if hasattr(self.sim, "wall_pressure_max"):
                    split["pressure"] = max(split["pressure"], float(self.sim.wall_pressure_max()))
                if hasattr(self.sim, "wall_load_impulse"):
                    split["impulse"] += float(self.sim.wall_load_impulse())
                if hasattr(self.sim, "wall_load_grid"):     # the largest total over the pieces
                    split["sum"] = max(split["sum"], float(np.sum(self.sim.wall_load_grid())))
        else:
            dt = 5e-3 * self.substeps
            self.sim.step(dt=dt, substeps=self._solver_substeps(dt), insertion=insertion, twist=turn)
        if not np.isfinite(np.asarray(self.sim.body_positions(), float)).all():
            return self._diverged()
        if self.quasi_static and insertion == 0.0 and turn == 0.0:
            self.sim.state_0.body_qd.zero_()
        if self.feed_stop_pen_mm is not None:
            self._step_pen = max(self._step_pen, self._contain())
            self._record_base_path()
        # route progress: tracked step to step, so a reading that jumped further than the tip
        # can travel since it was last seen in the route went through a wall
        tip = np.asarray(self.sim.body_positions()[-1], float)
        s_tip = self._route_s(tip)
        if s_tip is not None and (abs(s_tip - self._tip_s) <= self._progress_cap
                                  or self._wire_lies_along_route(s_tip)):
            self._tip_s, self._tip_s_age = s_tip, 0
            self._tip_best = max(self._tip_best, s_tip)
        else:
            self._tip_s_age += 1
        self.steps += 1
        f = self._features()
        f["max_pen"] = max(f["max_pen"], self._step_pen)
        obs = self._obs(f)
        if not (np.isfinite(obs).all() and np.isfinite([f["s"], f["r"], f["max_r"]]).all()):
            return self._diverged()
        # The tip is on the route when it is inside a route vessel (read and tracked this step).
        # Distance, reward and success use that tracked place, not the nearest point of the
        # route's centreline: a tip in the wrong branch or across a coil can lie close to the
        # centreline far ahead, and must not be paid for it.
        on_route = self._tip_s_age == 0
        dist = abs(self._tip_s - self.target_s)
        progress = self._prev - dist
        reward = (progress if on_route else -0.1) - 0.5 * f["max_pen"] - 0.01
        self._prev = dist
        unsafe = bool(f["max_pen"] > getattr(self, "safety_max_pen", 0.3))
        success = bool(dist <= self.success_tol and on_route)
        safe_success = bool(success and not unsafe)
        terminated = bool(success or (unsafe and getattr(self, "terminate_on_unsafe", False)))
        if success:
            reward += 10.0
        if unsafe:
            reward -= 10.0
        load_grid = (
            np.asarray(self.sim.wall_load_grid(), dtype=float)
            if hasattr(self.sim, "wall_load_grid") else np.zeros(0, dtype=float)
        )
        if split is not None and hasattr(self.sim, "wall_load_max"):
            wall_load_max = split["load"]
        elif hasattr(self.sim, "wall_load_max"):
            wall_load_max = float(self.sim.wall_load_max())
        else:
            wall_load_max = float(load_grid.max()) if load_grid.size else 0.0
        wall_load_sum = float(load_grid.sum()) if load_grid.size else 0.0
        if split is not None and load_grid.size:
            wall_load_sum = max(wall_load_sum, split["sum"])
        wall_pressure_max = (
            (split["pressure"] if split is not None else float(self.sim.wall_pressure_max()))
            if hasattr(self.sim, "wall_pressure_max") else 0.0
        )
        if hasattr(self.sim, "wall_load_impulse"):
            self._wall_load_impulse = getattr(self, "_wall_load_impulse", 0.0) + (
                split["impulse"] if split is not None else float(self.sim.wall_load_impulse())
            )
        else:
            self._wall_load_impulse = getattr(self, "_wall_load_impulse", 0.0) + (
                wall_load_sum * 5e-3 * self.substeps
            )
        info = {"route_s": f["s"], "tip_mm": float(self._tip_s), "furthest_mm": float(self._tip_best),
                "branch_mm": self._branch_mm,
                "dist": dist, "max_r": f["max_r"],
                "max_pen": f["max_pen"], "wall_load_max": wall_load_max,
                "wall_load_sum": wall_load_sum, "wall_pressure_max": wall_pressure_max,
                "wall_load_impulse": float(self._wall_load_impulse),
                "wall_load_units": "sim_units",
                "success": success, "safe_success": safe_success,
                "unsafe": unsafe, "edge": f["edge"], "off_route": not on_route,
                "twist": twist, "tip_roll": self._tip_roll()}
        return obs, float(reward), terminated, self.steps >= self.max_steps, info
