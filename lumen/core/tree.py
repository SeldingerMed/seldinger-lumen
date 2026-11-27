"""Vascular tree — a graph of centerline edges sharing branch nodes (doc §3.5.2).

A single `CenterlineFrame` is one tube. Real anatomy is a *tree*: a trunk splitting
into branches (Circle-of-Willis, coronary tree, peripheral runoff). This wraps the
per-edge frames + the shared `R(s,θ)` fields of an `Asset` graph and answers the two
things contact and navigation need across a junction:

  * `project(p)` — which edge owns a world point, and its tube-intrinsic (s, θ, r).
  * a `gap`/`radius` that is **continuous across a branch node**: the bible (§3.5.2)
    blends R near bifurcations so the lumen doesn't have a step/notch where a wide
    trunk meets a narrower branch. We blend toward the junction radius (the widest
    meeting vessel — a bifurcation bulges) over a short `blend_len`, tapering to each
    edge's own R away from the node.

Pure geometry/numpy — the Newton contact kernel (per-node edge assignment) builds on
this; nothing here imports the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumen.core.frame import CenterlineFrame


def is_open_entry(tree, node_id) -> bool:
    """The one open-end rule shared by the contact kernel's edge flags and the wall fit: the
    open vessel end the wire enters through has no wall past it. A tree that does not say where
    the wire enters keeps every free end open."""
    entry = getattr(tree, "entry_node", None)
    return not tree.is_junction(node_id) and (entry is None or node_id == entry)


@dataclass
class TreeProjection:
    edge_id: str          # which edge owns the point
    edge_index: int       # its index in tree.edges
    s: float              # arc-length along that edge
    theta: float          # circumferential angle
    r: float              # radial distance from the edge centerline
    e_r: np.ndarray       # unit radial direction (world)
    R: float              # blended lumen radius at the projection (branch-aware)
    gap: float            # R - r (>0 clearance, <=0 penetration)


class _Edge:
    __slots__ = ("id", "node_a", "node_b", "frame", "lf", "R_max")

    def __init__(self, edge, asset):
        pts, lf = asset.edge_arrays(edge)
        self.id = edge.id
        self.node_a, self.node_b = edge.node_a, edge.node_b
        self.frame = CenterlineFrame(np.asarray(pts, float))
        self.lf = lf
        self.R_max = float(np.max(lf.R))          # bound for skipping edges in project()


CAP_CLOSING = 0.15     # an end whose radius falls by more than this is a scan cap


def _mean_R(e, s: float) -> float:
    return float(np.mean([e.lf.eval(s, t) for t in np.linspace(0.0, 2 * np.pi, 16, endpoint=False)]))


class VascularTree:
    """Frame over an `Asset`'s edge graph with branch-node-aware R.

    `blend_len` is the arc-length over which R relaxes from the junction radius to an
    edge's own R near a shared node (mm). Set 0 to disable blending (sharp junctions).
    """

    def __init__(self, asset, blend_len: float = 4.0, entry_node: str | None = None):
        if not asset.edges:
            raise ValueError("asset has no edges")
        self.edges = [_Edge(e, asset) for e in asset.edges]
        blend_len = float(blend_len)
        if not np.isfinite(blend_len) or blend_len < 0:
            raise ValueError(f"blend_len must be >= 0 and finite, got {blend_len}")
        self.blend_len = blend_len
        self._node_pos = {n.id: np.asarray(n.position_mm, float) for n in asset.nodes}
        spawn = getattr(asset, "device_spawn", None)
        # where the wire comes in: the asset's spawn unless the caller enters elsewhere
        self.entry_node = entry_node if entry_node is not None else getattr(spawn, "node_id", None)
        if self.entry_node is not None and self.entry_node not in self._node_pos:
            what = "entry node" if entry_node is not None else "the asset's device spawn node"
            raise ValueError(f"{what} {self.entry_node!r} is not in the asset")
        # degree = how many edges touch a node; >1 marks a junction
        self._degree: dict[str, int] = {}
        for e in self.edges:
            for nid in (e.node_a, e.node_b):
                self._degree[nid] = self._degree.get(nid, 0) + 1
        # A scanned artery that runs off the edge of the scan (or where the segmentation
        # stopped) ends in a rounded cap: the stored radius closes toward a point over the last
        # ~1.5 radii. The vessel does not really end there, so over that cap the radius holds
        # at its value just before it. Only an end that really closes is held (by more than
        # CAP_CLOSING over that stretch): on the shipped scans most ends close by 25-98%, while
        # generated anatomy's tapered outlets narrow by under 5% there and stay as built. The
        # entry end is held too: it is open (no wall past it, see is_open_entry), and the wire
        # starts there, so a closing cap would squeeze it out of the vessel.
        self._caps: list[list[tuple[float, float]]] = []
        for e in self.edges:
            caps = []
            r_med = float(np.median(np.asarray(e.lf.R, float)))
            span = min(1.5 * r_med, 0.3 * float(e.frame.length))
            for end_s, nid, sign in ((0.0, e.node_a, 1.0), (float(e.frame.length), e.node_b, -1.0)):
                if self._degree.get(nid, 0) == 1 and span > 0:
                    ref_s = end_s + sign * span
                    if _mean_R(e, end_s) < (1.0 - CAP_CLOSING) * _mean_R(e, ref_s):
                        caps.append((end_s, ref_s))
            self._caps.append(caps)
        # project() skips an edge whose widest possible radius cannot beat the current owner;
        # near a junction the blend can widen an edge to the junction radius
        self._R_bound = [max([e.R_max] + [self._junction_R(n) for n in (e.node_a, e.node_b)
                                          if self._degree.get(n, 0) > 1])
                         for e in self.edges]

    # --- junction radius -----------------------------------------------------
    def _edge_end_R(self, e: _Edge, node_id: str) -> float:
        """Edge e's lumen radius at the end that touches node_id (s=0 or s=L).

        Averages over theta to properly account for non-axisymmetric sections."""
        s_end = 0.0 if node_id == e.node_a else e.frame.length
        # sample the full end-section profile to capture angular variation
        theta_samples = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        radii = [e.lf.eval(s_end, th) for th in theta_samples]
        return float(np.mean(radii))

    def _junction_R(self, node_id: str) -> float:
        """Widest meeting vessel at a node — the lumen bulges to at least this at the
        junction, so no branch introduces a notch narrower than an inlet."""
        return max(self._edge_end_R(e, node_id)
                   for e in self.edges if node_id in (e.node_a, e.node_b))

    def blended_R(self, ei: int, s: float, theta: float) -> float:
        """Branch-blended lumen radius at (edge ei, s, theta) — public so the contact
        baker (set_tree_contact) reads it without reaching into a private method.

        ponytail: linear taper (C0; the gap is continuous but its slope kinks at
        blend_len). Smoothstep (3w²−2w³) if a penalty-force kink ever shows in
        dynamics."""
        e = self.edges[ei]
        R_edge = self._edge_R(ei, s, theta)
        if self.blend_len <= 0:
            return R_edge
        best = R_edge
        for end_s, nid in ((0.0, e.node_a), (e.frame.length, e.node_b)):
            if self._degree.get(nid, 0) > 1:              # a junction end
                d = abs(s - end_s)
                if d < self.blend_len:
                    w = d / self.blend_len                # 0 at the node, 1 at blend_len
                    cand = w * R_edge + (1.0 - w) * self._junction_R(nid)
                    best = max(best, cand)                # bulge wins if two junctions overlap
        return best

    def _edge_R(self, ei: int, s: float, theta: float) -> float:
        """Edge ei's own radius at (s, theta), held open over a free end's cap."""
        lf = self.edges[ei].lf
        R = float(lf.eval(s, theta))
        for end_s, ref_s in self._caps[ei]:
            if abs(s - end_s) < abs(ref_s - end_s):
                R = max(R, float(lf.eval(ref_s, theta)))
        return R

    # --- projection ----------------------------------------------------------
    def project(self, p) -> TreeProjection:
        """Project a world point onto the edge whose lumen holds it best; R/gap are
        branch-blended.

        Owner = the edge where the point has the most room (radius minus distance from the
        axis), so a point is inside the vessel when it is inside ANY branch. The Newton tree
        kernel compares the same room with the same blended radius; past an edge's end the two
        still differ (here r is measured across the end's extended axis, the kernel uses the
        distance to the end point), see issue #153. Nearest-axis
        ownership got this wrong where a thin branch leaves a wide vessel: a point well inside
        the wide lumen could be nearer the thin axis and be judged against the thin wall.
        Near a junction the blended R keeps the gap MAGNITUDE continuous. Also, like CenterlineFrame, a point past an
        open vessel end clips to the tip (reported inside); the contact kernel culls
        open ends, but tree.gap does not flag axial-beyond."""
        p = np.asarray(p, float)
        if not np.isfinite(p).all():      # a diverged wire: report NaNs so the env's
            nan = float("nan")            # divergence guard sees them, rather than crashing
            return TreeProjection(edge_id=self.edges[0].id, edge_index=0, s=nan, theta=nan,
                                  r=nan, e_r=np.full(3, nan), R=nan, gap=nan)
        best_i, best_pr, best_need, best_R = 0, None, np.inf, 0.0
        for i, e in enumerate(self.edges):
            pr = e.frame.project(p)
            if pr.r - self._R_bound[i] > best_need:
                continue                       # cannot beat the current owner
            R = self.blended_R(i, pr.s, pr.theta)   # the radius the gap and the wall force use
            need = pr.r - R
            if need < best_need:
                best_i, best_pr, best_need, best_R = i, pr, need, R
        return TreeProjection(edge_id=self.edges[best_i].id, edge_index=best_i,
                              s=best_pr.s, theta=best_pr.theta, r=best_pr.r,
                              e_r=best_pr.e_r, R=best_R, gap=best_R - best_pr.r)

    def project_edge_s(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized nearest tree edge and local arc-length for many points.

        This intentionally returns only ``edge_index`` and ``s`` for hot paths such
        as tree-flow drag sampling. Full ``project()`` still computes θ/r/e_r/R/gap
        for contact and diagnostics, but looping through that richer object per
        solver node is unnecessary when flow only needs env×edge×s addresses.
        """
        P = np.asarray(points, dtype=float)
        if P.shape == (3,):
            leading = ()
            flat = P[None, :]
        elif P.ndim >= 2 and P.shape[-1] == 3:
            leading = P.shape[:-1]
            flat = P.reshape(-1, 3)
        else:
            raise ValueError("points must have shape (..., 3)")
        all_s = []
        all_d2 = []
        for e in self.edges:
            s, d2 = self._project_s_d2(e.frame, flat)
            all_s.append(s)
            all_d2.append(d2)
        d2_stack = np.stack(all_d2, axis=1)
        s_stack = np.stack(all_s, axis=1)
        rows = np.arange(flat.shape[0])
        edge = np.argmin(d2_stack, axis=1).astype(np.int32)
        s = s_stack[rows, edge].astype(np.float32)
        return edge.reshape(leading), s.reshape(leading)

    @staticmethod
    def _project_s_d2(frame: CenterlineFrame, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-segment arc length and squared distance for a batch of points."""
        a = frame.points[:-1]
        ab = frame.points[1:] - a
        denom = np.einsum("mj,mj->m", ab, ab)
        denom[denom == 0] = 1.0
        num = np.einsum("nj,mj->nm", points, ab) - np.einsum("mj,mj->m", a, ab)[None]
        u = np.clip(num / denom[None], 0.0, 1.0)
        foot = a[None] + u[:, :, None] * ab[None]
        diff = points[:, None, :] - foot
        d2 = np.einsum("nmj,nmj->nm", diff, diff)
        j = np.argmin(d2, axis=1)
        rows = np.arange(points.shape[0])
        seglen = np.linalg.norm(ab[j], axis=1)
        s = frame.cum_s[j] + u[rows, j] * seglen
        return s, d2[rows, j]

    def gap(self, p) -> float:
        return self.project(p).gap

    def is_junction(self, node_id: str) -> bool:
        return self._degree.get(node_id, 0) > 1

    def route(self, target_node: str, start_node: str) -> list[int]:
        """Edge indices forming the path from `start_node` to `target_node` (BFS over the
        edge graph). Raises if unreachable. Used to define a navigation target down a
        specific branch."""
        from collections import deque
        adj: dict[str, list[tuple[int, str]]] = {}
        for i, e in enumerate(self.edges):
            adj.setdefault(e.node_a, []).append((i, e.node_b))
            adj.setdefault(e.node_b, []).append((i, e.node_a))
        # validate both endpoints BEFORE the BFS: an unknown target==start would otherwise
        # return a false empty path (the start==target short-circuit) instead of failing.
        for nid in (start_node, target_node):
            if nid not in adj:
                raise ValueError(f"unknown node {nid!r} (not in the tree's edge graph)")
        seen = {start_node}
        queue: deque[tuple[str, list[int]]] = deque([(start_node, [])])
        while queue:
            node, path = queue.popleft()         # deque: O(1) pop, not list.pop(0)'s O(n)
            if node == target_node:
                return path
            for ei, other in adj.get(node, []):
                if other not in seen:
                    seen.add(other)
                    queue.append((other, path + [ei]))
        raise ValueError(f"no route from {start_node!r} to {target_node!r}")

    def route_length(self, route: list[int]) -> float:
        return float(sum(self.edges[i].frame.length for i in route))


if __name__ == "__main__":  # self-check (pure numpy): projection + branch continuity
    from lumen.assets import procedural

    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=35.0)
    tree = VascularTree(asset, blend_len=4.0)
    apex = np.asarray([n.position_mm for n in asset.nodes if n.id == "apex"][0], float)

    # a point just inside the trunk projects to the trunk edge
    assert tree.project(apex - np.array([0, 0, 10.0])).edge_id == "trunk"
    # a point up a branch projects to that branch
    onleft = apex + 10.0 * np.array([-np.sin(np.radians(35)), 0.0, np.cos(np.radians(35))])
    assert tree.project(onleft).edge_id in ("left", "right")

    # R is continuous across the junction: sample R just-before (trunk) and just-after
    # (branch) the apex along the centerline; without blending it would step 2.0 -> 1.6.
    before = tree.project(apex - np.array([0, 0, 0.5])).R       # trunk side, near apex
    after = tree.project(onleft * 0.0 + apex
                         + 0.5 * np.array([-np.sin(np.radians(35)), 0, np.cos(np.radians(35))])).R
    assert abs(before - after) < 0.2, (before, after)          # no big step at the junction
    assert before > 1.7, before                                # bulged toward the trunk radius

    # far down a branch, R relaxes to the branch's own (narrower) radius
    far = tree.project(apex + 30.0 * np.array([-np.sin(np.radians(35)), 0, np.cos(np.radians(35))])).R
    assert abs(far - 1.6) < 0.05, far
    print("vascular tree self-check ok")
