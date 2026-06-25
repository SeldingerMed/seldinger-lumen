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
    __slots__ = ("id", "node_a", "node_b", "frame", "lf")

    def __init__(self, edge, asset):
        pts, lf = asset.edge_arrays(edge)
        self.id = edge.id
        self.node_a, self.node_b = edge.node_a, edge.node_b
        self.frame = CenterlineFrame(np.asarray(pts, float))
        self.lf = lf


class VascularTree:
    """Frame over an `Asset`'s edge graph with branch-node-aware R.

    `blend_len` is the arc-length over which R relaxes from the junction radius to an
    edge's own R near a shared node (mm). Set 0 to disable blending (sharp junctions).
    """

    def __init__(self, asset, blend_len: float = 4.0):
        if not asset.edges:
            raise ValueError("asset has no edges")
        self.edges = [_Edge(e, asset) for e in asset.edges]
        blend_len = float(blend_len)
        if blend_len < 0:
            raise ValueError(f"blend_len must be >= 0, got {blend_len}")
        self.blend_len = blend_len
        self._node_pos = {n.id: np.asarray(n.position_mm, float) for n in asset.nodes}
        # degree = how many edges touch a node; >1 marks a junction
        self._degree: dict[str, int] = {}
        for e in self.edges:
            for nid in (e.node_a, e.node_b):
                self._degree[nid] = self._degree.get(nid, 0) + 1

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
        R_edge = float(e.lf.eval(s, theta))
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

    # --- projection ----------------------------------------------------------
    def project(self, p) -> TreeProjection:
        """Project a world point onto the nearest edge; R/gap are branch-blended.

        Nearest = smallest radial distance r (the point belongs to whichever lumen
        contains it); near a junction the blended R keeps the gap MAGNITUDE continuous.
        ponytail ceiling: at a Y the min-r winner can be the wrong branch, so the
        contact NORMAL e_r may point off the true lumen's axis within the junction
        band — fine for navigation, revisit (min-gap ownership / averaged normals) if
        junction contact accuracy matters. Also, like CenterlineFrame, a point past an
        open vessel end clips to the tip (reported inside); the contact kernel culls
        open ends, but tree.gap does not flag axial-beyond."""
        p = np.asarray(p, float)
        best_i, best_pr = 0, None
        for i, e in enumerate(self.edges):
            pr = e.frame.project(p)
            if best_pr is None or pr.r < best_pr.r:
                best_i, best_pr = i, pr
        R = self.blended_R(best_i, best_pr.s, best_pr.theta)
        return TreeProjection(edge_id=self.edges[best_i].id, edge_index=best_i,
                              s=best_pr.s, theta=best_pr.theta, r=best_pr.r,
                              e_r=best_pr.e_r, R=R, gap=R - best_pr.r)

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
