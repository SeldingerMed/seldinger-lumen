"""Narrowphase throughput: tube-intrinsic vs generic mesh-distance (doc M0).

The doc's core speed claim (§3.5.4): because the device is a slender rod in a
tube, the contact narrowphase collapses to a near-analytic per-node query in
tube-intrinsic coordinates -- O(active nodes) -- instead of a generic
point-vs-surface query that scales with the mesh discretisation of the wall.

This benchmark measures both on the same scene:
  * tube-intrinsic: project nodes -> (s,theta,r), gap = R(s) - r  (analytic)
  * generic mesh:   gap = signed distance from each node to a triangulated tube
                    surface (M segments x T circumferential vertices), brute force

The generic cost grows with the circumferential resolution T; the tube-intrinsic
cost does not. Run:  python -m benchmarks.throughput
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lumen.core.lumen_field import LumenField
from lumen.physics.contact import ContactGeometry


def _scene(M=60, n_nodes=32, batch=64, R=2.0, L=120.0, dtype=torch.float64):
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    geom = ContactGeometry(cl, LumenField.cylinder(L, R, n=2), dtype=dtype)
    # random device nodes loosely inside the tube
    z = np.linspace(2.0, L - 2.0, n_nodes)
    base = np.stack([np.full(n_nodes, 1.0), np.zeros(n_nodes), z], axis=1)
    x = torch.tensor(base, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
    x = x + 0.2 * torch.randn(batch, n_nodes, 3, dtype=dtype)
    return geom, x, R


def _tube_surface(geom: ContactGeometry, R: float, T: int) -> torch.Tensor:
    """Triangulated tube surface vertices: M rings x T points. [M*T, 3]."""
    th = torch.linspace(0, 2 * np.pi, T + 1, dtype=geom.dtype)[:-1]
    verts = []
    for i in range(len(geom.P)):
        ring = (geom.P[i][None, :]
                + R * (torch.cos(th)[:, None] * geom.M1[i][None, :]
                       + torch.sin(th)[:, None] * torch.cross(
                           geom.T[i], geom.M1[i], dim=-1)[None, :]))
        verts.append(ring)
    return torch.cat(verts, dim=0)


def _time(fn, iters=50):
    fn()                                            # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def run_throughput(T=32, iters=50):
    geom, x, R = _scene()
    B, N, _ = x.shape
    surf = _tube_surface(geom, R, T)

    def tube_intrinsic():
        proj = geom.project(x)
        return geom._R_of_s(proj["s"]) - proj["r"]      # per-node gap

    def generic_mesh():
        pts = x.reshape(-1, 3)
        d = torch.cdist(pts, surf)                       # [B*N, M*T]
        return d.min(dim=1).values.reshape(B, N)        # nearest-surface distance

    t_tube = _time(tube_intrinsic, iters)
    t_mesh = _time(generic_mesh, iters)
    nps = B * N / t_tube
    return {"batch": B, "nodes": N, "surf_T": T, "surf_verts": surf.shape[0],
            "t_tube_ms": t_tube * 1e3, "t_mesh_ms": t_mesh * 1e3,
            "speedup": t_mesh / t_tube, "tube_nodes_per_s": nps}


def main():
    print("tube-intrinsic narrowphase cost is independent of circumferential wall")
    print("resolution T; the generic mesh-distance cost grows with it:\n")
    for T in (32, 64, 128, 256):
        r = run_throughput(T=T)
        print(f"T={T:3d} surf_verts={r['surf_verts']:5d}  "
              f"tube={r['t_tube_ms']:.3f}ms (flat)  mesh={r['t_mesh_ms']:.3f}ms  "
              f"speedup={r['speedup']:.1f}x")
    print("\nNote: at coarse T the optimised cdist baseline is comparable; the "
          "residual\nO(M) axial scan is removed by the arc-length broadphase cull "
          "(future).")


if __name__ == "__main__":
    main()
