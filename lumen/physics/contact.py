"""Tube-intrinsic contact: the first novel core (doc §3.5).

Instead of querying device-versus-mesh in world space, project each rod node into
the tube-intrinsic frame (s, theta, r) of the centerline and evaluate a 1-D
per-node condition r <= R(s, theta). The narrowphase collapses to a near-analytic
per-node query, the barrier is a smooth analytic function of smooth coordinates,
and wall + contact share the same R field.

This module precomputes the (fixed) centerline frame as tensors and projects the
(variable, autograd-tracked) rod nodes against it, so contact forces come from
the gradient of the barrier energy.

ponytail: P1/P2 assume a rigid OR quasi-static centerline shared across the batch
(one anatomy, B rod states). Per-env anatomies and bifurcation blending of R near
branch points are deferred to P5 (doc §3.5.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lumen.core.frame import CenterlineFrame
from lumen.core.lumen_field import LumenField


@dataclass
class ContactParams:
    kappa: float = 1.0e3      # barrier stiffness
    d_hat: float = 0.2        # barrier activation thickness
    mu: float = 0.3           # Coulomb friction coefficient


class ContactGeometry:
    """Fixed centerline frame + lumen field, as a differentiable contact oracle."""

    def __init__(self, centerline: np.ndarray, lumen: LumenField,
                 dtype=torch.float64, device="cpu"):
        f = CenterlineFrame(centerline)
        t = lambda a: torch.tensor(np.asarray(a), dtype=dtype, device=device)
        self.P = t(f.points)            # [M, 3]
        self.T = t(f.tangents)          # [M, 3]
        self.M1 = t(f.m1)               # [M, 3]
        self.cum_s = t(f.cum_s)         # [M]
        self.seg = self.P[1:] - self.P[:-1]                       # [S, 3]
        self.seg_len = torch.linalg.norm(self.seg, dim=-1)       # [S]
        self.L2 = (self.seg ** 2).sum(-1).clamp_min(1e-12)       # [S]
        # lumen field R(s) sampled on its s-grid (axisymmetric in P1)
        self.s_grid = t(lumen.s)
        self.R_grid = t(lumen.R[:, 0])  # [len(s_grid)]
        self.dtype, self.device = dtype, device

    def _R_of_s(self, s: torch.Tensor) -> torch.Tensor:
        """1-D interpolation of R over arc-length (differentiable in s)."""
        sg = self.s_grid
        idx = torch.searchsorted(sg, s.clamp(sg[0], sg[-1]))
        idx = idx.clamp(1, len(sg) - 1)
        s0, s1 = sg[idx - 1], sg[idx]
        r0, r1 = self.R_grid[idx - 1], self.R_grid[idx]
        w = ((s - s0) / (s1 - s0).clamp_min(1e-12)).clamp(0, 1)
        return r0 + w * (r1 - r0)

    def project(self, x: torch.Tensor):
        """Project nodes x [B, N, 3] -> dict of (s, theta, r, e_r), each [B, N]."""
        B, N, _ = x.shape
        pts = x.reshape(-1, 3)                       # [P, 3]
        a = self.P[:-1]                              # [S, 3]
        ap = pts[:, None, :] - a[None, :, :]         # [P, S, 3]
        u = ((ap * self.seg[None]).sum(-1) / self.L2[None]).clamp(0, 1)   # [P, S]
        foot = a[None] + u[..., None] * self.seg[None]                    # [P, S, 3]
        d2 = ((pts[:, None, :] - foot) ** 2).sum(-1)                      # [P, S]
        j = torch.argmin(d2, dim=1)                  # [P]
        pj = torch.arange(pts.shape[0], device=x.device)
        uj = u[pj, j]                                # [P]
        footj = foot[pj, j]                          # [P, 3]
        s = self.cum_s[j] + uj * self.seg_len[j]     # [P]
        # tangent at foot (lerp between vertex j and j+1)
        t = self.T[j] + uj[:, None] * (self.T[j + 1] - self.T[j])
        t = t / torch.linalg.norm(t, dim=-1, keepdim=True).clamp_min(1e-12)
        radial = (pts - footj) - ((pts - footj) * t).sum(-1, keepdim=True) * t
        r = torch.linalg.norm(radial, dim=-1)        # [P]
        e_r = radial / r.clamp_min(1e-12)[:, None]
        # reference axes for theta
        m1 = self.M1[j] - (self.M1[j] * t).sum(-1, keepdim=True) * t
        m1 = m1 / torch.linalg.norm(m1, dim=-1, keepdim=True).clamp_min(1e-12)
        m2 = torch.cross(t, m1, dim=-1)
        theta = torch.atan2((radial * m2).sum(-1), (radial * m1).sum(-1))
        re = lambda z: z.reshape(B, N)
        return {"s": re(s), "theta": re(theta), "r": re(r),
                "e_r": e_r.reshape(B, N, 3)}

    def barrier_energy(self, x: torch.Tensor, cp: ContactParams,
                       R_override: torch.Tensor | None = None) -> torch.Tensor:
        """Compliant quadratic barrier energy per batch -> [B].

        Active within d_hat of the wall; penalises penetration. R_override lets a
        deformable wall (P2) supply a per-node radius instead of the rigid R(s).
        """
        proj = self.project(x)
        R = R_override if R_override is not None else self._R_of_s(proj["s"])
        g = R - proj["r"]                            # gap, [B, N]
        pen = (cp.d_hat - g).clamp_min(0.0)          # penetration into the barrier
        return 0.5 * cp.kappa * (pen ** 2).sum(dim=-1)

    def friction_force(self, x, v, cp: ContactParams):
        """Non-conservative Coulomb friction (regularised), as an explicit force.

        Returns force [B, N, 3] opposing the wall-tangential velocity, scaled by
        the normal load. Friction is velocity-dependent, so it is applied as a
        force in the stepper rather than baked into the energy.
        """
        proj = self.project(x)
        R = self._R_of_s(proj["s"])
        g = R - proj["r"]
        pen = (cp.d_hat - g).clamp_min(0.0)
        fn = cp.kappa * pen                          # normal load magnitude [B, N]
        e_r = proj["e_r"]
        v_t = v - (v * e_r).sum(-1, keepdim=True) * e_r   # tangential velocity
        v_t_mag = torch.linalg.norm(v_t, dim=-1, keepdim=True)
        dirn = v_t / (v_t_mag + 1e-6)
        return -(cp.mu * fn)[..., None] * dirn
