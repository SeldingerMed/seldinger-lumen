"""Deformable vessel wall as a reduced shell sharing R (doc §3.4.2, §3.5.6).

The single most important architectural choice in the doc: wall mechanics and
contact geometry are the *same object*, the lumen field R(s,theta). Here the wall
carries a radial-displacement field w(s,theta) on a grid; the deformed lumen is
R0 + w, and the contact barrier reads that deformed R. So:

  * wall deformation under device load and contact detection are consistent by
    construction -- there is no separate collision mesh to keep in sync;
  * pulsatility would be a temporal modulation of R0 at ~no extra contact cost.

The shell is reduced-order (a membrane-on-elastic-foundation in (s,theta), not a
volumetric FEM) precisely to preserve batching (doc §3.5.10). Anisotropy stands
in for HGO's two collagen fiber families as distinct axial vs hoop stiffness;
real HGO calibration against vessel data stays private.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WallShellParams:
    k_axial: float = 2.0e2     # stiffness along arc-length (fiber family ~axial)
    k_hoop: float = 6.0e2      # circumferential stiffness (fiber family ~hoop) -> anisotropy
    k_found: float = 5.0e2     # radial elastic-foundation stiffness (local restoring)
    drag: float = 2.0e2        # overdamped drag for the wall DOFs


class WallShell:
    """Radial displacement field w(s, theta) on a grid; deformed R = R0 + w."""

    def __init__(self, s_grid, R0_of_s, n_theta: int = 24,
                 params: WallShellParams | None = None, batch: int = 1,
                 dtype=torch.float64, device="cpu"):
        self.params = params or WallShellParams()
        self.dtype, self.device = dtype, device
        self.s = torch.as_tensor(np.asarray(s_grid), dtype=dtype, device=device)
        th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        self.theta = torch.as_tensor(th, dtype=dtype, device=device)
        # base (undeformed) lumen radius on the s-grid, broadcast over theta
        R0 = torch.as_tensor(np.asarray([R0_of_s(float(si)) for si in self.s]),
                             dtype=dtype, device=device)
        self.R0 = R0[:, None].repeat(1, n_theta)            # [n_s, n_theta]
        self.w = torch.zeros(batch, len(self.s), n_theta, dtype=dtype, device=device)

    # --- energy --------------------------------------------------------------
    def energy(self, w: torch.Tensor) -> torch.Tensor:
        """Anisotropic membrane-on-foundation energy per batch -> [B]."""
        p = self.params
        d_s = w[:, 1:, :] - w[:, :-1, :]                    # axial gradient
        d_th = torch.roll(w, -1, dims=2) - w                # hoop gradient (periodic)
        e = (0.5 * p.k_axial * (d_s ** 2).sum(dim=(-1, -2))
             + 0.5 * p.k_hoop * (d_th ** 2).sum(dim=(-1, -2))
             + 0.5 * p.k_found * (w ** 2).sum(dim=(-1, -2)))
        return e

    # --- sampling ------------------------------------------------------------
    def sample(self, s: torch.Tensor, theta: torch.Tensor,
               w: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of w at node coords (s,theta), each [B, N] -> [B, N].

        Differentiable in w (so contact load flows back onto the wall DOFs).
        Periodic in theta, clamped in s.
        """
        B, N = s.shape
        sg, tg = self.s, self.theta
        # s axis: clamp-interp
        si = torch.searchsorted(sg, s.clamp(sg[0], sg[-1])).clamp(1, len(sg) - 1)
        s0 = sg[si - 1]
        fs = ((s - s0) / (sg[si] - s0).clamp_min(1e-12)).clamp(0, 1)
        i0, i1 = si - 1, si
        # theta axis: periodic
        nt = len(tg)
        step = (tg[-1] - tg[0]) / (nt - 1)                  # uniform spacing
        tj = ((theta - tg[0]) % (2 * np.pi)) / step
        j0 = torch.floor(tj).long() % nt
        j1 = (j0 + 1) % nt
        ft = (tj - torch.floor(tj))
        bidx = torch.arange(B, device=s.device)[:, None].expand(B, N)
        g = lambda i, j: w[bidx, i, j]
        c00, c01 = g(i0, j0), g(i0, j1)
        c10, c11 = g(i1, j0), g(i1, j1)
        c0 = c00 * (1 - ft) + c01 * ft
        c1 = c10 * (1 - ft) + c11 * ft
        return c0 * (1 - fs) + c1 * fs
