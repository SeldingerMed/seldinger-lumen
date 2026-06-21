"""Continuum instrument as a discrete elastic rod (reduced, positional).

A guidewire/catheter/scope modelled as a polyline of nodes with stretch and
bending elastic energy. The fast-tier rod the doc starts from a Newton VBD cable;
here it is an energy whose gradient gives the internal force, so the same stepper
handles internal elasticity and contact uniformly.

ponytail: torsion DOF (whip/lag) is deferred to the P5 rod bake-off; M0 only needs
stretch+bend in a rigid tube. Torsional fidelity is tracked as a known risk
(doc §3.11) and added with the Cosserat upgrade, not faked here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RodParams:
    k_stretch: float = 5.0e2     # stretch stiffness
    k_bend: float = 5.0e0        # bending stiffness
    mass: float = 1.0e-2         # per-node mass (reserved for an inertial mode)
    damping: float = 2.0e2       # overdamped drag coefficient (blood-film proxy)


class Rod:
    """Batched rod state. Positions x: [B, N, 3]."""

    def __init__(self, x0: torch.Tensor, params: RodParams | None = None):
        if x0.ndim == 2:
            x0 = x0.unsqueeze(0)
        self.x = x0.clone()
        self.v = torch.zeros_like(self.x)
        self.params = params or RodParams()
        self.x_ref = None        # latched engagement positions (occlusion adhesion)
        self.engaged = None      # latched engaged-node mask
        # rest edge lengths from the initial shape
        self.l0 = torch.linalg.norm(self.x[:, 1:] - self.x[:, :-1], dim=-1)  # [B, N-1]

    @property
    def n_nodes(self) -> int:
        return self.x.shape[1]

    @classmethod
    def straight(cls, n: int, spacing: float, origin=(0.0, 0.0, 0.0),
                 axis=(0.0, 0.0, 1.0), batch: int = 1, params=None,
                 dtype=torch.float64) -> "Rod":
        axis = torch.tensor(axis, dtype=dtype)
        axis = axis / torch.linalg.norm(axis)
        s = torch.arange(n, dtype=dtype)[:, None] * spacing
        x0 = torch.tensor(origin, dtype=dtype)[None, :] + s * axis[None, :]
        return cls(x0.unsqueeze(0).repeat(batch, 1, 1), params)

    def internal_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Stretch + bending energy per batch element -> [B]."""
        p = self.params
        e = x[:, 1:] - x[:, :-1]
        length = torch.linalg.norm(e, dim=-1)
        e_stretch = 0.5 * p.k_stretch * ((length - self.l0) ** 2).sum(dim=-1)
        # discrete bending: second difference (linearised elastica)
        lap = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        e_bend = 0.5 * p.k_bend * (lap ** 2).sum(dim=(-1, -2))
        return e_stretch + e_bend
