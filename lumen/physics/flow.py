"""Blood flow as an analytic reduced-order surrogate (doc §3.4.3).

Online Navier-Stokes is incompatible with RL throughput, so flow is supplied by a
reduced model: here a 2-element Windkessel giving a pulsatile flow rate Q(t) and
pressure P(t), coupled *one-way* to the device as a downstream drag force (flow
modulates the device; the device does not yet redistribute the flow). Two-way
coupling -- occlusion redistributing flow, aspiration -- is where it becomes the
mechanism (flow diversion, aspiration thrombectomy) and attaches at the same seam.

This is the generic analytic fallback the doc keeps for validation; the learned
GNN hemodynamic surrogate is the private/later upgrade behind the same drag API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class WindkesselFlow:
    Q_mean: float = 4.0        # mean flow (arbitrary units)
    Q_pulse: float = 2.0       # pulsatile amplitude
    heart_rate: float = 1.0    # Hz
    R_periph: float = 1.0      # peripheral resistance
    C: float = 1.5             # arterial compliance
    drag_coeff: float = 30.0   # device drag per unit flow

    def Q(self, t: float) -> float:
        """Pulsatile inflow: mean + a (rectified) systolic pulse."""
        phase = 2 * math.pi * self.heart_rate * t
        return self.Q_mean + self.Q_pulse * max(0.0, math.sin(phase))

    def pressure_decay(self, p0: float, dt: float) -> float:
        """Diastolic Windkessel decay over dt: P <- P * exp(-dt / (R*C))."""
        return p0 * math.exp(-dt / (self.R_periph * self.C))

    def drag_force(self, tangents: torch.Tensor, t: float) -> torch.Tensor:
        """One-way drag on device nodes along the local downstream tangent.

        tangents [B, N, 3] (unit, pointing distally) -> force [B, N, 3].
        """
        return self.drag_coeff * self.Q(t) * tangents
