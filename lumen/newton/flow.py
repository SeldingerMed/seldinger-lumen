"""Blood flow on the Newton platform (doc §3.4.3).

Reduced-order flow, coupled to the Newton guidewire sim three ways:

  * pulsatility  — a 2-element Windkessel drives a pulsatile pressure/flow; the
    lumen radius R(s,θ,t) breathes with the cycle (the wall's resting r0_field is
    modulated each step — "pulsatility = temporal modulation of R", §3.5.6).
  * device drag  — flow exerts a one-way axial drag on the device, ∝ downstream Q.
  * two-way      — a clot/device occlusion reduces downstream Q; an aspiration
    pressure sink raises the local mobilising flow and assists clot retrieval.

The learned GNN hemodynamic surrogate is the private/later upgrade behind the same
``downstream_Q`` / drag API; the Windkessel here is the analytic fallback the doc
keeps for validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FlowParams:
    Q_mean: float = 4.0          # mean inflow (sim units)
    Q_pulse: float = 2.0         # systolic pulse amplitude
    heart_rate: float = 1.0      # Hz
    R_periph: float = 1.0        # peripheral resistance (Windkessel)
    C: float = 1.5               # arterial compliance (Windkessel)
    drag_coeff: float = 20.0     # device axial drag per unit flow
    pulse_amp: float = 0.05      # lumen-radius pulsatility amplitude (fraction of R0)


class NewtonFlow:
    """Windkessel flow driver coupled to a NewtonGuidewireSim."""

    def __init__(self, params: FlowParams | None = None):
        self.p = params or FlowParams()
        self.t = 0.0
        self.occlusion = 0.0     # [0,1] downstream blockage (set by the clot)
        self.aspiration = 0.0    # sink strength toward the catheter (set externally)

    # --- Windkessel ----------------------------------------------------------
    def Q(self, t: float | None = None) -> float:
        """Pulsatile inflow: mean + rectified systolic pulse."""
        tt = self.t if t is None else t
        return self.p.Q_mean + self.p.Q_pulse * max(0.0, math.sin(2 * math.pi * self.p.heart_rate * tt))

    def downstream_Q(self, t: float | None = None) -> float:
        """Two-way: occlusion reduces flow; aspiration recovers part of it."""
        occ = min(max(self.occlusion, 0.0), 1.0)
        asp = min(max(self.aspiration, 0.0), 1.0)
        return self.Q(t) * (1.0 - occ * (1.0 - asp))

    def pressure_decay(self, p0: float, dt: float) -> float:
        """Diastolic Windkessel decay over dt: P <- P·exp(-dt/(R·C))."""
        return p0 * math.exp(-dt / (self.p.R_periph * self.p.C))

    # --- coupling helpers ----------------------------------------------------
    def pulse_factor(self, t: float | None = None) -> float:
        """Lumen-radius modulation 1 + amp·(normalised pressure pulse) for R(s,θ,t)."""
        tt = self.t if t is None else t
        return 1.0 + self.p.pulse_amp * math.sin(2 * math.pi * self.p.heart_rate * tt)

    def drag_per_unit_tangent(self, t: float | None = None) -> float:
        """Axial drag force magnitude per (unit) device tangent, ∝ downstream Q."""
        return self.p.drag_coeff * self.downstream_Q(t)

    def advance(self, dt: float) -> None:
        self.t += dt
