"""Blood flow on the Newton platform (doc §3.4.3).

Reduced-order flow, coupled to the Newton guidewire sim three ways:

  * pulsatility  — a 2-element Windkessel drives a pulsatile pressure/flow; the
    lumen radius R(s,θ,t) breathes with the cycle (the wall's resting r0_field is
    modulated each step — "pulsatility = temporal modulation of R", §3.5.6).
  * device drag  — flow exerts a one-way axial drag on the device, ∝ downstream Q.
  * two-way      — a clot/device occlusion reduces downstream Q; an aspiration
    pressure sink raises the local mobilising flow and assists clot retrieval.

Two models, same coupling API:
  * ``NewtonFlow`` — the lumped 2-element Windkessel: one global scalar Q with an
    occlusion fraction. The analytic fallback the doc keeps for validation.
  * ``FlowField`` — a 1-D resistive network along the centerline: P(s)/v(s) fields
    where the clot's resistance, the velocity jet through a narrowing, and
    aspiration-as-a-pressure-sink all emerge from the SHARED lumen radius (§3.4.3).

The learned GNN hemodynamic surrogate is the private/later upgrade behind the same
``downstream_Q`` / drag API.
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


@dataclass
class FlowFieldParams:
    P_mean: float = 100.0        # mean inflow (driving) pressure
    P_pulse: float = 40.0        # systolic pressure pulse amplitude
    heart_rate: float = 1.0      # Hz
    R_periph: float = 2.0        # lumped distal/peripheral resistance (Windkessel)
    visc: float = 1.0            # lumped blood viscosity: r_seg = visc·ds/R⁴ (Poiseuille)
    R_floor: float = 0.05        # min lumen radius (a fully-collapsed cell isn't 0 -> inf)
    drag_coeff: float = 4.0      # device axial drag per unit LOCAL flow velocity
    pulse_amp: float = 0.05      # lumen-radius pulsatility amplitude (fraction of R0)
    asp_gain: float = 150.0      # suction pressure at the tip per unit aspiration
                                 # (> P_mean so full suction can reverse flow across a clot)


class FlowField:
    """1-D resistive-network blood flow along the centerline (doc §3.4.3).

    The vessel is a chain of segments in series; each has a Poiseuille hydraulic
    resistance r ∝ visc·ds/R(s)⁴ read from the SHARED lumen radius (so a clot/stenosis
    raises local resistance directly). A pulsatile inflow pressure drives the chain
    against a lumped peripheral resistance; continuity gives one through-flow Q, and
    the cumulative pressure drop gives the field P(s). Velocity v(s)=Q/A(s) varies
    along s — fast through a narrowing — so device drag is LOCAL, not uniform.

    Aspiration is a pressure SINK at the catheter tip: it splits the chain at the
    tip node and pulls its pressure down, which can reverse the gradient across a
    distal clot (retrograde flow) — the physical mobilising force for retrieval.

    Duck-types NewtonFlow's API (pulse_factor / advance / Q / downstream_Q /
    drag_per_unit_tangent / .aspiration) so it drops into NewtonGuidewireSim, and
    adds the field accessors (pressure_field / velocity_field / drag_at /
    clot_mobilizing_force) the sim uses when present.
    """

    def __init__(self, params: FlowFieldParams | None = None):
        self.p = params or FlowFieldParams()
        self.t = 0.0
        self.aspiration = 0.0            # [0,1] suction command at the tip
        self.R_s = None                  # per-node open lumen radius R(s)
        self.s_grid = None
        self.tip_s = None                # catheter-tip arc-length (aspiration point)
        self._P = None                   # solved pressure field P(s)
        self._v = None                   # solved velocity field v(s)
        self._Q = 0.0                    # solved through-flow (proximal)

    # --- geometry / actuation set each step by the sim ----------------------
    def set_lumen(self, radius_s, s_max: float) -> None:
        import numpy as np
        self.R_s = np.maximum(np.asarray(radius_s, dtype=float), self.p.R_floor)
        self.s_grid = np.linspace(0.0, s_max, len(self.R_s))

    def set_tip(self, s_tip: float) -> None:
        self.tip_s = float(s_tip)

    def P_in(self, t: float | None = None) -> float:
        tt = self.t if t is None else t
        return self.p.P_mean + self.p.P_pulse * max(0.0, math.sin(2 * math.pi * self.p.heart_rate * tt))

    def pulse_factor(self, t: float | None = None) -> float:
        tt = self.t if t is None else t
        return 1.0 + self.p.pulse_amp * math.sin(2 * math.pi * self.p.heart_rate * tt)

    def advance(self, dt: float) -> None:
        self.t += dt

    # --- the 1-D solve -------------------------------------------------------
    def solve(self) -> None:
        """Solve the series network for Q, P(s), v(s) at the current geometry/phase."""
        import numpy as np
        if self.R_s is None:
            return
        R = self.R_s
        A = math.pi * R ** 2
        ds = self.s_grid[1] - self.s_grid[0]
        R_mid = 0.5 * (R[:-1] + R[1:])                      # per-segment radius
        r_seg = self.p.visc * ds / R_mid ** 4               # Poiseuille resistance per segment
        Pin = self.P_in()
        n = len(R)
        # tip node (aspiration sink); default to the distal end if unset
        it = n - 1 if self.tip_s is None else int(np.clip(
            round(self.tip_s / self.s_grid[-1] * (n - 1)), 0, n - 1))
        cum = np.concatenate([[0.0], np.cumsum(r_seg)])     # cum resistance to each node
        R_up = cum[it]                                      # inflow -> tip
        R_down = (cum[-1] - cum[it]) + self.p.R_periph      # tip -> peripheral bed
        asp = min(max(self.aspiration, 0.0), 1.0)
        # natural tip pressure (no suction), then impose suction P_suction at the tip
        Q_nat = Pin / (R_up + R_down)
        P_tip_nat = Pin - Q_nat * R_up
        P_tip = P_tip_nat - asp * self.p.asp_gain
        Q_up = (Pin - P_tip) / max(R_up, 1e-9)
        Q_down = P_tip / max(R_down, 1e-9)                 # can go NEGATIVE under suction (retrograde)
        P = np.empty(n)
        P[:it + 1] = Pin - Q_up * cum[:it + 1]
        P[it:] = P_tip - Q_down * (cum[it:] - cum[it])
        # per-node through-flow: Q_up proximal of tip, Q_down distal
        Qn = np.where(np.arange(n) < it, Q_up, Q_down)
        self._P, self._v, self._Q = P, Qn / A, Q_up
        self._Q_down = Q_down

    # --- accessors -----------------------------------------------------------
    def pressure_field(self):
        return self._P

    def velocity_field(self):
        return self._v

    def Q(self, t: float | None = None) -> float:
        return float(self._Q)

    def downstream_Q(self, t: float | None = None) -> float:
        """Through-flow reaching the distal bed (drops toward 0 as the clot occludes)."""
        return float(max(getattr(self, "_Q_down", self._Q), 0.0))

    def drag_at(self, s_query):
        """Local axial drag magnitude at arc-length(s) s_query, ∝ local velocity v(s)."""
        import numpy as np
        if self._v is None:
            return np.zeros_like(np.asarray(s_query, dtype=float))
        v = np.interp(np.asarray(s_query, dtype=float), self.s_grid, self._v)
        return self.p.drag_coeff * v

    def drag_per_unit_tangent(self, t: float | None = None) -> float:
        """API-compat scalar drag (mean local velocity) for non-field callers."""
        import numpy as np
        return float(self.p.drag_coeff * (np.mean(self._v) if self._v is not None else 0.0))

    def clot_mobilizing_force(self, s0: float, s1: float) -> float:
        """Signed proximal force on the clot from the pressure field across [s0,s1].

        = (P_distal − P_proximal)·A_clot. POSITIVE assists retrieval (a retrograde
        gradient, e.g. under aspiration — flow pulls the clot toward the catheter);
        NEGATIVE resists it (antegrade flow pushes the clot downstream). The sign is
        what makes aspiration physical: without it the pressure behind an occluding
        clot pushes it away from the catheter.

        ponytail: force is ΔP·A in sim-consistent units; absolute SI calibration of
        the flow/clot force balance against thrombectomy data is the private layer (§8).
        """
        import numpy as np
        if self._P is None:
            return 0.0
        m = (self.s_grid >= s0) & (self.s_grid <= s1)
        if not m.any():
            return 0.0
        idx = np.where(m)[0]
        P_prox, P_dist = self._P[idx[0]], self._P[idx[-1]]
        A_clot = math.pi * float(np.mean(self.R_s[idx])) ** 2
        return (P_dist - P_prox) * A_clot
