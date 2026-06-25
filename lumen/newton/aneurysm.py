"""Saccular aneurysm + flow-diversion coupling (doc §3.4.3).

An aneurysm is a compliant side-pouch off the parent lumen, fed through a NECK.
Unlike a clot (a local collapse of the through-lumen R, handled by ``ClotField``),
the sac is a parallel compartment that the through-flow never traverses — so it is
a 0-D lumped element coupled to the 1-D ``FlowField`` at one arc-length.

Model (reduced-order, faithful to §3.4.3's "reduced surrogate" flow philosophy):
the sac is a compliant reservoir (compliance ``C_sac``) connected to the parent
lumen through a neck hydraulic resistance ``R_neck``. The pulsatile lumen pressure
P(s_neck, t) drives an oscillatory EXCHANGE flow across the neck — the clinical
"inflow jet". This is a first-order RC low-pass:

    Q_neck = (P_lumen − P_sac) / R_neck          # +ve = into the sac
    dP_sac/dt = Q_neck / C_sac

For a sinusoidal pulse of amplitude ΔP at frequency ω the steady cyclic inflow
amplitude is  Q ≈ ΔP·ωC / √(1 + (ωR_neck·C)²)  — monotonically DECREASING in
R_neck. A flow diverter is a porous tube laid across the neck: it raises R_neck
(a screen's resistance ~ 1/(1−coverage)²), cutting the inflow jet and lengthening
the sac turnover time. Sustained low turnover → intra-saccular stasis → thrombosis
→ occlusion (the cure). That device→neck→inflow→stasis loop is the §3.4.3 two-way
"device occlusion redistributes flow" mechanism for flow diversion.

ponytail: the sac→parent back-reaction (the neck draw shaving the parent's
through-flow) is O(A_neck/A_lumen) and dropped — the parent network has no sac
node and aneurysm flow ≪ parent flow. Absolute SI calibration of the inflow/
stasis balance against PC-MRI / CFD is the private layer (§8); here the units are
sim-consistent and the CLAIM is the monotone direction (diverter ⇒ less inflow,
more stasis), not an absolute occlusion-time prediction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Aneurysm:
    """A saccular aneurysm on the parent vessel (geometry + lumped mechanics).

    Geometry (``s_neck``/``neck_width``) is what a flow diverter's deployed span
    must cover; ``sac_volume`` sets the turnover time. ``wall_stiffness`` and the
    neck conductance are the lumped calibratable knobs (private-layer targets)."""

    s_neck: float                  # arc-length of the neck centre on the parent [mm]
    neck_width: float = 4.0        # neck opening along s [mm] (the device-coverage span)
    sac_volume: float = 100.0      # sac volume [mm^3] — sets turnover time
    wall_stiffness: float = 2.0e3  # lumped sac wall stiffness (C_sac = volume/stiffness)
    neck_resistance_coeff: float = 4.0  # neck R = coeff·visc/neck_width. The default puts
                                        # the scene near ω·R_neck·C ~ O(1), where flow
                                        # diversion actually bites (a calibratable knob;
                                        # the wide-neck/low-R = hard-to-treat ordering holds)


class AneurysmSac:
    """0-D compliant sac compartment coupled to a 1-D ``FlowField`` at the neck.

    Driven each substep by the live lumen pressure at ``s_neck``; integrates the RC
    exchange and accumulates the inflow / turnover metrics. A flow diverter enters
    only through ``diversion`` in :meth:`update` (it raises the neck resistance)."""

    def __init__(self, aneurysm: Aneurysm, visc: float = 1.0):
        self.a = aneurysm
        # neck hydraulic resistance: a narrower neck resists more (wide-neck
        # aneurysms are the hard-to-treat ones). Poiseuille-like in the neck width.
        self.R_neck_base = aneurysm.neck_resistance_coeff * float(visc) / max(aneurysm.neck_width, 1e-3)
        self.C_sac = aneurysm.sac_volume / max(aneurysm.wall_stiffness, 1e-9)
        self.reset()

    def reset(self) -> None:
        self.P_sac = None          # lazy-init to the first lumen pressure (skip the
        self.peak_inflow = 0.0     # charging transient: the sac sits at mean pressure,
        self._exch = 0.0           # the cardiac PULSE drives the cyclic exchange)
        self._t = 0.0
        self.last_Q = 0.0
        self.last_diversion = 0.0

    def update(self, P_lumen: float, dt: float, diversion: float = 0.0) -> float:
        """Advance the sac by ``dt`` under lumen pressure ``P_lumen``; ``diversion``
        in [0,1) is the flow-diverter's neck coverage. Returns the neck flow Q."""
        if self.P_sac is None:                 # equilibrate to the lumen on first contact
            self.P_sac = float(P_lumen)
        div = min(max(float(diversion), 0.0), 0.999)
        R_neck = self.R_neck_base / (1.0 - div) ** 2     # porous screen raises R
        Q = (float(P_lumen) - self.P_sac) / max(R_neck, 1e-12)
        self.P_sac += Q / self.C_sac * dt                # RC integration
        self.peak_inflow = max(self.peak_inflow, abs(Q))
        self._exch += abs(Q) * dt
        self._t += dt
        self.last_Q, self.last_diversion = Q, div
        return Q

    # --- metrics (the clinically meaningful outputs) -------------------------
    def inflow_peak(self) -> float:
        """Peak |neck flow| seen — the systolic inflow jet. Diverter lowers it."""
        return self.peak_inflow

    def mean_exchange(self) -> float:
        """Time-averaged |neck flow| — the sac's exchange rate."""
        return self._exch / max(self._t, 1e-9)

    def turnover_time(self) -> float:
        """Washout time = sac volume / mean exchange. Longer = more stasis (cure)."""
        m = self.mean_exchange()
        return self.a.sac_volume / m if m > 1e-12 else float("inf")
