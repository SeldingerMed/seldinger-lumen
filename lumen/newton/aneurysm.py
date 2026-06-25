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
→ occlusion (the cure). This is the DEVICE→flow direction of §3.4.3's flow-diversion
coupling: the diverter's metal coverage redistributes flow away from the sac.

ponytail: the coupling realized here is one-directional — the diverter throttles
neck inflow, but the sac→parent back-reaction (the neck draw shaving the parent's
through-flow) is O(A_neck/A_lumen) and dropped — the parent network has no sac
node and aneurysm flow ≪ parent flow, so the 1-D field is unperturbed by the sac.
Absolute SI calibration of the inflow/stasis balance against PC-MRI / CFD is the
private layer (§8); here the units are sim-consistent and the CLAIM is the monotone
direction (diverter ⇒ less inflow, more stasis), not an absolute occlusion time.
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
        self.P_sac = None          # lazy-init to the FIRST lumen pressure sample (skips
        self.mark_window()         # the gross charging transient; the cardiac PULSE then
        self.last_Q = 0.0          # drives the cyclic exchange the metrics measure)
        self.last_diversion = 0.0

    def mark_window(self) -> None:
        """Open a fresh measurement window — call at flow-diverter DEPLOYMENT so the
        post-deployment inflow/turnover aren't blended with the pre-deployment phase.
        Zeros the accumulators but KEEPS the sac-pressure equilibrium (no re-incurred
        charging transient, unlike :meth:`reset`)."""
        self.peak_inflow = 0.0
        self._exch = 0.0
        self._t = 0.0

    def update(self, P_lumen: float, dt: float, diversion: float = 0.0) -> float:
        """Advance the sac by ``dt`` under lumen pressure ``P_lumen``; ``diversion``
        in [0,1) is the flow-diverter's neck coverage. Returns the neck flow Q."""
        P = float(P_lumen)
        if self.P_sac is None:                 # equilibrate to the lumen on first contact
            self.P_sac = P
        div = min(max(float(diversion), 0.0), 0.999)
        R_neck = self.R_neck_base / (1.0 - div) ** 2     # porous screen raises R
        tau = R_neck * self.C_sac
        # explicit Euler on dP/dt=(P_lumen−P)/(R·C) is stable only for dt<2τ; sub-divide
        # so ANY caller dt/substeps is safe (a very compliant sac shrinks τ — L1).
        n_sub = max(1, int(dt / (0.5 * tau)) + 1)
        h = dt / n_sub
        Q = 0.0
        for _ in range(n_sub):                 # zero-order hold on P_lumen across micro-steps
            Q = (P - self.P_sac) / max(R_neck, 1e-12)
            self.P_sac += Q / self.C_sac * h
            self.peak_inflow = max(self.peak_inflow, abs(Q))
            self._exch += abs(Q) * h
        self._t += dt
        self.last_Q, self.last_diversion = Q, div
        return Q

    # --- metrics over the current window (since reset/mark_window) ------------
    def inflow_peak(self) -> float:
        """Peak |neck flow| in the current window — the systolic inflow jet. Lower
        with a diverter. Call :meth:`mark_window` at deployment to read the POST-
        deployment peak (else this retains the pre-deployment systolic max)."""
        return self.peak_inflow

    def mean_exchange(self) -> float:
        """Time-averaged |neck flow| over the current window — the sac exchange rate."""
        return self._exch / max(self._t, 1e-9)

    def turnover_time(self) -> float:
        """Washout time = sac volume / mean exchange. Longer = more stasis (cure)."""
        m = self.mean_exchange()
        return self.a.sac_volume / m if m > 1e-12 else float("inf")
