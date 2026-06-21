"""Clot (thrombectomy) constitutive + two-way flow coupling (doc §3.4.4, §3.4.3, M3).

Ports the INSIST / Luraghi clot model into the fast, coupled setting (doc §3.4.4:
"port these constitutive/failure models into the fast, differentiable, GPU-batched,
coupled clot patch"). Parameters are grounded in Luraghi et al. (Interface Focus
2020, 10.1098/rsfs.2019.0123 / the INSIST in-silico-thrombectomy line), not invented:

  * clot bulk: quasi-hyperelastic 1-term Ogden, μ ≈ 0.5 kPa, α ≈ 0.3, ρ ≈ 1.06 g/cm³
  * clot–device / clot–wall friction μ_f ≈ 0.1
  * fragmentation during retrieval (a failure criterion on the retrieval load)

The clot is the local segment where the lumen R collapses (occlusion); the
contact barrier becomes an adhesive, frictional patch coupled to the device
(stent-retriever engagement) and to the flow (aspiration = a pressure sink that
mobilises the clot). Two-way flow: the clot occludes downstream flow; aspiration
pulls the clot toward the catheter. This is the reduced fast-tier port; the full
hybrid FEA–SPH is the accurate-tier oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClotParams:
    mu: float = 0.5e3            # Ogden shear modulus [Pa]  (Luraghi ~0.5 kPa)
    alpha: float = 0.3           # Ogden exponent [-]
    density: float = 1060.0      # [kg/m³]
    friction_mu: float = 0.1     # clot–device/wall friction (Luraghi ~0.1)
    area: float = 4.0e-6         # clot cross-section [m²] (~2.3 mm vessel)
    failure_stress: float = 8.0e3  # retrieval failure / fragmentation stress [Pa]
    viscous: float = 0.05        # speed-dependent retrieval load [N·s/m] (yank-rate effect)


def ogden_stress(stretch, p: ClotParams):
    """1-term incompressible Ogden uniaxial Cauchy stress σ(λ) [Pa].

    W = (2μ/α²)(λ₁^α+λ₂^α+λ₃^α−3); uniaxial incompressible λ₁=λ, λ₂=λ₃=λ^(−1/2):
        σ = (2μ/α)·(λ^α − λ^(−α/2)).
    Zero at λ=1; tension (λ>1) positive, compression (λ<1) negative. Monotone.
    """
    lam = np.asarray(stretch, dtype=float)
    return (2.0 * p.mu / p.alpha) * (lam ** p.alpha - lam ** (-p.alpha / 2.0))


class ClotModel:
    """Reduced 1-DOF clot along the centerline, coupled to the device + flow.

    State: arc-length position s_clot, engaged/fragmented flags. The device tip
    engages the clot within `engage_radius`; once engaged it drags the clot via an
    adhesive bond, resisted by the clot's Ogden deformation + Coulomb friction.
    If the retrieval load exceeds the failure load, the clot fragments (bond
    breaks). Aspiration adds a sink force toward the catheter.
    """

    def __init__(self, s_clot: float, params: ClotParams | None = None,
                 engage_radius: float = 3.0e-3):
        self.p = params or ClotParams()
        self.s_clot = float(s_clot)
        self.s0 = float(s_clot)
        self.engage_radius = engage_radius
        self.engaged = False
        self.fragmented = False
        self.last_load = 0.0
        self._prev_tip = None

    @property
    def failure_load(self) -> float:
        return self.p.failure_stress * self.p.area

    def static_resistance(self) -> float:
        """Quasi-static retrieval resistance: Ogden bulk + Coulomb friction [N]."""
        ogden = abs(ogden_stress(1.3, self.p)) * self.p.area    # clot dragged through lumen
        return ogden * (1.0 + self.p.friction_mu)

    def step(self, s_tip: float, dt: float, aspiration: float = 0.0) -> dict:
        """Advance the clot one step given the device-tip arc-length s_tip.

        When the engaged device retracts (pulls proximally), the retrieval load is
        the clot's static resistance + a yank-rate (viscous) term, minus aspiration
        (a pressure sink that mobilises the clot). If that load exceeds the clot's
        failure load, the clot fragments; otherwise the clot follows the device
        (retrieval). Aspiration both lowers the load (assists) and clears flow.
        """
        v_dev = 0.0 if self._prev_tip is None else (s_tip - self._prev_tip) / dt
        self._prev_tip = s_tip
        if not self.fragmented and abs(s_tip - self.s_clot) < self.engage_radius:
            self.engaged = True
        load = 0.0
        if self.engaged and not self.fragmented:
            retracting = v_dev < -1.0e-9                        # device moving proximally -> pulling out
            if retracting:
                load = self.static_resistance() + self.p.viscous * abs(v_dev) - aspiration
                load = max(load, 0.0)
                self.last_load = load
                if load > self.failure_load:                    # fragmentation criterion
                    self.fragmented = True
                    self.engaged = False
                else:
                    self.s_clot = s_tip                          # clot follows (retrieved)
            # while advancing/holding, the engaged clot stays put (no retrieval)
        return {"engaged": self.engaged, "fragmented": self.fragmented,
                "load": load, "retrieved": self.s0 - self.s_clot,
                "s_clot": self.s_clot}


def downstream_flow(base_flow: float, clot_present: bool, occlusion: float = 0.95,
                    aspiration_fraction: float = 0.0) -> float:
    """Two-way flow: a clot occludes downstream flow; aspiration restores some.

    Returns the downstream flow rate given an upstream `base_flow`. With a clot,
    flow drops by `occlusion`; aspiration (fraction of occlusion cleared) recovers
    part of it as the clot is mobilised/removed.
    """
    if not clot_present:
        return base_flow
    return base_flow * (1.0 - occlusion * (1.0 - aspiration_fraction))
