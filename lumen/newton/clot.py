"""Clot as a finite-extent, deformable, damageable occlusion field (doc §3.4.4).

A real port (reduced to a fast 1-D arc-length field) of the INSIST/Luraghi clot,
replacing the earlier 1-DOF behavioural stub. Parameters are grounded in Luraghi
et al. (Interface Focus 2020): Ogden bulk (μ≈0.5 kPa, α≈0.3), clot-device friction
≈0.1, a failure criterion calibrated on clot analogs.

The clot is a segment [s0, s1] along the centerline where the lumen radius
collapses by an occlusion profile o(s): the contact barrier reads the SHARED field
R_eff(s,θ) = R0(s,θ) − o(s) + w(s,θ), so the device physically meets the clot as a
narrowing it must push through (real R-collapse + contact coupling, not a point).

Constitutive behaviour (per arc-length cell, each substep):
  * the device contact load on the clot (read from the solver's wall_load) compresses
    it; the compression follows the Ogden CURVE (σ(λ), λ = o/o0), not a fixed stretch;
  * friction with the wall/device uses μ · (actual contact normal force), not the
    clot's bulk stress;
  * progressive damage D(s) accumulates where the stress exceeds the failure stress
    (not a boolean); D→1 clears the occlusion locally (fragmentation);
  * the residual occlusion sets the downstream-flow blockage (two-way coupling).

Retrieval by a stent-retriever (translation of the occlusion) is in
lumen.newton.devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClotParams:
    mu: float = 0.5e3            # Ogden shear modulus (Luraghi ~0.5 kPa)
    alpha: float = 0.3          # Ogden exponent
    area: float = 4.0e-6        # clot contact cross-section (per cell)
    friction_mu: float = 0.1    # clot-device/wall friction (Luraghi ~0.1)
    failure_stress: float = 8.0e3   # fragmentation stress (clot-analog calibrated)
    damage_rate: float = 4.0    # progressive-damage accumulation rate [1/s per overstress]
    min_stretch: float = 0.05   # compression floor (incompressible-ish)
    grip_coeff: float = 0.15    # wall-grip (clot→wall normal force) per unit occlusion


def ogden_stress(stretch, p: ClotParams):
    """1-term incompressible Ogden uniaxial Cauchy stress σ(λ) [Pa].

    σ = (2μ/α)(λ^α − λ^(−α/2)). Zero at λ=1; tension (λ>1) positive, compression
    (λ<1) negative; monotone. This is the actual constitutive law used below.
    """
    lam = np.asarray(stretch, dtype=float)
    return (2.0 * p.mu / p.alpha) * (lam ** p.alpha - lam ** (-p.alpha / 2.0))


class ClotField:
    """Finite-extent deformable/damageable clot occlusion o(s) along the centerline."""

    def __init__(self, s_max: float, n_s: int, n_th: int, R_base: float,
                 s0: float, s1: float, height: float,
                 params: ClotParams | None = None):
        self.p = params or ClotParams()
        self.n_s, self.n_th, self.R_base = n_s, n_th, R_base
        s_grid = np.linspace(0.0, s_max, n_s)
        self.mask = (s_grid >= s0) & (s_grid <= s1)          # clot region
        self.o0 = np.where(self.mask, float(height), 0.0)    # initial occlusion
        self.o = self.o0.copy()                              # current occlusion
        self.D = np.zeros(n_s)                               # progressive damage [0,1]
        self.s_grid = s_grid
        self.retrieved = 0.0                                 # proximal distance the clot was pulled

    def occlusion_grid(self) -> np.ndarray:
        """Per-(s,θ) occlusion [n_s*n_th] to subtract from the base lumen radius."""
        return np.repeat(self.o[:, None], self.n_th, axis=1).ravel()

    def _compression_stretch(self, pressure):
        """Solve |σ_compressive(λ)| = pressure for λ∈(min_stretch, 1] (Ogden curve)."""
        lam = np.ones_like(pressure)
        for _ in range(12):
            resist = -ogden_stress(lam, self.p)              # >0 in compression (λ<1)
            f = resist - np.maximum(pressure, 0.0)
            h = 1e-4
            df = (-ogden_stress(lam + h, self.p) + ogden_stress(lam - h, self.p)) / (2 * h)
            lam = np.clip(lam - f / np.where(np.abs(df) < 1e-6, -1e3, df),
                          self.p.min_stretch, 1.0)
        return lam

    def update(self, wall_load_grid: np.ndarray, dt: float) -> float:
        """Advance the clot one substep from the device contact load.

        wall_load_grid: [n_s*n_th] device→wall/clot normal force per cell (from the
        solver). Returns the downstream occlusion fraction for the flow coupling.
        """
        F_dev = wall_load_grid.reshape(self.n_s, self.n_th).sum(axis=1)   # per-s contact force
        pressure = F_dev / self.p.area
        lam = self._compression_stretch(pressure)            # Ogden elastic compression
        over = np.maximum(pressure / self.p.failure_stress - 1.0, 0.0)
        self.D = np.clip(self.D + self.p.damage_rate * over * dt, 0.0, 1.0)  # progressive
        # residual occlusion: initial × (elastic compression) × (1 − damage)
        self.o = np.where(self.mask, self.o0 * lam * (1.0 - self.D), 0.0)
        return float((self.o / self.R_base).max()) if self.mask.any() else 0.0

    def friction_resistance(self, wall_load_grid: np.ndarray) -> float:
        """Coulomb wall friction opposing clot translation = μ·(contact normal force)."""
        F_normal = wall_load_grid.reshape(self.n_s, self.n_th).sum()
        return self.p.friction_mu * F_normal

    def max_damage(self) -> float:
        return float(self.D.max())

    def retrieve(self, delta_s: float, engagement: float, aspiration: float = 0.0) -> dict:
        """Attempt to drag the clot proximally by delta_s with a stent-retriever.

        Force balance (doc §3.4.4): the clot is held by wall friction (μ·N, N ∝
        occlusion) minus aspiration; the stent-retriever grips with `engagement`.
          * net_hold > cohesive strength  -> the clot tears (progressive fragmentation)
          * engagement < net_hold         -> the retriever slips (clot not moved)
          * otherwise                     -> the clot translates proximally (retrieved)
        """
        if not self.mask.any():
            return {"status": "none", "retrieved": self.retrieved}
        occ_mean = float(self.o[self.mask].mean())
        N = self.p.grip_coeff * occ_mean                     # clot→wall normal (grip) force
        net_hold = max(self.p.friction_mu * N - aspiration, 0.0)
        R_coh = self.p.failure_stress * self.p.area          # clot cohesive strength
        if net_hold > R_coh:
            self.D = np.clip(self.D + self.mask * 0.3, 0.0, 1.0)   # tears -> fragments
            self.o = np.where(self.mask, self.o0 * (1.0 - self.D), 0.0)
            return {"status": "fragment", "retrieved": self.retrieved}
        if engagement < net_hold:
            return {"status": "slip", "retrieved": self.retrieved}
        # retrieve: translate the occlusion + damage profiles proximally by delta_s
        self.o = np.interp(self.s_grid + delta_s, self.s_grid, self.o, left=0.0, right=0.0)
        self.D = np.interp(self.s_grid + delta_s, self.s_grid, self.D, left=0.0, right=0.0)
        self.o0 = np.interp(self.s_grid + delta_s, self.s_grid, self.o0, left=0.0, right=0.0)
        self.mask = self.o0 > 1e-6
        self.retrieved += delta_s
        return {"status": "retrieve", "retrieved": self.retrieved}
