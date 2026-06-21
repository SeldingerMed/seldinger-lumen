"""Holzapfel-Gasser-Ogden (HGO) anisotropic hyperelastic vessel wall (doc §3.4.2).

The wall is a reduced thin shell whose lumen radius is the shared field
R(s,θ,t) = R0 + w(s,θ) (doc §3.5.6) — the SAME field the tube-intrinsic contact
barrier reads. Its constitutive response is HGO: an isotropic neo-Hookean ground
matrix plus two collagen fiber families oriented at ±γ to the circumferential
direction, with exponential stiffening (fibers bear tension only). This is the
gold standard in vascular biomechanics and the basis of clinically meaningful
inverse estimates (wall stiffness ↔ perforation/dissection/vasospasm risk).

Reduced kinematics for radial expansion of a cylindrical wall (incompressible,
axially tethered):
    λθ = 1 + w/R0           (circumferential / hoop stretch)
    λz = 1                  (reduced model; axial tethering)
    λr = 1/(λθ·λz)          (incompressibility)
    I1 = λr² + λθ² + λz²
    I4 = I6 = λθ²·cos²γ + λz²·sin²γ      (fiber stretch invariant, both families)

HGO strain energy (per unit reference volume), Holzapfel-Gasser-Ogden 2000 with
Gasser dispersion κd ∈ [0, 1/3]:
    Ψ = C10·(I1−3) + (k1/k2)·[exp(k2·⟨E⟩²) − 1]            (two symmetric families)
    E = κd·(I1−3) + (1−3κd)·(I4−1),   ⟨E⟩ = max(E, 0)

Real-data HGO calibration is the proprietary layer; the defaults here are generic
literature-scale carotid values (calibratable), per the open/closed split (§8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import warp as wp
except Exception:  # pragma: no cover
    wp = None


@dataclass
class HGOParams:
    C10: float = 8.0e3       # neo-Hookean ground matrix [Pa]
    k1: float = 5.0e3        # fiber stiffness [Pa]
    k2: float = 1.0          # fiber exponential stiffening [-]
    kappa_d: float = 0.1     # Gasser fiber dispersion [0=aligned, 1/3=isotropic]
    gamma_deg: float = 40.0  # fiber angle from circumferential direction [deg]
    thickness: float = 0.5e-3  # wall thickness [m]


def _invariants(lam_theta, p: HGOParams):
    lam_th = np.asarray(lam_theta, dtype=float)
    lam_z = 1.0
    lam_r = 1.0 / (lam_th * lam_z)
    I1 = lam_r ** 2 + lam_th ** 2 + lam_z ** 2
    g = np.radians(p.gamma_deg)
    I4 = lam_th ** 2 * np.cos(g) ** 2 + lam_z ** 2 * np.sin(g) ** 2
    return I1, I4, lam_th


def hgo_psi(lam_theta, p: HGOParams):
    """HGO strain energy density Ψ(λθ) [Pa]. Zero at λθ=1."""
    I1, I4, _ = _invariants(lam_theta, p)
    E = p.kappa_d * (I1 - 3.0) + (1.0 - 3.0 * p.kappa_d) * (I4 - 1.0)
    E = np.maximum(E, 0.0)                       # fibers bear tension only
    psi_iso = p.C10 * (I1 - 3.0)
    psi_fib = (p.k1 / p.k2) * (np.exp(p.k2 * E ** 2) - 1.0)
    return psi_iso + psi_fib


def hgo_radial_stress(lam_theta, p: HGOParams):
    """Restoring stress conjugate to radial expansion, σ = dΨ/dλθ [Pa].

    Positive σ resists outward expansion (λθ>1). Computed analytically.
    """
    I1, I4, lam_th = _invariants(lam_theta, p)
    g = np.radians(p.gamma_deg)
    # dI1/dλθ = 2λθ − 2/λθ³   (λz=1, λr=1/λθ)
    dI1 = 2.0 * lam_th - 2.0 / lam_th ** 3
    dI4 = 2.0 * lam_th * np.cos(g) ** 2
    E = p.kappa_d * (I1 - 3.0) + (1.0 - 3.0 * p.kappa_d) * (I4 - 1.0)
    Epos = np.maximum(E, 0.0)
    dE = p.kappa_d * dI1 + (1.0 - 3.0 * p.kappa_d) * dI4
    dpsi_iso = p.C10 * dI1
    # d/dλθ [(k1/k2)(exp(k2 E²)−1)] = (k1/k2)·exp(k2 E²)·2 k2 E·dE  (only E>0)
    dpsi_fib = np.where(E > 0.0,
                        2.0 * p.k1 * Epos * np.exp(p.k2 * Epos ** 2) * dE, 0.0)
    return dpsi_iso + dpsi_fib


def hgo_wall_pressure(w, R0, p: HGOParams):
    """Inward restoring pressure of the thin HGO shell at radial displacement w [Pa].

    Thin-shell hoop relation p = σ·t/R, with σ the HGO circumferential stress at
    hoop stretch λθ = 1 + w/R0. Monotone increasing in w (stiffening), so the
    per-cell equilibrium solve below is well-posed.
    """
    lam = 1.0 + np.asarray(w, dtype=float) / R0
    return hgo_radial_stress(lam, p) * p.thickness / R0


class WallField:
    """Deformable lumen radius field w(s,θ) with HGO mechanics, sharing R (§3.5.6).

    Each substep the contact deposits a normal load per cell; the wall solves to
    quasi-static equilibrium (the wall relaxes far faster than the device moves)
    where the HGO restoring pressure balances the contact pressure, then a light
    shell smoothing couples neighbours (hoop/axial continuity). The resulting
    w-field is uploaded for the contact barrier to read as R_eff = R0 + w.
    """

    def __init__(self, R0, s_max: float, n_s: int = 40, n_th: int = 16,
                 params: HGOParams | None = None, smooth: float = 0.15,
                 device: str = "cpu"):
        """R0 may be a scalar (cylinder) or a per-cell array R0(s,θ) of length
        n_s*n_th (stenosis/aneurysm/patient anatomy)."""
        self.s_max, self.n_s, self.n_th = s_max, n_s, n_th
        self.p = params or HGOParams()
        self.smooth = smooth
        self.device = device
        n = n_s * n_th
        if np.isscalar(R0):
            self.R0_grid = np.full(n, float(R0))
        else:
            self.R0_grid = np.asarray(R0, dtype=float).ravel()
            assert self.R0_grid.size == n, "R0 grid must be length n_s*n_th"
        self.cell_area = (s_max / n_s) * (self.R0_grid * 2.0 * np.pi / n_th)  # [m²] per cell
        self.w = np.zeros(n, dtype=np.float64)
        if wp is not None:
            self.w_field = wp.zeros(n, dtype=wp.float32, device=device)
            self.wall_load = wp.zeros(n, dtype=wp.float32, device=device)
            self.r0_field = wp.array(self.R0_grid.astype(np.float32), dtype=wp.float32,
                                     device=device)   # base R0(s,θ) for the contact kernel

    def _solve_cell(self, p_contact, w0, iters=8):
        """1-D Newton solve of hgo_wall_pressure(w)=p_contact per cell (vectorised, per-cell R0)."""
        w = np.maximum(w0, 0.0)
        R0 = self.R0_grid
        for _ in range(iters):
            f = hgo_wall_pressure(w, R0, self.p) - p_contact
            h = 1e-7
            df = (hgo_wall_pressure(w + h, R0, self.p)
                  - hgo_wall_pressure(w - h, R0, self.p)) / (2 * h)
            w = np.clip(w - f / np.maximum(df, 1e-3), 0.0, 0.9 * R0)
        return w

    def update_from_load(self):
        """Read accumulated contact load, solve HGO equilibrium, smooth, upload."""
        load = self.wall_load.numpy().astype(np.float64)         # normal force per cell [N]
        p_contact = load / self.cell_area                        # pressure [Pa]
        w_eq = self._solve_cell(p_contact, self.w)
        W = w_eq.reshape(self.n_s, self.n_th)
        lap = (np.roll(W, 1, 0) + np.roll(W, -1, 0)
               + np.roll(W, 1, 1) + np.roll(W, -1, 1) - 4 * W)
        self.w = (W + self.smooth * lap).ravel()
        self.w_field.assign(self.w.astype(np.float32))

    def max_deflection(self) -> float:
        return float(self.w.max())
