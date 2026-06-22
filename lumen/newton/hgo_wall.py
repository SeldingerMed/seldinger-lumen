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
    # Units are the solver's CONSISTENT non-dimensional system (#12): geometry
    # (R0, thickness) and stiffnesses (C10, k1) share one scale so HGO pressure
    # ~ contact load. Absolute SI calibration against real vessel data is the
    # private layer (§8). Defaults below are sim-consistent (not SI).
    C10: float = 8.0e3       # neo-Hookean ground matrix
    k1: float = 5.0e3        # fiber stiffness
    k2: float = 1.0          # fiber exponential stiffening [-]
    kappa_d: float = 0.1     # Gasser fiber dispersion [0=aligned, 1/3=isotropic]
    gamma_deg: float = 40.0  # fiber angle from circumferential direction [deg]
    thickness: float = 0.3   # wall thickness (sim units, consistent with R0)


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

    Incompressible thin-shell Laplace law p = σ_θθ·t / R with the *current* config:
    σ_θθ = λθ·(dΨ/dλθ) (Cauchy hoop), thinned wall t = t0/λθ, current radius R =
    R0·λθ. Those combine to p = (dΨ/dλθ)·t0/(R0·λθ) — i.e. the energy-derivative
    stress times t0/R0, divided by λθ (the wall-thinning factor). Monotone in w.
    """
    lam = 1.0 + np.asarray(w, dtype=float) / R0
    return hgo_radial_stress(lam, p) * p.thickness / (R0 * lam)


if wp is not None:
    @wp.func
    def _hgo_pressure_wp(w: float, R0: float, C10: float, k1: float, k2: float,
                         kd: float, cg2: float, sg2: float, thickness: float):
        """HGO inward shell pressure at radial displacement w (device-side)."""
        lam = 1.0 + w / R0
        I1 = 1.0 / (lam * lam) + lam * lam + 1.0
        I4 = lam * lam * cg2 + sg2
        dI1 = 2.0 * lam - 2.0 / (lam * lam * lam)
        dI4 = 2.0 * lam * cg2
        E = kd * (I1 - 3.0) + (1.0 - 3.0 * kd) * (I4 - 1.0)
        dpsi = C10 * dI1
        if E > 0.0:
            dE = kd * dI1 + (1.0 - 3.0 * kd) * dI4
            dpsi = dpsi + 2.0 * k1 * E * wp.exp(k2 * E * E) * dE
        return dpsi * thickness / (R0 * lam)        # /λθ: incompressible wall thinning

    @wp.kernel
    def _wall_solve_kernel(
        wall_load: wp.array(dtype=wp.float32), cell_area: wp.array(dtype=wp.float32),
        r0_grid: wp.array(dtype=wp.float32), w_in: wp.array(dtype=wp.float32),
        clot_mask: wp.array(dtype=wp.float32),
        C10: float, k1: float, k2: float, kd: float, cg2: float, sg2: float,
        thickness: float, w_eq: wp.array(dtype=wp.float32)):
        c = wp.tid()
        # H1 — at clot cells the contact load is borne by the clot (it compresses),
        # not the HGO wall; routing it to both would double-count it in R_eff.
        pc = wall_load[c] * clot_mask[c] / cell_area[c]
        R0 = r0_grid[c]
        w = wp.max(w_in[c], 0.0)
        for _ in range(8):                         # per-cell Newton solve hgo_p(w)=pc
            f = _hgo_pressure_wp(w, R0, C10, k1, k2, kd, cg2, sg2, thickness) - pc
            h = 1.0e-5
            df = (_hgo_pressure_wp(w + h, R0, C10, k1, k2, kd, cg2, sg2, thickness)
                  - _hgo_pressure_wp(w - h, R0, C10, k1, k2, kd, cg2, sg2, thickness)) / (2.0 * h)
            w = wp.clamp(w - f / wp.max(df, 1.0e-3), 0.0, 0.9 * R0)
        w_eq[c] = w

    @wp.kernel
    def _wall_smooth_kernel(
        w_eq: wp.array(dtype=wp.float32), r0_grid: wp.array(dtype=wp.float32),
        n_s: int, n_th: int, smooth: float, w_out: wp.array(dtype=wp.float32)):
        c = wp.tid()
        # per-env block of n_s*n_th cells; neighbours must stay within the same env
        ncell = n_s * n_th
        off = (c // ncell) * ncell
        local = c % ncell
        i_s = local // n_th
        i_th = local % n_th
        sm = wp.max(i_s - 1, 0)                    # zero-flux at s ends (not connected)
        sp = wp.min(i_s + 1, n_s - 1)
        lap_s = w_eq[off + sm * n_th + i_th] + w_eq[off + sp * n_th + i_th] - 2.0 * w_eq[c]
        tm = (i_th - 1 + n_th) % n_th             # periodic in theta
        tp = (i_th + 1) % n_th
        lap_th = w_eq[off + i_s * n_th + tm] + w_eq[off + i_s * n_th + tp] - 2.0 * w_eq[c]
        val = w_eq[c] + smooth * (lap_s + lap_th)
        w_out[c] = wp.clamp(val, 0.0, 0.9 * r0_grid[c])


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
                 device: str = "cpu", n_envs: int = 1):
        """R0 may be a scalar (cylinder) or a per-cell array R0(s,θ) of length
        n_s*n_th (stenosis/aneurysm/patient anatomy).

        For batched sims (n_envs>1) every device array carries a leading env block of
        n_s*n_th cells, laid out [env0 cells | env1 cells | ...]; the same base R0(s,θ)
        is replicated to each env (one shared vessel) but each env's wall deforms
        independently from its own wire's contact load."""
        self.s_max, self.n_s, self.n_th = s_max, n_s, n_th
        self.n_envs = int(n_envs)
        self.p = params or HGOParams()
        self.smooth = smooth
        self.device = device
        n = n_s * n_th
        self.n_cells = n
        if np.isscalar(R0):
            base = np.full(n, float(R0))
        else:
            base = np.asarray(R0, dtype=float).ravel()
            assert base.size == n, "R0 grid must be length n_s*n_th"
        self.R0_grid = np.tile(base, self.n_envs)                  # [n_envs*n]
        cell_area = (s_max / n_s) * (base * 2.0 * np.pi / n_th)    # per cell, one env
        self.cell_area = np.tile(cell_area, self.n_envs)
        total = self.n_envs * n
        self.w = np.zeros(total, dtype=np.float64)
        if wp is not None:
            self.w_field = wp.zeros(total, dtype=wp.float32, device=device)
            self.wall_load = wp.zeros(total, dtype=wp.float32, device=device)
            self.r0_field = wp.array(self.R0_grid.astype(np.float32), dtype=wp.float32,
                                     device=device)   # base R0(s,θ) for the contact kernel
            self._R0_base = self.R0_grid.astype(np.float32).copy()   # unpulsed resting radius
            self.cell_area_field = wp.array(self.cell_area.astype(np.float32),
                                            dtype=wp.float32, device=device)
            self.w_eq_field = wp.zeros(total, dtype=wp.float32, device=device)  # solve scratch
            # H1 — 1=wall bears the contact load here, 0=clot cell (clot bears it)
            self.clot_mask_field = wp.array(np.ones(total, np.float32), dtype=wp.float32,
                                            device=device)

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
        """Solve HGO equilibrium + shell smoothing ON THE DEVICE (no host roundtrip).

        #8 — two Warp kernels (per-cell Newton solve, then zero-flux-s / periodic-θ
        smoothing with re-clamp) run on the same device as the sim, so the wall
        co-sim does not break GPU batching with a per-step D2H/H2D copy.
        """
        g = np.radians(self.p.gamma_deg)
        cg2, sg2 = float(np.cos(g) ** 2), float(np.sin(g) ** 2)
        total = self.n_envs * self.n_cells
        wp.launch(_wall_solve_kernel, dim=total,
                  inputs=[self.wall_load, self.cell_area_field, self.r0_field,
                          self.w_field, self.clot_mask_field, float(self.p.C10),
                          float(self.p.k1), float(self.p.k2), float(self.p.kappa_d),
                          cg2, sg2, float(self.p.thickness)],
                  outputs=[self.w_eq_field], device=self.device)
        wp.launch(_wall_smooth_kernel, dim=total,
                  inputs=[self.w_eq_field, self.r0_field, self.n_s, self.n_th,
                          float(self.smooth)],
                  outputs=[self.w_field], device=self.device)

    def max_deflection(self) -> float:
        # read the device field on demand (diagnostics only, not per-step)
        return float(self.w_field.numpy().max())

    def set_pulse(self, factor: float) -> None:
        """Modulate the resting lumen radius R0(s,θ,t) = R0_base · factor (pulsatility)."""
        self.r0_field.assign(self._R0_base * float(factor))

    def set_clot_mask(self, occlusion_grid) -> None:
        """Route contact load away from clot cells (H1): cells with occlusion are
        borne by the clot, so the HGO wall solve zeroes its load there."""
        m = (np.asarray(occlusion_grid, dtype=np.float32).ravel() <= 1e-9).astype(np.float32)
        self.clot_mask_field.assign(m)
