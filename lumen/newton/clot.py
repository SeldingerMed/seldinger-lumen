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

try:
    import warp as wp
except Exception:  # pragma: no cover
    wp = None


if wp is not None:
    @wp.func
    def _ogden_resist(lam: float, mu: float, alpha: float):
        """Compressive Ogden resistance −σ(λ) (>0 for λ<1); matches ogden_stress()."""
        return -(2.0 * mu / alpha) * (wp.pow(lam, alpha) - wp.pow(lam, -alpha * 0.5))

    @wp.kernel
    def _clot_update_k(
        wall_load: wp.array(dtype=wp.float32), n_s: int, n_th: int, area: float,
        mu: float, alpha: float, failure: float, dmg_rate: float, min_stretch: float,
        dt: float, o0: wp.array(dtype=wp.float32), mask: wp.array(dtype=wp.float32),
        D: wp.array(dtype=wp.float32), o_out: wp.array(dtype=wp.float32)):
        """One thread per (env, s) cell: Ogden compression + progressive damage, on
        device. Reads the per-env wall_load block directly (no D2H). Mirrors
        ClotField.update()'s numpy math so the two stay in parity."""
        j = wp.tid()                                   # env*n_s + i_s
        if mask[j] == 0.0:
            o_out[j] = 0.0
            return
        env = j // n_s
        i_s = j % n_s
        base = env * (n_s * n_th) + i_s * n_th
        F = float(0.0)
        for it in range(n_th):
            F += wall_load[base + it]
        pressure = wp.max(F / area, 0.0)
        lam = float(1.0)
        for _ in range(12):                            # Newton solve |σ_comp(λ)| = pressure
            f = _ogden_resist(lam, mu, alpha) - pressure
            h = float(1.0e-4)
            df = (_ogden_resist(lam + h, mu, alpha) - _ogden_resist(lam - h, mu, alpha)) / (2.0 * h)
            denom = df
            if wp.abs(df) < 1.0e-6:
                denom = -1.0e3
            lam = wp.clamp(lam - f / denom, min_stretch, 1.0)
        over = wp.max(pressure / failure - 1.0, 0.0)
        nd = wp.clamp(D[j] + dmg_rate * over * dt, 0.0, 1.0)
        D[j] = nd
        o_out[j] = o0[j] * lam * (1.0 - nd)


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
                 params: ClotParams | None = None, n_envs: int = 1,
                 device: str = "cpu"):
        self.p = params or ClotParams()
        self.n_s, self.n_th, self.R_base = n_s, n_th, R_base
        self.n_envs = int(n_envs)
        self.device = device
        s_grid = np.linspace(0.0, s_max, n_s)
        self.mask = (s_grid >= s0) & (s_grid <= s1)          # clot region (per-s, shared)
        self.o0 = np.where(self.mask, float(height), 0.0)    # initial occlusion
        self.o = self.o0.copy()                              # current occlusion
        self.D = np.zeros(n_s)                               # progressive damage [0,1]
        self.s_grid = s_grid
        self.retrieved = 0.0                                 # proximal distance the clot was pulled
        # Batched retrieval state. Public single-env attributes above mirror env 0
        # for backward compatibility; n_envs>1 updates these env arrays and pushes
        # them back to the device mirrors after each retrieval event.
        self.o0_env = np.tile(self.o0, (self.n_envs, 1))
        self.o_env = self.o0_env.copy()
        self._initial_o0_env = self.o0_env.copy()
        self.D_env = np.zeros((self.n_envs, n_s), dtype=float)
        self.mask_env = np.tile(self.mask, (self.n_envs, 1))
        self.retrieved_env = np.zeros(self.n_envs, dtype=float)
        # device mirrors (per env): same initial clot replicated, evolves per env
        if wp is not None:
            mask_f = self.mask.astype(np.float32)
            self.o0_d = wp.array(self.o0_env.astype(np.float32).ravel(),
                                 dtype=wp.float32, device=device)
            self.mask_d = wp.array(np.tile(mask_f, self.n_envs), dtype=wp.float32, device=device)
            self.D_d = wp.zeros(self.n_envs * n_s, dtype=wp.float32, device=device)
            self.o_d = wp.array(self.o_env.astype(np.float32).ravel(),
                                dtype=wp.float32, device=device)

    def _sync_public_from_env(self, env: int = 0) -> None:
        self.o0 = self.o0_env[env].copy()
        self.o = self.o_env[env].copy()
        self.D = self.D_env[env].copy()
        self.mask = self.mask_env[env].copy()
        self.retrieved = float(self.retrieved_env[env])

    def _sync_env0_from_public(self) -> None:
        self.o0_env[0] = self.o0
        self.o_env[0] = self.o
        self.D_env[0] = self.D
        self.mask_env[0] = self.mask
        self.retrieved_env[0] = self.retrieved

    def sync_from_device(self) -> None:
        """Refresh host batched arrays from device clot state before host retrieval."""
        if wp is None or not hasattr(self, "o_d"):
            return
        self.o_env = self.o_d.numpy().reshape(self.n_envs, self.n_s).astype(float)
        self.D_env = self.D_d.numpy().reshape(self.n_envs, self.n_s).astype(float)
        self.o0_env = self.o0_d.numpy().reshape(self.n_envs, self.n_s).astype(float)
        self.mask_env = self.mask_d.numpy().reshape(self.n_envs, self.n_s) > 0.5
        self._sync_public_from_env(0)

    def sync_to_device(self) -> None:
        """Push host batched retrieval state back to device mirrors."""
        if wp is None or not hasattr(self, "o_d"):
            return
        self.o_d.assign(np.ascontiguousarray(self.o_env.astype(np.float32).ravel()))
        self.D_d.assign(np.ascontiguousarray(self.D_env.astype(np.float32).ravel()))
        self.o0_d.assign(np.ascontiguousarray(self.o0_env.astype(np.float32).ravel()))
        self.mask_d.assign(np.ascontiguousarray(self.mask_env.astype(np.float32).ravel()))

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
        wall_load_grid = np.nan_to_num(wall_load_grid, nan=0.0, posinf=0.0, neginf=0.0)  # L5
        F_dev = wall_load_grid.reshape(self.n_s, self.n_th).sum(axis=1)   # per-s contact force
        pressure = F_dev / self.p.area
        lam = self._compression_stretch(pressure)            # Ogden elastic compression
        over = np.maximum(pressure / self.p.failure_stress - 1.0, 0.0)
        self.D = np.clip(self.D + self.p.damage_rate * over * dt, 0.0, 1.0)  # progressive
        # residual occlusion: initial × (elastic compression) × (1 − damage)
        self.o = np.where(self.mask, self.o0 * lam * (1.0 - self.D), 0.0)
        self._sync_env0_from_public()
        return float((self.o / self.R_base).max()) if self.mask.any() else 0.0

    def update_device(self, wall_load_field, dt: float) -> None:
        """On-device clot update (no D2H): launch the Ogden/damage kernel over all
        (env, s) cells, reading the solver's per-env wall_load block in place. Updates
        o_d/D_d on device for the radius-composition kernel to read."""
        wp.launch(_clot_update_k, dim=self.n_envs * self.n_s,
                  inputs=[wall_load_field, self.n_s, self.n_th, float(self.p.area),
                          float(self.p.mu), float(self.p.alpha), float(self.p.failure_stress),
                          float(self.p.damage_rate), float(self.p.min_stretch), float(dt),
                          self.o0_d, self.mask_d, self.D_d],
                  outputs=[self.o_d], device=self.device)

    def friction_resistance(self, wall_load_grid: np.ndarray) -> float:
        """Coulomb wall friction opposing clot translation = μ·(contact normal force)."""
        F_normal = wall_load_grid.reshape(self.n_s, self.n_th).sum()
        return self.p.friction_mu * F_normal

    def max_damage(self) -> float:
        return float(self.D.max())

    def clot_centers(self) -> np.ndarray:
        """Per-env mean arc-length of live clot cells (nan when no clot remains)."""
        out = np.full(self.n_envs, np.nan, dtype=float)
        for e in range(self.n_envs):
            if self.mask_env[e].any():
                out[e] = float(self.s_grid[self.mask_env[e]].mean())
        return out

    def retrieve(self, delta_s: float, engagement: float, aspiration: float = 0.0,
                 dt: float = 2.5e-2) -> dict:
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
            # M2: progressive, rate- and overstress-scaled fragmentation (same damage
            # law as update()), not a fixed 0.3 jump — so it integrates with dt.
            over = net_hold / R_coh - 1.0
            self.D = np.clip(self.D + self.mask * self.p.damage_rate * over * dt, 0.0, 1.0)
            self.o = np.where(self.mask, self.o0 * (1.0 - self.D), 0.0)
            self._sync_env0_from_public()
            self.sync_to_device()
            return {"status": "fragment", "retrieved": self.retrieved}
        if engagement < net_hold:
            self._sync_env0_from_public()
            return {"status": "slip", "retrieved": self.retrieved}
        # retrieve: translate the occlusion + damage profiles proximally by delta_s
        self.o = np.interp(self.s_grid + delta_s, self.s_grid, self.o, left=0.0, right=0.0)
        self.D = np.interp(self.s_grid + delta_s, self.s_grid, self.D, left=0.0, right=0.0)
        self.o0 = np.interp(self.s_grid + delta_s, self.s_grid, self.o0, left=0.0, right=0.0)
        self.mask = self.o0 > 1e-6
        self.retrieved += delta_s
        self._sync_env0_from_public()
        self.sync_to_device()
        return {"status": "retrieve", "retrieved": self.retrieved}

    def retrieve_batched(self, delta_s: float | np.ndarray,
                         engagement: float | np.ndarray,
                         aspiration: float | np.ndarray = 0.0,
                         dt: float = 2.5e-2) -> list[dict]:
        """Attempt independent retrieval in each env from batched host arrays.

        Scalar inputs will be broadcast to all environments. Array inputs must
        have shape (n_envs,).

        ``delta_s``, ``engagement``, and ``aspiration`` accept either scalars
        (broadcast to every env) or arrays with shape ``(n_envs,)``. Retrieval
        results and clot mutation stay per-env in ``*_env`` arrays; public
        single-env attributes mirror env 0 after the batched update.

        Returns:
            list[dict]: per-environment results, each with "status" ("retrieve", "slip",
            "fragment", or "none") and "retrieved" distance.
        """
        self.sync_from_device()
        for name, val in [("delta_s", delta_s), ("engagement", engagement), ("aspiration", aspiration)]:
            # 0-d arrays count as scalars and broadcast like Python floats.
            if isinstance(val, np.ndarray) and val.ndim > 0 and val.shape != (self.n_envs,):
                raise ValueError(f"Expected shape ({self.n_envs},) for {name}, got {val.shape}")
        delta = np.broadcast_to(np.asarray(delta_s, dtype=float), (self.n_envs,))
        grip = np.broadcast_to(np.asarray(engagement, dtype=float), (self.n_envs,))
        asp = np.broadcast_to(np.asarray(aspiration, dtype=float), (self.n_envs,))
        results: list[dict] = []
        R_coh = self.p.failure_stress * self.p.area
        for e in range(self.n_envs):
            mask = self.mask_env[e]
            if not mask.any() or delta[e] <= 0.0:
                results.append({"status": "none", "retrieved": float(self.retrieved_env[e])})
                continue
            occ_mean = float(self.o_env[e, mask].mean())
            N = self.p.grip_coeff * occ_mean
            net_hold = max(self.p.friction_mu * N - float(asp[e]), 0.0)
            if net_hold > R_coh:
                over = net_hold / R_coh - 1.0
                self.D_env[e] = np.clip(
                    self.D_env[e] + mask * self.p.damage_rate * over * dt, 0.0, 1.0)
                self.o_env[e] = np.where(mask, self.o0_env[e] * (1.0 - self.D_env[e]), 0.0)
                results.append({"status": "fragment", "retrieved": float(self.retrieved_env[e])})
                continue
            if grip[e] < net_hold:
                results.append({"status": "slip", "retrieved": float(self.retrieved_env[e])})
                continue
            self.o_env[e] = np.interp(self.s_grid + delta[e], self.s_grid, self.o_env[e],
                                      left=0.0, right=0.0)
            self.D_env[e] = np.interp(self.s_grid + delta[e], self.s_grid, self.D_env[e],
                                      left=0.0, right=0.0)
            self.o0_env[e] = np.interp(self.s_grid + delta[e], self.s_grid, self.o0_env[e],
                                       left=0.0, right=0.0)
            self.mask_env[e] = self.o0_env[e] > 1e-6
            self.retrieved_env[e] += delta[e]
            results.append({"status": "retrieve", "retrieved": float(self.retrieved_env[e])})
        self._sync_public_from_env(0)
        self.sync_to_device()
        return results
