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
import numpy as np
from dataclasses import dataclass

try:
    import warp as wp
except Exception:  # pragma: no cover
    wp = None


if wp is not None:
    @wp.func
    def _avg_r(r0_field: wp.array(dtype=wp.float32), celloff: int, i_s: int,
               n_th: int, R_floor: float):
        """θ-averaged open lumen radius at (env, s), floored — the 1-D flow's R(s)."""
        acc = float(0.0)
        for it in range(n_th):
            acc += r0_field[celloff + i_s * n_th + it]
        return wp.max(acc / float(n_th), R_floor)

    @wp.kernel
    def _flow_solve_k(
        r0_field: wp.array(dtype=wp.float32), n_s: int, n_th: int, ds: float,
        visc: float, R_floor: float, Pin: float, R_periph: float, asp_gain: float,
        asp: wp.array(dtype=wp.float32), tip: wp.array(dtype=wp.int32),
        P_out: wp.array(dtype=wp.float32), v_out: wp.array(dtype=wp.float32),
        Qdown_out: wp.array(dtype=wp.float32)):
        """One thread per env: series resistive solve over the env's composed lumen
        radius (reads r0_field directly, no host round-trip). Mirrors FlowField.solve()."""
        e = wp.tid()
        celloff = e * (n_s * n_th)
        soff = e * n_s
        it = tip[e]
        cum_it = float(0.0)
        cum_tot = float(0.0)
        for k in range(n_s - 1):
            rm = 0.5 * (_avg_r(r0_field, celloff, k, n_th, R_floor)
                        + _avg_r(r0_field, celloff, k + 1, n_th, R_floor))
            rseg = visc * ds / (rm * rm * rm * rm)
            if k < it:
                cum_it += rseg
            cum_tot += rseg
        R_up = cum_it
        R_down = (cum_tot - cum_it) + R_periph
        Q_nat = Pin / (R_up + R_down)
        P_tip_nat = Pin - Q_nat * R_up
        a = wp.clamp(asp[e], 0.0, 1.0)
        P_tip = P_tip_nat - a * asp_gain
        Q_up = (Pin - P_tip) / wp.max(R_up, 1.0e-9)
        Q_down = P_tip / wp.max(R_down, 1.0e-9)
        Qdown_out[e] = Q_down
        cum = float(0.0)
        for i in range(n_s):
            ri = _avg_r(r0_field, celloff, i, n_th, R_floor)
            A = 3.14159265 * ri * ri
            if i < it:
                P_out[soff + i] = Pin - Q_up * cum
                v_out[soff + i] = Q_up / A
            else:
                P_out[soff + i] = P_tip - Q_down * (cum - R_up)
                v_out[soff + i] = Q_down / A
            if i < n_s - 1:
                rm = 0.5 * (_avg_r(r0_field, celloff, i, n_th, R_floor)
                            + _avg_r(r0_field, celloff, i + 1, n_th, R_floor))
                cum += visc * ds / (rm * rm * rm * rm)


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

    def __init__(self, params: FlowFieldParams | None = None, n_envs: int = 1,
                 device: str = "cpu"):
        self.p = params or FlowFieldParams()
        self.t = 0.0
        self.n_envs = int(n_envs)
        self.device = device
        self.aspiration = 0.0            # [0,1] suction command at the tip
        self.R_s = None                  # per-node open lumen radius R(s)  [host path]
        self.s_grid = None
        self.tip_s = None                # catheter-tip arc-length (aspiration point)
        self._P = None                   # solved pressure field P(s)  [host path]
        self._v = None                   # solved velocity field v(s)
        self._Q = 0.0                    # solved through-flow (proximal)
        self._Q_down = 0.0
        # Edge-aware vascular-tree host path. Shape convention is [env, edge, s].
        # It deliberately does not flatten edges onto one route: a branch node keeps
        # independent edge fields so tree contact/drag can use the projected edge.
        self._tree_R = None
        self._tree_s_grids = None
        self._tree_P = None
        self._tree_v = None
        self._tree_Q_down = None
        self._tree_tip_edge = None
        self._tree_tip_s = None
        # device fields for the on-device per-env solve (set up lazily on first use)
        self._n_s = None
        self._P_d = self._v_d = self._tip_d = self._asp_d = self._Qdown_d = None

    # --- geometry / actuation set each step by the sim ----------------------
    def set_lumen(self, radius_s, s_max: float) -> None:
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

    # --- on-device per-env solve (batched, no host round-trip) ----------------
    def _ensure_device(self, n_s: int) -> None:
        if self._P_d is not None and self._n_s == n_s:
            return
        self._n_s = n_s
        self._P_d = wp.zeros(self.n_envs * n_s, dtype=wp.float32, device=self.device)
        self._v_d = wp.zeros(self.n_envs * n_s, dtype=wp.float32, device=self.device)
        self._Qdown_d = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
        self._tip_d = wp.zeros(self.n_envs, dtype=wp.int32, device=self.device)
        self._asp_d = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)

    def set_tips(self, s_nodes_per_env, s_max: float, n_s: int) -> None:
        """Per-env catheter tip = the env's deepest node, as a wall-grid s index."""
        self._ensure_device(n_s)
        s = np.asarray(s_nodes_per_env, dtype=float).reshape(self.n_envs, -1)
        tip_s = s.max(axis=1)
        it = np.clip(np.round(tip_s / s_max * (n_s - 1)), 0, n_s - 1).astype(np.int32)
        self._tip_d.assign(it)
        asp = np.clip(np.broadcast_to(np.asarray(self.aspiration, dtype=np.float32), (self.n_envs,)), 0.0, 1.0)
        self._asp_d.assign(asp)

    def solve_device(self, r0_field, n_s: int, n_th: int, s_max: float) -> None:
        """Solve the series network for every env on device, reading the composed
        wall radius field directly (no D2H). Writes the device velocity field v_d."""
        self._ensure_device(n_s)
        ds = s_max / (n_s - 1)
        wp.launch(_flow_solve_k, dim=self.n_envs,
                  inputs=[r0_field, n_s, n_th, float(ds), float(self.p.visc),
                          float(self.p.R_floor), float(self.P_in()), float(self.p.R_periph),
                          float(self.p.asp_gain), self._asp_d, self._tip_d],
                  outputs=[self._P_d, self._v_d, self._Qdown_d], device=self.device)

    def velocity_field_device(self):
        if self._v_d is None:
            raise RuntimeError("Device flow fields not initialized.")
        return self._v_d

    # --- edge-aware vascular-tree solve (host, batched) ----------------------
    def set_tree_lumen(self, radius_env_edge_s, edge_lengths) -> None:
        """Set env×edge lumen radii for tree flow.

        ``radius_env_edge_s`` is [n_envs, n_edges, n_s] after θ-averaging the
        tree wall blocks. Every edge keeps its own arc-length grid; no single
        route-centered centerline is fabricated across a branch.
        """
        R = np.maximum(np.asarray(radius_env_edge_s, dtype=float), self.p.R_floor)
        if R.ndim != 3:
            raise ValueError("tree lumen radii must have shape [n_envs, n_edges, n_s]")
        self.n_envs = int(R.shape[0])
        self._tree_R = R
        self._tree_s_grids = None
        self._tree_tip_edge = None
        self._tree_tip_s = None
        self._tree_P = None
        self._tree_v = None
        self._tree_Q_down = None
        lengths = np.asarray(edge_lengths, dtype=float)
        if lengths.shape != (R.shape[1],):
            raise ValueError("edge_lengths length must match tree lumen edge count")
        self._tree_s_grids = [np.linspace(0.0, float(L), R.shape[2]) for L in lengths]

    def set_tree_tips(self, edge_index, s_tip) -> None:
        """Per-env aspiration/drag tip as an edge id plus local edge arc-length."""
        if self._tree_R is None:
            raise RuntimeError("set_tree_lumen() must be called before set_tree_tips()")
        try:
            edge = np.broadcast_to(np.asarray(edge_index, dtype=int), (self.n_envs,))
        except ValueError as exc:
            raise ValueError("tree tip edge_index must broadcast to (n_envs,)") from exc
        try:
            s = np.broadcast_to(np.asarray(s_tip, dtype=float), (self.n_envs,))
        except ValueError as exc:
            raise ValueError("tree tip s_tip must broadcast to (n_envs,)") from exc
        edge = edge.astype(int, copy=True)
        n_edges = self._tree_R.shape[1]
        if np.any((edge < 0) | (edge >= n_edges)):
            raise ValueError(f"tree tip edge_index values must be in [0, {n_edges})")
        self._tree_tip_edge = edge
        self._tree_tip_s = s.astype(float, copy=True)
        self._tree_P = None
        self._tree_v = None
        self._tree_Q_down = None

    def solve_tree(self) -> None:
        """Solve independent 1-D fields on every tree edge for every env.

        This is edge-aware, not a full Kirchhoff graph solver yet: each edge gets
        the current inlet pressure and distal bed resistance. Local drag is keyed
        by env×edge wall fields instead of a fabricated linear route centerline.
        """
        if self._tree_R is None:
            raise RuntimeError("solve_tree called before setting tree lumen")
        R = self._tree_R
        E, G, S = R.shape
        P = np.empty((E, G, S), dtype=float)
        v = np.empty((E, G, S), dtype=float)
        qdown = np.empty((E, G), dtype=float)
        Pin = self.P_in()
        asp = min(max(self.aspiration, 0.0), 1.0)
        idx = np.arange(S)
        for e in range(E):
            tip_edge = -1 if self._tree_tip_edge is None else int(self._tree_tip_edge[e])
            for g in range(G):
                s_grid = self._tree_s_grids[g]
                radii = R[e, g]
                area = math.pi * radii ** 2
                if S == 1:
                    # Degenerate edge: no along-edge resistance to integrate.
                    # Treat the single sample as a uniform tube with the wall
                    # at the mean radius and R_periph at the bed.
                    r_eff = float(np.maximum(np.mean(radii), self.p.R_floor))
                    R_total = self.p.visc / (r_eff ** 4 + 1e-12) + self.p.R_periph
                    Q_nat = Pin / max(R_total, 1e-9)
                    P_tip_nat = Pin - Q_nat * self.p.visc / (r_eff ** 4 + 1e-12)
                    P_tip = P_tip_nat - (asp * self.p.asp_gain if g == tip_edge else 0.0)
                    P[e, g, 0] = P_tip
                    v[e, g, 0] = P_tip / max(self.p.R_periph, 1e-9) / (math.pi * r_eff ** 2)
                    qdown[e, g] = P_tip / max(self.p.R_periph, 1e-9)
                    continue
                ds = s_grid[1] - s_grid[0]
                r_mid = np.maximum(0.5 * (radii[:-1] + radii[1:]), self.p.R_floor)
                r_seg = self.p.visc * ds / (r_mid ** 4 + 1e-12)
                cum = np.concatenate([[0.0], np.cumsum(r_seg)])
                if g == tip_edge and self._tree_tip_s is not None:
                    # np.rint is half-to-even, deterministic across platforms;
                    # clip into [0, S-1] before the cast so an out-of-range
                    # s_tip (e.g. wall overshoot) can't trip an IndexError
                    # in the cum lookup.
                    raw = self._tree_tip_s[e] / max(s_grid[-1], 1e-9) * (S - 1)
                    it = int(np.clip(np.rint(raw), 0, S - 1))
                else:
                    it = S - 1
                R_up = cum[it]
                R_down = (cum[-1] - cum[it]) + self.p.R_periph
                Q_nat = Pin / max(R_up + R_down, 1e-9)
                P_tip_nat = Pin - Q_nat * R_up
                P_tip = P_tip_nat - (asp * self.p.asp_gain if g == tip_edge else 0.0)
                Q_up = (Pin - P_tip) / max(R_up, 1e-9)
                Q_down = P_tip / max(R_down, 1e-9)
                if it == 0:
                    P[e, g] = np.full(S, P_tip)
                else:
                    P_up_segment = Pin - Q_up * cum[:it + 1]
                    P_down_segment = P_tip - Q_down * (cum[it:] - cum[it])
                    P[e, g, :it + 1] = P_up_segment
                    P[e, g, it + 1:] = P_down_segment[1:]
                Qn = np.where(idx < it, Q_up, Q_down)
                v[e, g] = Qn / area
                qdown[e, g] = Q_down
        self._tree_P, self._tree_v, self._tree_Q_down = P, v, qdown
        self._Q = float(np.mean(np.maximum(qdown, 0.0)))
        self._Q_down = self._Q

    def drag_at_tree(self, env_index, edge_index, s_query):
        """Local axial drag for tree nodes, addressed by env + projected edge + s.

        Args:
            env_index (int or np.ndarray): Environment index.
            edge_index (int or np.ndarray): Projected edge index in the tree topology.
            s_query (float or np.ndarray): 1D local axial coordinate along the edge.

        Expected Shapes:
            Inputs `env_index`, `edge_index`, and `s_query` must have matching shapes.

        Interpolation:
            Retrieves the fluid velocity and area along the queried edge and linearly
            interpolates the local flow properties to compute the resulting axial drag.
        """
        env = np.asarray(env_index, dtype=int)
        edge = np.asarray(edge_index, dtype=int)
        s = np.asarray(s_query, dtype=float)
        if env.shape != edge.shape or env.shape != s.shape:
            raise ValueError(f"tree drag inputs must have matching shapes, got env={env.shape}, edge={edge.shape}, s={s.shape}")
        tree_v = self._tree_v
        tree_s_grids = self._tree_s_grids
        if tree_v is None or tree_s_grids is None:
            raise RuntimeError("solve_tree() must be called before drag_at_tree()")
        env_f = env.ravel()
        edge_f = edge.ravel()
        s_f = s.ravel()
        n_envs, n_edges, n_s = tree_v.shape
        if np.any((env_f < 0) | (env_f >= n_envs)):
            raise ValueError(f"tree drag env_index values must be in [0, {n_envs})")
        if np.any((edge_f < 0) | (edge_f >= n_edges)):
            raise ValueError(f"tree drag edge_index values must be in [0, {n_edges})")
        if n_s == 1:
            out_f = tree_v[env_f, edge_f, 0]
            return (self.p.drag_coeff * out_f).reshape(s.shape)
        lengths = np.asarray([grid[-1] for grid in tree_s_grids], dtype=float)
        denom = np.maximum(lengths[edge_f], 1e-12)
        x = np.clip(s_f / denom, 0.0, 1.0) * (n_s - 1)
        lo = np.floor(x).astype(int)
        hi = np.minimum(lo + 1, n_s - 1)
        w = x - lo
        v_lo = tree_v[env_f, edge_f, lo]
        v_hi = tree_v[env_f, edge_f, hi]
        out_f = self.p.drag_coeff * ((1.0 - w) * v_lo + w * v_hi)
        return out_f.reshape(s.shape)

    def tree_pressure_fields(self):
        return self._tree_P

    def tree_velocity_fields(self):
        return self._tree_v

    def tree_downstream_Q(self):
        return self._tree_Q_down

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
        if self._v is None:
            return np.zeros_like(np.asarray(s_query, dtype=float))
        v = np.interp(np.asarray(s_query, dtype=float), self.s_grid, self._v)
        return self.p.drag_coeff * v

    def drag_per_unit_tangent(self, t: float | None = None) -> float:
        """API-compat scalar drag (mean local velocity) for non-field callers."""
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
        if self._P is None:
            return 0.0
        m = (self.s_grid >= s0) & (self.s_grid <= s1)
        if not m.any():
            return 0.0
        idx = np.where(m)[0]
        P_prox, P_dist = self._P[idx[0]], self._P[idx[-1]]
        A_clot = math.pi * float(np.mean(self.R_s[idx])) ** 2
        return (P_dist - P_prox) * A_clot
