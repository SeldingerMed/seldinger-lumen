"""Newton-based guidewire simulation with tube-intrinsic contact (doc §3.2).

The guidewire is a Newton ``add_rod`` cable (stretch + bend/twist), integrated by
the forked ``TubeVBDSolver`` (``lumen.newton.tube_vbd``) which injects the
tube-intrinsic barrier (force + Hessian) natively into VBD's per-color AVBD solve
— so contact is implicit and stable, not an external predictor force. This is the
faithful replatform of Layer 0 onto Newton (doc §3.2: a domain-specialized module
inside the engine, replacing generic device-vs-mesh collision).

Proximal-end actuation (insertion = translate the kinematic base along the vessel
tangent; rotation = spin its quaternion) matches the continuum action space (§1.2).
"""

from __future__ import annotations

import numpy as np
import warp as wp
import newton

from lumen.core.frame import CenterlineFrame
from lumen.newton.tube_vbd import TubeVBDSolver
from lumen.newton.forces import add_world_force, add_body_forces, actuate_bases


class NewtonGuidewireSim:
    def __init__(self, vessel_centerline: np.ndarray, R: float,
                 device_points: np.ndarray, radius: float = 0.2,
                 stretch_stiffness: float = 1.0e4, bend_stiffness: float = 5.0e1,
                 bend_damping: float = 1.0, density: float = 1.0,
                 kappa: float = 2.0e3, d_hat: float = 0.3,
                 barrier_mode: str = "compliant",
                 deformable_wall: bool = False, hgo_params=None,
                 mu_along: float = 0.0, mu_across: float = 0.0,
                 gamma_fric_deg: float = 40.0, lumen_field=None, flow=None,
                 clot_segment=None, clot_height: float = 1.6, clot_params=None,
                 stentriever=None, n_envs: int = 1,
                 vbd_iterations: int = 10, device: str | None = None):
        from lumen.hardware import detect_device
        self.device = device or detect_device()      # cuda if available, else cpu
        self.R, self.kappa, self.d_hat = R, kappa, d_hat
        self.n_envs = int(n_envs)
        self.contact_frame = CenterlineFrame(vessel_centerline)
        # Batched envs share one vessel (the contact is wire-vs-wall, never wire-vs-wire,
        # so E rods in one model are independent). Per-env wall/clot/flow state needs an
        # env dimension that isn't ported yet, so those features are single-env for now.
        if self.n_envs > 1 and (deformable_wall or flow is not None or clot_segment is not None):
            raise NotImplementedError(
                "batched n_envs>1 currently supports the rigid-wall contact sim only; "
                "per-env deformable wall / clot / flow is the next step")

        builder = newton.ModelBuilder(gravity=0.0)
        builder.default_shape_cfg.density = density
        pts = [wp.vec3(*map(float, p)) for p in device_points]
        quats = newton.utils.create_parallel_transport_cable_quaternions(pts)
        self.bodies, self.bases = [], []
        for _ in range(self.n_envs):
            bodies, _ = builder.add_rod(
                pts, quats, radius=radius, stretch_stiffness=stretch_stiffness,
                bend_stiffness=bend_stiffness, bend_damping=bend_damping,
                body_frame_origin="com")
            self.bodies.extend(bodies)
            self.bases.append(bodies[0])
            builder.body_mass[bodies[0]] = 0.0       # kinematic base (proximal actuation)
            builder.body_inv_mass[bodies[0]] = 0.0
            builder.body_inertia[bodies[0]] = wp.mat33(0.0)
            builder.body_inv_inertia[bodies[0]] = wp.mat33(0.0)
        self.n_per_env = len(self.bodies) // self.n_envs   # add_rod: N+1 points -> N bodies
        self.base = self.bases[0]
        builder.color()
        self.model = builder.finalize(device=self.device)

        self.solver = TubeVBDSolver(self.model, iterations=vbd_iterations)
        self.solver.set_tube_contact(vessel_centerline, R, bodies,
                                     kappa=kappa, d_hat=d_hat,
                                     barrier_mode=barrier_mode,
                                     deformable_wall=deformable_wall,
                                     hgo_params=hgo_params, mu_along=mu_along,
                                     mu_across=mu_across, gamma_fric_deg=gamma_fric_deg,
                                     lumen_field=lumen_field)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.body_ids = wp.array(np.array(bodies, dtype=np.int32), dtype=wp.int32,
                                 device=self.device)
        # on-device base actuation (no per-substep body_q host round-trip), one per env
        self._base_ids = wp.array(np.array(self.bases, dtype=np.int32), dtype=wp.int32,
                                  device=self.device)
        self._ins_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
        self._tw_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
        self.flow = flow                 # optional NewtonFlow (lumped) or FlowField (1-D)
        # FlowField exposes a per-s lumen/pressure API; the lumped NewtonFlow doesn't.
        self._flow_is_field = flow is not None and hasattr(flow, "set_lumen")
        # optional finite-extent deformable clot (shares the wall's s,θ grid)
        self.clot = None
        if clot_segment is not None:
            from lumen.newton.clot import ClotField
            w = self.solver._wall
            self.clot = ClotField(s_max=w.s_max, n_s=w.n_s, n_th=w.n_th, R_base=R,
                                  s0=clot_segment[0], s1=clot_segment[1],
                                  height=clot_height, params=clot_params)
        self.stentriever = stentriever   # optional device for clot retrieval
        # snapshot for fast reset (avoid rebuilding the model/solver each episode)
        self._init_body_q = self.state_0.body_q.numpy().copy()

    def reset(self):
        """Restore the initial state cheaply (no model/solver rebuild) — for RL."""
        dev = self.device
        self.state_0.body_q = wp.array(self._init_body_q.copy(), dtype=wp.transform, device=dev)
        self.state_1.body_q = wp.array(self._init_body_q.copy(), dtype=wp.transform, device=dev)
        self.state_0.body_qd.zero_()
        self.state_1.body_qd.zero_()
        self.solver.body_q_prev = wp.array(self._init_body_q.copy(), dtype=wp.transform, device=dev)
        w = getattr(self.solver, "_wall", None)
        if w is not None:
            w.w[:] = 0.0
            w.w_field.zero_()
            w.wall_load.zero_()
        if self.clot is not None:
            self.clot.o = self.clot.o0.copy()
            self.clot.D[:] = 0.0

    def _actuate_base(self, insertion, twist):
        # insertion/twist may be scalars (same action for every env) or length-n_envs
        # arrays (per-env RL actions). On device: translate/rotate each kinematic base
        # about its current axis (#23).
        ins = np.broadcast_to(np.asarray(insertion, np.float32), (self.n_envs,))
        tw = np.broadcast_to(np.asarray(twist, np.float32), (self.n_envs,))
        if not ins.any() and not tw.any():
            return
        self._ins_arr.assign(np.ascontiguousarray(ins))
        self._tw_arr.assign(np.ascontiguousarray(tw))
        wp.launch(actuate_bases, dim=self.n_envs,
                  inputs=[self._base_ids, self._ins_arr, self._tw_arr],
                  outputs=[self.state_0.body_q], device=self.device)

    def step(self, dt: float = 2.5e-2, substeps: int = 5,
             insertion: float = 0.0, twist: float = 0.0, preload=(0.0, 0.0, 0.0),
             aspiration: float = 0.0):
        """Advance the simulation by `dt` total, as `substeps` sub-steps of
        `dt/substeps` each (the standard substep convention)."""
        sub_dt = dt / substeps
        if self.flow is not None and aspiration:
            self.flow.aspiration = aspiration        # aspiration recovers downstream flow
        # flow drag acts along the (slowly-varying) device tangents — compute once/step
        tang = None
        s_nodes = None
        if self.flow is not None:
            pos = self.state_0.body_q.numpy()[self.bodies, :3]
            tang = np.zeros_like(pos)
            tang[1:-1] = pos[2:] - pos[:-2]
            tang[0] = pos[1] - pos[0]
            tang[-1] = pos[-1] - pos[-2]
            tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
            if self._flow_is_field:                  # device-node arc-lengths for local v(s)
                s_nodes = np.array([self.contact_frame.project(p).s for p in pos])
                self.flow.set_tip(float(s_nodes.max()))   # catheter tip = deepest node
        for _ in range(substeps):
            self.state_0.clear_forces()
            self._actuate_base(insertion / substeps, twist / substeps)
            if any(preload):
                wp.launch(add_world_force, dim=self.body_ids.shape[0],
                          inputs=[self.body_ids, float(preload[0]), float(preload[1]),
                                  float(preload[2]), 1],
                          outputs=[self.state_0.body_f], device=self.device)
            wall = getattr(self.solver, "_wall", None)
            # compose the shared effective base radius R0(s,θ,t) = R0_base × pulse −
            # clot occlusion, which the contact kernel reads (+ wall w). M1: pulse
            # modulates the *vessel* wall only, NOT the clot (a clot is incompressible
            # tissue — it doesn't breathe with the cardiac cycle).
            if wall is not None and (self.flow is not None or self.clot is not None):
                pulse = self.flow.pulse_factor() if self.flow is not None else 1.0
                base = wall._R0_base * np.float32(pulse)
                if self.clot is not None:
                    occ = self.clot.occlusion_grid().astype(np.float32)
                    base = base - occ
                    wall.set_clot_mask(occ)              # H1: clot bears its own load
                wall.r0_field.assign(base.astype(np.float32))
            if self.flow is not None:
                self.flow.advance(sub_dt)
                if self._flow_is_field:
                    # feed the SHARED lumen radius (θ-averaged open radius) to the 1-D
                    # network, solve P(s)/v(s), then drag each node by its LOCAL v(s).
                    r_open_s = base.reshape(wall.n_s, wall.n_th).mean(axis=1)
                    self.flow.set_lumen(r_open_s, wall.s_max)
                    self.flow.solve()
                    drag = self.flow.drag_at(s_nodes)[:, None]
                else:
                    drag = self.flow.drag_per_unit_tangent()
                dvecs = wp.array((drag * tang).astype(np.float32),
                                 dtype=wp.vec3, device=self.device)
                wp.launch(add_body_forces, dim=self.body_ids.shape[0],
                          inputs=[self.body_ids, dvecs, 1],
                          outputs=[self.state_0.body_f], device=self.device)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, sub_dt)
            if self.clot is not None:                # clot deforms/damages from contact load
                occ = self.clot.update(wall.wall_load.numpy(), sub_dt)
                if self.flow is not None and not self._flow_is_field:
                    self.flow.occlusion = occ        # lumped model: feed the scalar blockage
            self.state_0, self.state_1 = self.state_1, self.state_0
        # stent-retriever: on retraction, drag the clot proximally (retrieve/slip/fragment)
        if self.stentriever is not None and self.clot is not None and insertion < 0.0:
            if self._flow_is_field and self.clot.mask.any():
                # mobilising force = retrograde pressure gradient the aspiration sink
                # builds across the clot (real hemodynamics), plus any direct command.
                sc = self.clot.s_grid[self.clot.mask]
                asp = aspiration + self.flow.clot_mobilizing_force(float(sc[0]), float(sc[-1]))
            else:
                asp = aspiration + (self.flow.aspiration if self.flow is not None else 0.0)
            self.last_retrieval = self.clot.retrieve(
                -insertion, self.stentriever.engagement_strength(self.clot), asp, dt)

    def body_positions(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.bodies, :3]

    def env_positions(self) -> np.ndarray:
        """Device-node positions per env, shape (n_envs, n_per_env, 3)."""
        return self.body_positions().reshape(self.n_envs, self.n_per_env, 3)

    def node_radii(self) -> np.ndarray:
        return np.array([self.contact_frame.project(p).r for p in self.body_positions()])

    def wall_max_deflection(self) -> float:
        return self.solver.wall_max_deflection()
