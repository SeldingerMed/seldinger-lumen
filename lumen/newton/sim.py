"""Newton-based guidewire simulation with tube-intrinsic contact (doc §3.2).

The guidewire is a Newton ``add_rod`` cable (stretch + bend/twist), integrated by
the forked ``TubeVBDSolver`` (``lumen.newton.tube_vbd``) which injects the
tube-intrinsic barrier (force + Hessian) natively into VBD's per-color AVBD solve
— so contact is implicit and stable, not an external predictor force. This is the
faithful replatform of Layer 0 onto Newton (doc §3.2: a domain-specialized module
inside the engine, replacing generic device-vs-mesh collision).

Proximal-end actuation (insertion = advance the kinematic base along the centerline
arc-length, staying in the lumen through curves; rotation = spin its quaternion)
matches the continuum action space (§1.2).
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp
import newton

from lumen.core.frame import CenterlineFrame
from lumen.newton.tube_vbd import TubeVBDSolver
from lumen.newton.forces import add_world_force, add_body_forces, actuate_bases
from lumen.newton.coupling import compose_radius_k, flow_drag_k


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
                 stentriever=None, aneurysm=None, flow_diverter=None, n_envs: int = 1,
                 vbd_iterations: int = 10, device: str | None = None,
                 catheter_points=None, catheter_radius: float = 0.65,
                 catheter_stretch_stiffness: float = 2.0e4,
                 catheter_bend_stiffness: float = 1.5e2,
                 couple_coaxial: bool = True, catheter_inner_radius: float = 0.5,
                 coax_kappa: float = 2.0e3, coax_d_hat: float = 0.1,
                 coax_two_way: bool = True, tree=None, route_centerline=None):
        from lumen.hardware import configure_backend_logging, detect_device
        configure_backend_logging()
        self.device = device or detect_device()      # cuda if available, else cpu
        self.R, self.kappa, self.d_hat = R, kappa, d_hat
        self.n_envs = int(n_envs)
        # L0d.2a — optional COAXIAL microcatheter: a second rod (larger radius, stiffer)
        # sharing the lumen with the guidewire. Single-env for now; flow/clot/stentriever
        # run through the host path, with aspiration keyed to the catheter tip and
        # flow-drag applied to the guidewire.
        self.coaxial = catheter_points is not None
        if self.coaxial:
            if len(catheter_points) < 2:          # fail fast, before _add_rod's opaque error
                raise ValueError("catheter_points needs >= 2 nodes (a rod centerline)")
            if self.n_envs != 1:
                raise NotImplementedError("coaxial assemblies are single-env (batched coaxial is future)")
        self.contact_frame = CenterlineFrame(vessel_centerline)
        # Batched envs share one vessel (the contact is wire-vs-wall, never wire-vs-wire,
        # so E rods in one model are independent). For n_envs>1 the wall/clot/flow co-sim
        # runs through on-device kernels (no per-substep host round-trip); n_envs==1 keeps
        # the original host path. The lumped Windkessel (NewtonFlow) is single-env only —
        # batched flow needs the 1-D FlowField.
        self._flow_is_field = flow is not None and hasattr(flow, "set_lumen")
        if self.n_envs > 1 and flow is not None and not self._flow_is_field:
            raise NotImplementedError(
                "batched flow requires the 1-D FlowField; the lumped NewtonFlow is "
                "single-env (analytic fallback)")
        if self.n_envs > 1 and stentriever is not None:
            raise NotImplementedError(
                "batched stent-retriever retrieval is not ported (per-env host force "
                "balance); run retrieval single-env")

        builder = newton.ModelBuilder(gravity=0.0)
        builder.default_shape_cfg.density = density

        def _add_rod(seed_pts, rod_radius, stretch, bend):
            pts = [wp.vec3(*map(float, p)) for p in seed_pts]
            quats = newton.utils.create_parallel_transport_cable_quaternions(pts)
            # Newton 1.4 interprets damping as an absolute physical coefficient; Lumen's
            # public knob remains a stiffness-relative multiplier for continuity with
            # the validated pre-1.4 behavior.
            bend_damping_abs = bend_damping * bend
            bodies, _ = builder.add_rod(pts, quats, radius=rod_radius, stretch_stiffness=stretch,
                                        bend_stiffness=bend, bend_damping=bend_damping_abs,
                                        body_frame_origin="com")
            builder.body_mass[bodies[0]] = 0.0       # kinematic base (proximal actuation)
            builder.body_inv_mass[bodies[0]] = 0.0
            builder.body_inertia[bodies[0]] = wp.mat33(0.0)
            builder.body_inv_inertia[bodies[0]] = wp.mat33(0.0)
            return bodies

        # guidewire (one rod per env). self.bodies stays the GUIDEWIRE so positions /
        # projection / flow are backward-compatible; the catheter rides _contact_bodies.
        self.bodies, self.bases = [], []
        for _ in range(self.n_envs):
            gw = _add_rod(device_points, radius, stretch_stiffness, bend_stiffness)
            self.bodies.extend(gw)
            self.bases.append(gw[0])
        self.n_per_env = len(self.bodies) // self.n_envs   # add_rod: N+1 points -> N bodies
        self.base = self.bases[0]
        # optional coaxial microcatheter (single-env): a second, larger, stiffer rod
        self.cath_bodies, self.cath_bases = [], []
        if self.coaxial:
            cath = _add_rod(catheter_points, catheter_radius,
                            catheter_stretch_stiffness, catheter_bend_stiffness)
            self.cath_bodies.extend(cath)
            self.cath_bases.append(cath[0])
        self._contact_bodies = self.bodies + self.cath_bodies   # both rods hit the wall
        if self.coaxial:
            # the guidewire and catheter are coupled by the radial constraint (L0d.2b),
            # NOT by capsule contact — disable Newton body-body collision between the two
            # rods so the catheter slides freely over the guidewire (else it drags it).
            shape_of: dict[int, list[int]] = {}
            for s, bod in enumerate(builder.shape_body):
                shape_of.setdefault(int(bod), []).append(s)
            gw_shapes = [s for bod in self.bodies for s in shape_of.get(bod, [])]
            cath_shapes = [s for bod in self.cath_bodies for s in shape_of.get(bod, [])]
            for sa in gw_shapes:
                for sb in cath_shapes:
                    builder.add_shape_collision_filter_pair(sa, sb)
        builder.color()
        self.model = builder.finalize(device=self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="SolverVBD damping behavior changed.*",
                                    category=UserWarning)
            self.solver = TubeVBDSolver(self.model, iterations=vbd_iterations)
        self.tree = tree
        if tree is not None:                          # multi-edge vascular tree (single-env)
            if self.n_envs != 1:
                raise NotImplementedError("tree contact is single-env (batched trees are future)")
            # the tree uses each edge's own lumen field for the base R0, so a sim-level
            # lumen_field doesn't apply; flow/clot project a single centerline. deformable_
            # wall + hgo_params ARE supported (per-edge HGO wall, L0d.1d).
            if lumen_field is not None:
                raise NotImplementedError("tree contact takes R0 from each edge's lumen "
                                          "field; a sim-level lumen_field doesn't apply")
            if flow is not None or clot_segment is not None:
                raise NotImplementedError(
                    "tree + flow/clot is not wired (flow drag / clot grids use a single "
                    "centerline, not the edge graph)")
            self.solver.set_tree_contact(tree, self._contact_bodies, kappa=kappa, d_hat=d_hat,
                                         barrier_mode=barrier_mode, mu_along=mu_along,
                                         mu_across=mu_across, gamma_fric_deg=gamma_fric_deg,
                                         actuation_centerline=route_centerline,
                                         deformable_wall=deformable_wall, hgo_params=hgo_params)
        else:
            # single-env coaxial passes n_per_env = ALL bodies so every body maps to env 0
            # (env = body_id // n_per_env); non-coaxial keeps the per-env guidewire blocking.
            n_per_env_contact = len(self._contact_bodies) if self.coaxial else self.n_per_env
            self.solver.set_tube_contact(vessel_centerline, R, self._contact_bodies,
                                         kappa=kappa, d_hat=d_hat,
                                         barrier_mode=barrier_mode,
                                         deformable_wall=deformable_wall,
                                         hgo_params=hgo_params, mu_along=mu_along,
                                         mu_across=mu_across, gamma_fric_deg=gamma_fric_deg,
                                         lumen_field=lumen_field,
                                         n_envs=self.n_envs, n_per_env=n_per_env_contact)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.body_ids = wp.array(np.array(self._contact_bodies, dtype=np.int32),  # forces on both rods
                                 dtype=wp.int32, device=self.device)
        self._guidewire_body_ids = wp.array(np.array(self.bodies, dtype=np.int32),
                                            dtype=wp.int32, device=self.device)
        # on-device base actuation (no per-substep body_q host round-trip), one per env
        self._base_ids = wp.array(np.array(self.bases, dtype=np.int32), dtype=wp.int32,
                                  device=self.device)
        self._ins_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
        self._tw_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
        if self.coaxial:                              # independent catheter proximal actuation
            self._cath_base_ids = wp.array(np.array(self.cath_bases, dtype=np.int32),
                                           dtype=wp.int32, device=self.device)
            self._cath_ins_arr = wp.zeros(1, dtype=wp.float32, device=self.device)
            self._cath_tw_arr = wp.zeros(1, dtype=wp.float32, device=self.device)
            if couple_coaxial:                        # gw rides inside the catheter lumen (L0d.2b)
                # forward the gw radius so the coupling keeps the gw SURFACE (not just its
                # centre) inside the catheter inner lumen (the barrier limit is
                # catheter_inner_radius − gw radius; the solver rejects an impossible fit).
                self.solver.set_coaxial_coupling(self.bodies, self.cath_bodies,
                                                 catheter_inner_radius, kappa=coax_kappa,
                                                 d_hat=coax_d_hat, two_way=coax_two_way,
                                                 gw_radius=radius)
        self.flow = flow                 # optional NewtonFlow (lumped) or FlowField (1-D)
        if self._flow_is_field:
            # Bind the FlowField to this sim's batch/device. Its device arrays are
            # sized to n_envs, so a FlowField must not be shared across sims with
            # different shapes — refuse to rebind an already-used or conflicting one.
            if getattr(flow, "_P_d", None) is not None:
                raise ValueError("this FlowField is already bound to a sim (device "
                                 "arrays allocated); use one FlowField per NewtonGuidewireSim")
            if getattr(flow, "n_envs", 1) not in (1, self.n_envs):
                raise ValueError(f"FlowField.n_envs ({flow.n_envs}) conflicts with sim "
                                 f"n_envs ({self.n_envs})")
            if getattr(flow, "device", "cpu") not in ("cpu", self.device):
                raise ValueError(f"FlowField.device ({flow.device}) conflicts with sim "
                                 f"device ({self.device})")
            flow.n_envs = self.n_envs
            flow.device = self.device
        # optional finite-extent deformable clot (shares the wall's s,θ grid)
        self.clot = None
        if clot_segment is not None:
            from lumen.newton.clot import ClotField
            w = self.solver._wall
            self.clot = ClotField(s_max=w.s_max, n_s=w.n_s, n_th=w.n_th, R_base=R,
                                  s0=clot_segment[0], s1=clot_segment[1],
                                  height=clot_height, params=clot_params,
                                  n_envs=self.n_envs, device=self.device)
        self.stentriever = stentriever   # optional device for clot retrieval
        # optional saccular aneurysm + flow diverter. Needs the 1-D FlowField, which
        # supplies the pressure P(s_neck) that drives each sac. Batched sims keep one
        # AneurysmSac per env and read the matching env's pressure block from the
        # batched FlowField snapshot after every device-side flow solve.
        self.aneurysm = aneurysm
        self.flow_diverter = flow_diverter
        self.aneurysm_sac = None
        if aneurysm is not None:
            if not self._flow_is_field:
                raise NotImplementedError("an aneurysm needs the 1-D FlowField (it reads "
                                          "the neck pressure P(s)); pass flow=FlowField(...)")
            assert flow is not None
            s_max = self.solver._wall.s_max
            from lumen.newton.aneurysm import AneurysmSac
            self.aneurysms = self._per_env_objects(aneurysm, "aneurysm")
            self.flow_diverters = self._per_env_objects(flow_diverter, "flow_diverter",
                                                        allow_none=True)
            for an in self.aneurysms:
                if not (0.0 <= an.s_neck <= s_max):    # else np.interp silently clamps
                    raise ValueError(f"aneurysm s_neck ({an.s_neck}) is outside the "
                                     f"vessel arc-length [0, {s_max:.1f}]")
            self.aneurysm_sacs = [AneurysmSac(an, visc=flow.p.visc) for an in self.aneurysms]
            self.aneurysm_sac = self.aneurysm_sacs[0]  # backward-compatible single-env handle
        elif flow_diverter is not None:
            raise ValueError("flow_diverter set without an aneurysm to divert from")
        else:
            self.aneurysms = []
            self.flow_diverters = []
            self.aneurysm_sacs = []
        # batched on-device coupling scratch (n_envs>1): node drag arrays + zero occ
        self._use_device_coupling = self.n_envs > 1 and (self.clot is not None
                                                         or self._flow_is_field)
        if self._use_device_coupling:
            w = self.solver._wall
            n_node = self.n_per_env * self.n_envs
            self._snodes_d = wp.zeros(n_node, dtype=wp.float32, device=self.device)
            self._tang_d = wp.zeros(n_node, dtype=wp.vec3, device=self.device)
            self._zero_occ_d = wp.zeros(self.n_envs * w.n_s, dtype=wp.float32,
                                        device=self.device)
        # snapshot for fast reset (avoid rebuilding the model/solver each episode)
        self._init_body_q = self.state_0.body_q.numpy().copy()

    def _per_env_objects(self, value, name: str, allow_none: bool = False):
        """Return one object per env; scalars broadcast, explicit sequences must match.

        The objects are config/state holders, not numeric arrays. Strings are treated as
        scalars for the generic helper even though current callers pass dataclasses.
        """
        if value is None:
            if allow_none:
                return [None] * self.n_envs
            raise ValueError(f"{name} is required")
        if isinstance(value, (list, tuple)):
            if len(value) != self.n_envs:
                raise ValueError(f"{name} length ({len(value)}) must match n_envs ({self.n_envs})")
            if not allow_none and any(v is None for v in value):
                raise ValueError(f"{name} entries cannot be None")
            return list(value)
        return [value for _ in range(self.n_envs)]

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
        tw = getattr(self.solver, "_tree_wall", None)        # tree path: clear wall deformation + load
        if tw is not None:
            tw.w[:] = 0.0
            tw.w_field.zero_()
            tw.wall_load.zero_()
        if self.clot is not None:
            self.clot.o = self.clot.o0.copy()
            self.clot.D[:] = 0.0
        for sac in self.aneurysm_sacs:
            sac.reset()

    def _actuate(self, base_ids, ins_arr, tw_arr, insertion, twist, n):
        """Centerline-following insertion/twist of `n` kinematic bases (one per env, or
        the single catheter base). Scalars broadcast; arrays are per-env RL actions."""
        ins = np.broadcast_to(np.asarray(insertion, np.float32), (n,))
        tw = np.broadcast_to(np.asarray(twist, np.float32), (n,))
        if not ins.any() and not tw.any():
            return
        ins_arr.assign(np.ascontiguousarray(ins))
        tw_arr.assign(np.ascontiguousarray(tw))
        sv = self.solver
        wp.launch(actuate_bases, dim=n,
                  inputs=[base_ids, ins_arr, tw_arr,
                          sv._tube_P, sv._tube_Tg, sv._tube_cum_s, sv._tube_M, sv._tube_s_max],
                  outputs=[self.state_0.body_q], device=self.device)

    def _actuate_base(self, insertion, twist):
        # translate/rotate each guidewire kinematic base along the centerline (#23)
        self._actuate(self._base_ids, self._ins_arr, self._tw_arr, insertion, twist, self.n_envs)

    def _actuate_catheter(self, insertion, twist):
        self._actuate(self._cath_base_ids, self._cath_ins_arr, self._cath_tw_arr,
                      insertion, twist, 1)

    def step(self, dt: float = 2.5e-2, substeps: int = 5,
             insertion: float = 0.0, twist: float = 0.0, preload=(0.0, 0.0, 0.0),
             aspiration: float = 0.0, insertion_cath: float = 0.0, twist_cath: float = 0.0):
        """Advance the simulation by `dt` total, as `substeps` sub-steps of
        `dt/substeps` each (the standard substep convention).

        `insertion_cath`/`twist_cath` independently actuate the coaxial microcatheter
        (ignored when there is no catheter)."""
        sub_dt = dt / substeps
        if self.flow is not None and aspiration:
            self.flow.aspiration = aspiration        # aspiration recovers downstream flow
        if self._use_device_coupling:                # n_envs>1 with clot/flow: on-device
            self._step_device(sub_dt, substeps, insertion, twist, preload)
            return
        self._step_host(sub_dt, substeps, insertion, twist, preload, aspiration, dt,
                        insertion_cath, twist_cath)

    def _step_device(self, sub_dt, substeps, insertion, twist, preload):
        """Batched per-substep co-sim entirely on device (no host round-trip):
        compose R0 (pulse − clot occlusion) -> flow solve -> local drag -> contact
        solve -> clot update. Each env reads/writes its own wall/clot/flow block."""
        wall = self.solver._wall
        n_s, n_th, s_max, ncell = wall.n_s, wall.n_th, wall.s_max, wall.n_cells
        field_flow = self._flow_is_field
        if field_flow:                               # per-step host prep for local drag
            pos = self.state_0.body_q.numpy()[self.bodies, :3].reshape(
                self.n_envs, self.n_per_env, 3)
            tang = np.zeros_like(pos)
            tang[:, 1:-1] = pos[:, 2:] - pos[:, :-2]
            tang[:, 0] = pos[:, 1] - pos[:, 0]
            tang[:, -1] = pos[:, -1] - pos[:, -2]
            tang /= (np.linalg.norm(tang, axis=2, keepdims=True) + 1e-12)
            s_nodes = self.contact_frame.project_s(pos.reshape(-1, 3))   # vectorized
            self._tang_d.assign(np.ascontiguousarray(tang.reshape(-1, 3).astype(np.float32)))
            self._snodes_d.assign(s_nodes.astype(np.float32))
            self.flow.set_tips(s_nodes, s_max, n_s)
        occ_arr = self.clot.o_d if self.clot is not None else self._zero_occ_d
        for _ in range(substeps):
            self.state_0.clear_forces()
            self._actuate_base(insertion / substeps, twist / substeps)   # batched: no coaxial
            if any(preload):
                wp.launch(add_world_force, dim=self.body_ids.shape[0],
                          inputs=[self.body_ids, float(preload[0]), float(preload[1]),
                                  float(preload[2]), 1],
                          outputs=[self.state_0.body_f], device=self.device)
            pulse = self.flow.pulse_factor() if self.flow is not None else 1.0
            wp.launch(compose_radius_k, dim=self.n_envs * ncell,
                      inputs=[wall._R0_base_d, occ_arr, float(pulse), n_s, n_th],
                      outputs=[wall.r0_field, wall.clot_mask_field], device=self.device)
            if field_flow:
                self.flow.advance(sub_dt)
                self.flow.solve_device(wall.r0_field, n_s, n_th, s_max)
                if self.aneurysm_sacs:
                    # Synchronize the just-solved batched pressure field to host and
                    # update each env's independent 0-D sac from its own P(s_neck) and
                    # flow-diverter coverage. This preserves the device-side FlowField
                    # solve while avoiding cross-env host state bleed-through.
                    P_env = self.flow._P_d.numpy().reshape(self.n_envs, n_s)
                    s_grid = np.linspace(0.0, s_max, n_s)
                    for env, (an, sac, fd) in enumerate(zip(self.aneurysms,
                                                           self.aneurysm_sacs,
                                                           self.flow_diverters)):
                        P_neck = float(np.interp(an.s_neck, s_grid, P_env[env]))
                        div = fd.diversion(an) if fd is not None else 0.0
                        sac.update(P_neck, sub_dt, diversion=div)
                wp.launch(flow_drag_k, dim=self.n_envs * self.n_per_env,
                          inputs=[self._snodes_d, self._tang_d, self.body_ids,
                                  self.flow._v_d, self.n_per_env, n_s, float(s_max),
                                  float(self.flow.p.drag_coeff)],
                          outputs=[self.state_0.body_f], device=self.device)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            if self.clot is not None:
                self.clot.update_device(wall.wall_load, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _step_host(self, sub_dt, substeps, insertion, twist, preload, aspiration, dt,
                   insertion_cath=0.0, twist_cath=0.0):
        """Single-env (n_envs==1) path: the original host-side co-sim, unchanged."""
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
                if self.coaxial and len(self.cath_bodies):
                    cath_pos = self.state_0.body_q.numpy()[self.cath_bodies, :3]
                    cath_s = np.array([self.contact_frame.project(p).s for p in cath_pos])
                    self.flow.set_tip(float(cath_s.max()))   # aspiration point = catheter tip
                else:
                    self.flow.set_tip(float(s_nodes.max()))   # no catheter: deepest wire node
        for _ in range(substeps):
            self.state_0.clear_forces()
            self._actuate_base(insertion / substeps, twist / substeps)
            if self.coaxial:                          # independent microcatheter actuation
                self._actuate_catheter(insertion_cath / substeps, twist_cath / substeps)
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
                    if self.aneurysm_sac is not None:    # drive the sac from P(s_neck)
                        an = self.aneurysms[0]
                        fd = self.flow_diverters[0]
                        P_neck = float(np.interp(an.s_neck, self.flow.s_grid,
                                                 self.flow.pressure_field()))
                        div = fd.diversion(an) if fd is not None else 0.0
                        self.aneurysm_sac.update(P_neck, sub_dt, diversion=div)
                    drag = self.flow.drag_at(s_nodes)[:, None]
                else:
                    drag = self.flow.drag_per_unit_tangent()
                dvecs = wp.array((drag * tang).astype(np.float32),
                                 dtype=wp.vec3, device=self.device)
                wp.launch(add_body_forces, dim=self._guidewire_body_ids.shape[0],
                          inputs=[self._guidewire_body_ids, dvecs, 1],
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
        return self.state_0.body_q.numpy()[self.bodies, :3]      # the GUIDEWIRE (backward-compat)

    def catheter_positions(self) -> np.ndarray:
        """Microcatheter node positions (empty if no coaxial catheter)."""
        return self.state_0.body_q.numpy()[self.cath_bodies, :3]

    def env_positions(self) -> np.ndarray:
        """Device-node positions per env, shape (n_envs, n_per_env, 3)."""
        return self.body_positions().reshape(self.n_envs, self.n_per_env, 3)

    def node_radii(self) -> np.ndarray:
        # for a tree, radius is measured against the nearest edge (junction-aware)
        proj = self.tree.project if self.tree is not None else self.contact_frame.project
        return np.array([proj(p).r for p in self.body_positions()])

    def catheter_node_radii(self) -> np.ndarray:
        proj = self.tree.project if self.tree is not None else self.contact_frame.project
        return np.array([proj(p).r for p in self.catheter_positions()])

    def wall_max_deflection(self) -> float:
        return self.solver.wall_max_deflection()      # tree-aware (per-edge HGO wall, L0d.1d)

    # --- aneurysm flow-diversion outputs (None if no aneurysm) ----------------
    # inflow/turnover are CUMULATIVE over the current window (since reset/sac_mark);
    # diversion is the CURRENT (static) neck coverage. Call sac_mark() at deployment
    # to isolate the post-deployment inflow/turnover from the pre-deployment phase.
    def sac_inflow_peak(self):
        """Peak neck inflow jet over the current window (a flow diverter lowers it)."""
        if not self.aneurysm_sacs:
            return 0.0
        vals = np.array([sac.inflow_peak() for sac in self.aneurysm_sacs], dtype=float)
        return float(vals[0]) if self.n_envs == 1 else vals

    def sac_turnover_time(self):
        """Sac washout time over the current window (a diverter lengthens it -> stasis)."""
        if not self.aneurysm_sacs:
            return float("inf")
        vals = np.array([sac.turnover_time() for sac in self.aneurysm_sacs], dtype=float)
        return float(vals[0]) if self.n_envs == 1 else vals

    def sac_diversion(self):
        """Current effective neck coverage of the deployed flow diverter (0 if none/missed)."""
        if not self.aneurysm_sacs:
            return 0.0
        vals = np.array([sac.last_diversion for sac in self.aneurysm_sacs], dtype=float)
        return float(vals[0]) if self.n_envs == 1 else vals

    def sac_mark(self) -> None:
        """Open a fresh aneurysm measurement window (call at flow-diverter deployment),
        keeping the sac-pressure equilibrium — so post-deployment stasis is isolated."""
        for sac in self.aneurysm_sacs:
            sac.mark_window()
