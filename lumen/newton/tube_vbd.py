"""Thin fork of Newton's SolverVBD: tube-intrinsic contact in the AVBD solve.

Instead of vendoring the entire ~2900-line SolverVBD, TubeVBDSolver SUBCLASSES the
upstream solver and overrides ONLY ``_solve_rigid_body_iteration`` to inject the
tube-intrinsic barrier (force + Hessian) into the per-color rigid solve. Every
other method — and every upstream bug fix — is inherited (#11). The overridden
method is adapted from newton/_src/solvers/vbd/solver_vbd.py (Apache-2.0,
(c) 2025 The Newton Developers); modifications (c) 2026 Seldinger. If Newton
changes that one method upstream, re-sync this single method.
"""

from __future__ import annotations

import numpy as _np
import warp as wp
import warp as _wp

from newton.solvers import SolverVBD
from newton import State, Control, Contacts        # L3: used in method annotations
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    accumulate_body_body_contacts_per_body,
    accumulate_body_particle_contacts_per_body,
    solve_rigid_body,
    update_duals_body_body_contacts,
    update_duals_body_particle_contacts,
    update_duals_joint,
)

from lumen.newton.tube_barrier_kernel import (accumulate_coaxial_coupling,
                                              accumulate_tree_barrier, accumulate_tube_barrier)


class TubeVBDSolver(SolverVBD):
    """SolverVBD with the tube-intrinsic barrier injected into the AVBD solve."""

    def _solve_rigid_body_iteration(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float
    ):
        """Solve one AVBD iteration for rigid bodies (per-iteration phase).

        Accumulates contact and joint forces/hessians, solves 6x6 rigid body systems per color,
        and updates AVBD penalty parameters (dual update).
        """
        model = self.model

        # Body-particle soft contacts still need penalty updates when VBD skips rigid solves:
        # external rigid mode uses state_out.body_q, while static-shape contacts use _empty_body_q.
        skip_rigid_solve = self.integrate_with_external_rigid_solver or model.body_count == 0
        if skip_rigid_solve:
            if model.particle_count > 0 and contacts is not None:
                body_q = state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q
                if body_q is None:
                    body_q = self._empty_body_q

                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        body_q,
                        self.body_particle_contact_material_ke,
                        self.rigid_linear_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )
            return

        # Zero out forces and hessians
        self.body_torques.zero_()
        self.body_forces.zero_()
        self.body_hessian_aa.zero_()
        self.body_hessian_al.zero_()
        self.body_hessian_ll.zero_()
        if getattr(self, "_tube_enabled", False):
            self._tube_wall_load.zero_()      # holds this iteration's wall contact load
        if getattr(self, "_tree_enabled", False):
            self._tree_wall_load.zero_()

        body_color_groups = model.body_color_groups

        # Gauss-Seidel-style per-color updates
        for color in range(len(body_color_groups)):
            color_group = body_color_groups[color]

            # Accumulate body-particle contact forces/hessians for bodies in this color
            if model.particle_count > 0 and contacts is not None:
                wp.launch(
                    kernel=accumulate_body_particle_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        state_in.particle_q,
                        self.particle_q_prev,
                        model.particle_radius,
                        self.body_q_prev,
                        state_in.body_q,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.friction_epsilon,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_ke,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        self.body_particle_contact_buffer_pre_alloc,
                        self.body_particle_contact_counts,
                        self.body_particle_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            # Accumulate body-body (rigid-rigid) contact forces and Hessians on bodies (per-body, per-color)
            if contacts is not None:
                wp.launch(
                    kernel=accumulate_body_body_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        self.body_q_prev,
                        state_in.body_q,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.friction_epsilon,
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_material_ke,
                        self.body_body_contact_material_kd,
                        self.body_body_contact_material_mu,
                        self.body_body_contact_lambda,
                        self.body_body_contact_C0,
                        self.rigid_contact_alpha,
                        self.rigid_contact_hard,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        self.body_body_contact_buffer_pre_alloc,
                        self.body_body_contact_counts,
                        self.body_body_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            if getattr(self, "_tube_enabled", False):
                wp.launch(
                    kernel=accumulate_tube_barrier,
                    dim=color_group.size,
                    inputs=[color_group, self._tube_wire_mask, state_in.body_q,
                            state_in.body_qd,
                            self._tube_P, self._tube_Tg, self._tube_M1,
                            self._tube_cum_s, self._tube_M,
                            self._wall.r0_field, self._tube_s_max, self._tube_ns,
                            self._tube_nth, self._tube_n_per_env, self._wall.w_field,
                            self._tube_kappa, self._tube_d_hat, self._tube_mode,
                            self._tube_mu_along, self._tube_mu_across,
                            self._tube_gamma_fric, dt],
                    outputs=[self.body_forces, self.body_hessian_ll,
                             self._tube_wall_load],
                    device=self.device,
                )
            if getattr(self, "_tree_enabled", False):
                wp.launch(
                    kernel=accumulate_tree_barrier,
                    dim=color_group.size,
                    inputs=[color_group, self._tree_wire_mask, state_in.body_q,
                            state_in.body_qd,
                            self._tree_P, self._tree_Tg, self._tree_M1, self._tree_cum_s,
                            self._tree_vstart, self._tree_vcount, self._tree_smax,
                            self._tree_start_junc, self._tree_end_junc, self._tree_n_edges,
                            self._tree_R0, self._tree_w, self._tree_ns, self._tree_nth,
                            self._tree_kappa, self._tree_d_hat, self._tree_mode,
                            self._tree_mu_along, self._tree_mu_across,
                            self._tree_gamma_fric, dt],
                    outputs=[self.body_forces, self.body_hessian_ll, self._tree_wall_load],
                    device=self.device,
                )
            if getattr(self, "_coax_enabled", False):
                wp.launch(
                    kernel=accumulate_coaxial_coupling,
                    dim=color_group.size,
                    inputs=[color_group, self._coax_gw_mask, state_in.body_q,
                            self._coax_cath_ids, self._coax_n_cath, self._coax_r_inner,
                            self._coax_kappa, self._coax_d_hat, self._coax_two_way],
                    outputs=[self.body_forces, self.body_hessian_ll],
                    device=self.device,
                )
            wp.launch(
                kernel=solve_rigid_body,
                inputs=[
                    dt,
                    color_group,
                    state_in.body_q,
                    self.body_q_prev,
                    model.body_q,
                    model.body_mass,
                    self.body_inv_mass_effective,
                    model.body_inertia,
                    self.body_inertia_q,
                    model.body_com,
                    self.rigid_adjacency,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_penalty_kd,
                    self.joint_sigma_start,
                    self.joint_C_fric,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    self.joint_lambda_lin,
                    self.joint_lambda_ang,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    model.joint_dof_dim,
                    self.joint_rest_angle,
                    self.body_forces,
                    self.body_torques,
                    self.body_hessian_ll,
                    self.body_hessian_al,
                    self.body_hessian_aa,
                ],
                outputs=[
                    state_in.body_q,
                ],
                dim=color_group.size,
                device=self.device,
            )

        if contacts is not None:
            contact_launch_dim = contacts.rigid_contact_max
            wp.launch(
                kernel=update_duals_body_body_contacts,
                dim=contact_launch_dim,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    model.shape_body,
                    state_in.body_q,
                    self.body_q_prev,
                    self.body_body_contact_material_mu,
                    self.body_body_contact_C0,
                    self.rigid_contact_alpha,
                    self.rigid_contact_stick_motion_eps,
                    self.rigid_contact_hard,
                    self.body_inv_mass_effective,
                    self.body_body_contact_material_ke,
                    self.rigid_linear_beta,
                    self.body_body_contact_penalty_k,  # input/output
                    self.body_body_contact_lambda,  # input/output
                ],
                outputs=[
                    self.body_body_contact_stick_flag,
                ],
                device=self.device,
            )

            if model.particle_count > 0:
                soft_contact_launch_dim = contacts.soft_contact_max
                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=soft_contact_launch_dim,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        state_in.body_q,
                        self.body_particle_contact_material_ke,
                        self.rigid_linear_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )

        if model.joint_count > 0:
            wp.launch(
                kernel=update_duals_joint,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    self.joint_constraint_start,
                    state_in.body_q,
                    model.body_q,
                    model.joint_dof_dim,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    self.joint_penalty_k_max,
                    self.rigid_linear_beta,
                    self.rigid_angular_beta,
                    model.joint_target_ke,
                    control.joint_target_q,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    self.joint_rest_angle,
                    self.joint_penalty_k,  # input/output
                    self.joint_lambda_lin,  # input/output
                    self.joint_lambda_ang,  # input/output
                ],
                device=self.device,
            )

    def set_tube_contact(self, centerline, R, wire_body_ids, kappa=2.0e3, d_hat=0.3,
                         barrier_mode="compliant", deformable_wall=False,
                         hgo_params=None, n_s=40, n_th=16,
                         mu_along=0.0, mu_across=0.0, gamma_fric_deg=40.0,
                         lumen_field=None, n_envs=1, n_per_env=None):
        """barrier_mode: 'compliant' (fast tier) | 'log' (bounded IPC option).

        R is the base lumen radius: a scalar (cylinder) OR, via ``lumen_field`` (a
        ``lumen.core.LumenField``), the true R(s,θ) anatomy (stenosis/aneurysm/
        patient). The contact reads R0(s,θ)+w(s,θ) — the shared field of §3.5.6.
        deformable_wall=True activates the HGO wall; False = rigid base R0(s,θ).
        """
        from lumen.core.frame import CenterlineFrame
        from lumen.newton.hgo_wall import WallField
        f = CenterlineFrame(_np.asarray(centerline))
        dev = self.device
        self._tube_P = _wp.array(f.points.astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tube_Tg = _wp.array(f.tangents.astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tube_M1 = _wp.array(f.m1.astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tube_cum_s = _wp.array(f.cum_s.astype(_np.float32), dtype=_wp.float32, device=dev)
        self._tube_M = len(f.points)
        s_max = float(f.length)
        self._tube_s_max = s_max
        self._tube_ns = int(n_s)
        self._tube_nth = int(n_th)
        self._tube_kappa = float(kappa)
        self._tube_d_hat = float(d_hat)
        self._tube_mode = 1 if barrier_mode == "log" else 0
        self._tube_mu_along = float(mu_along)
        self._tube_mu_across = float(mu_across)
        self._tube_gamma_fric = float(_np.radians(gamma_fric_deg))
        self._tube_deformable = bool(deformable_wall)
        # base radius grid R0(s,θ) on the wall cells (cell = i_s*n_th + i_th)
        if lumen_field is None:
            R0_grid = float(R)
        else:
            ss = _np.linspace(0.0, s_max, n_s)
            th = -_np.pi + (_np.arange(n_th) + 0.5) / n_th * 2.0 * _np.pi
            R0_grid = _np.array([[lumen_field.eval(float(s), float(t)) for t in th]
                                 for s in ss]).ravel()
        self._tube_n_envs = int(n_envs)
        self._tube_n_per_env = int(n_per_env if n_per_env is not None
                                   else len(wire_body_ids) // n_envs)
        self._wall = WallField(R0=R0_grid, s_max=s_max, n_s=n_s, n_th=n_th,
                               params=hgo_params, device=dev, n_envs=n_envs)
        self._tube_wall_load = self._wall.wall_load
        mask = _np.zeros(self.model.body_count, dtype=_np.int32)
        mask[_np.asarray(wire_body_ids, dtype=_np.int32)] = 1
        self._tube_wire_mask = _wp.array(mask, dtype=_wp.int32, device=dev)
        self._tube_enabled = True

    def set_coaxial_coupling(self, gw_body_ids, cath_body_ids, r_inner,
                             kappa=2.0e3, d_hat=0.3, two_way=True, gw_radius=0.0):
        """Constrain the guidewire to ride inside the catheter's inner lumen (radius
        `r_inner`), reading the catheter's LIVE centerline each AVBD iteration so the gw
        follows the catheter as it bends, sliding freely axially (L0d.2b). `two_way`
        (L0d.2d) deposits the equal-opposite reaction on the catheter (responsive).

        `gw_radius` keeps the guidewire SURFACE (centre + radius), not just its centre,
        inside the lumen: the barrier limit is r_inner − gw_radius, and the band d_hat
        is clamped to half that clearance (so it stays a near-wall barrier, not a
        constant central pull)."""
        if len(cath_body_ids) < 2:
            raise ValueError("coaxial coupling needs a catheter centerline of >= 2 nodes")
        dev = self.device
        mask = _np.zeros(self.model.body_count, dtype=_np.int32)
        mask[_np.asarray(gw_body_ids, dtype=_np.int32)] = 1
        self._coax_gw_mask = _wp.array(mask, dtype=_wp.int32, device=dev)
        self._coax_cath_ids = _wp.array(_np.asarray(cath_body_ids, _np.int32),
                                        dtype=_wp.int32, device=dev)
        self._coax_n_cath = len(cath_body_ids)
        if float(gw_radius) >= float(r_inner):       # impossible fit -> fail fast, not near-singular
            raise ValueError(f"guidewire radius ({gw_radius}) must be < catheter inner "
                             f"radius ({r_inner}); the guidewire can't fit inside")
        r_eff = float(r_inner) - float(gw_radius)     # gw SURFACE stays inside the lumen
        self._coax_r_inner = r_eff
        self._coax_kappa = float(kappa)
        self._coax_d_hat = min(float(d_hat), 0.5 * r_eff)        # proper near-wall band
        self._coax_two_way = 1 if two_way else 0
        self._coax_enabled = True
    def set_tree_contact(self, tree, wire_body_ids, kappa=2.0e3, d_hat=0.3,
                         barrier_mode="compliant", n_s=40, n_th=16,
                         mu_along=0.0, mu_across=0.0, gamma_fric_deg=40.0,
                         actuation_centerline=None, deformable_wall=False, hgo_params=None):
        """Multi-edge (vascular-tree) contact: each wire node contacts its nearest edge,
        with R branch-blended across junctions (the §3.5.2 work, pre-baked into the grid
        here so the kernel stays simple). Single-env. `deformable_wall=True` gives each
        edge an HGO wall (w field shared with the barrier, like the single tube, §3.5.6),
        with each edge's OWN arc-length feeding its cell area (correct for unequal-length
        edges). `tree` is a ``lumen.core.VascularTree``.

        `actuation_centerline` is the path the kinematic base follows for insertion
        (centerline-following). Pass the full route polyline (trunk→target branch) so the
        base can be pushed PAST the junction into a branch; default = the entry edge (the
        base stops at the apex — only trunk targets reachable)."""
        if getattr(self, "_tube_enabled", False):     # one contact model per solver, never both
            raise RuntimeError("tube contact already set on this solver; cannot also enable tree "
                               "contact (the barriers would double up)")
        dev = self.device
        P, Tg, M1, cum_s = [], [], [], []
        vstart, vcount, smax, sj, ej = [], [], [], [], []
        ss = _np.linspace(0.0, 1.0, n_s)              # fractional s; scaled per edge below
        th = -_np.pi + (_np.arange(n_th) + 0.5) / n_th * 2.0 * _np.pi
        R0_blocks = []
        off = 0
        for i, e in enumerate(tree.edges):
            f = e.frame
            vstart.append(off)
            vcount.append(len(f.points))
            off += len(f.points)
            smax.append(float(f.length))
            sj.append(1 if tree.is_junction(e.node_a) else 0)
            ej.append(1 if tree.is_junction(e.node_b) else 0)
            P.append(f.points); Tg.append(f.tangents); M1.append(f.m1); cum_s.append(f.cum_s)
            # bake the branch-blended R(s,θ) for this edge into its grid block
            block = _np.array([[tree.blended_R(i, float(sf) * f.length, float(t)) for t in th]
                               for sf in ss])
            R0_blocks.append(block.ravel())
        self._tree_P = _wp.array(_np.concatenate(P).astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tree_Tg = _wp.array(_np.concatenate(Tg).astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tree_M1 = _wp.array(_np.concatenate(M1).astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tree_cum_s = _wp.array(_np.concatenate(cum_s).astype(_np.float32), dtype=_wp.float32, device=dev)
        self._tree_vstart = _wp.array(_np.array(vstart, _np.int32), dtype=_wp.int32, device=dev)
        self._tree_vcount = _wp.array(_np.array(vcount, _np.int32), dtype=_wp.int32, device=dev)
        self._tree_smax = _wp.array(_np.array(smax, _np.float32), dtype=_wp.float32, device=dev)
        self._tree_start_junc = _wp.array(_np.array(sj, _np.int32), dtype=_wp.int32, device=dev)
        self._tree_end_junc = _wp.array(_np.array(ej, _np.int32), dtype=_wp.int32, device=dev)
        self._tree_n_edges = len(tree.edges)
        # one HGO wall over all edges (each edge is a block, like a batched env). r0_field
        # is the blended base R0; w_field is the shared deformation the barrier reads and
        # the contact load relaxes. rigid (deformable_wall=False) just leaves w≡0.
        from lumen.newton.hgo_wall import WallField
        self._tree_wall = WallField(R0=_np.concatenate(R0_blocks).astype(_np.float32),
                                    s_max=_np.asarray(smax, float),   # per-edge arc-length (H1 fix)
                                    n_s=n_s, n_th=n_th, params=hgo_params,
                                    device=dev, n_envs=self._tree_n_edges)
        self._tree_R0 = self._tree_wall.r0_field
        self._tree_w = self._tree_wall.w_field
        self._tree_wall_load = self._tree_wall.wall_load
        self._tree_deformable = bool(deformable_wall)
        self._tree_ns, self._tree_nth = int(n_s), int(n_th)
        self._tree_kappa, self._tree_d_hat = float(kappa), float(d_hat)
        self._tree_mode = 1 if barrier_mode == "log" else 0
        self._tree_mu_along, self._tree_mu_across = float(mu_along), float(mu_across)
        self._tree_gamma_fric = float(_np.radians(gamma_fric_deg))
        mask = _np.zeros(self.model.body_count, dtype=_np.int32)
        mask[_np.asarray(wire_body_ids, dtype=_np.int32)] = 1
        self._tree_wire_mask = _wp.array(mask, dtype=_wp.int32, device=dev)
        self._tree_enabled = True
        # base actuation (centerline-following insertion) follows the route polyline if
        # given (so the base can be pushed past a junction into a branch), else the entry
        # edge. These _tube_* arrays feed _actuate_base ONLY — the tube CONTACT kernel
        # stays off (only _tree_enabled drives contact).
        if actuation_centerline is not None:
            from lumen.core.frame import CenterlineFrame
            fa = CenterlineFrame(_np.asarray(actuation_centerline, float))
        else:
            fa = tree.edges[0].frame
        self._tube_P = _wp.array(fa.points.astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tube_Tg = _wp.array(fa.tangents.astype(_np.float32), dtype=_wp.vec3, device=dev)
        self._tube_cum_s = _wp.array(fa.cum_s.astype(_np.float32), dtype=_wp.float32, device=dev)
        self._tube_M = len(fa.points)
        self._tube_s_max = float(fa.length)

    def step(self, state_in, state_out, control, contacts, dt):
        super().step(state_in, state_out, control, contacts, dt)
        # staggered HGO co-sim: relax the shared-R wall to the contact load it just saw
        if getattr(self, "_tube_enabled", False) and self._tube_deformable:
            self._wall.update_from_load()
        if getattr(self, "_tree_enabled", False) and getattr(self, "_tree_deformable", False):
            self._tree_wall.update_from_load()      # per-edge HGO relaxation

    def wall_max_deflection(self):
        if getattr(self, "_tree_enabled", False):
            return self._tree_wall.max_deflection()
        return self._wall.max_deflection() if getattr(self, "_wall", None) else 0.0
