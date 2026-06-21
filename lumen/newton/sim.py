"""Newton-based guidewire simulation with tube-intrinsic contact (doc §3.2).

The guidewire is a Newton ``add_rod`` cable (stretch + bend/twist), integrated by
the forked ``TubeVBDSolver`` (``lumen.newton.vbd_fork``) which injects the
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
from lumen.newton.vbd_fork import TubeVBDSolver
from lumen.newton.tube_contact import add_world_force


class NewtonGuidewireSim:
    def __init__(self, vessel_centerline: np.ndarray, R: float,
                 device_points: np.ndarray, radius: float = 0.2,
                 stretch_stiffness: float = 1.0e4, bend_stiffness: float = 5.0e1,
                 bend_damping: float = 1.0, density: float = 1.0,
                 kappa: float = 2.0e3, d_hat: float = 0.3,
                 barrier_mode: str = "compliant",
                 deformable_wall: bool = False, hgo_params=None,
                 mu_along: float = 0.0, mu_across: float = 0.0,
                 gamma_fric_deg: float = 40.0,
                 vbd_iterations: int = 10, device: str | None = None):
        self.device = device or ("cuda" if wp.get_cuda_device_count() > 0 else "cpu")
        self.R, self.kappa, self.d_hat = R, kappa, d_hat
        self.contact_frame = CenterlineFrame(vessel_centerline)
        self._fwd = self.contact_frame.tangents[0].astype(np.float32)

        builder = newton.ModelBuilder(gravity=0.0)
        builder.default_shape_cfg.density = density
        pts = [wp.vec3(*map(float, p)) for p in device_points]
        quats = newton.utils.create_parallel_transport_cable_quaternions(pts)
        bodies, joints = builder.add_rod(
            pts, quats, radius=radius, stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness, bend_damping=bend_damping,
            body_frame_origin="com")
        self.bodies = bodies
        self.base = bodies[0]
        builder.body_mass[self.base] = 0.0
        builder.body_inv_mass[self.base] = 0.0
        builder.body_inertia[self.base] = wp.mat33(0.0)
        builder.body_inv_inertia[self.base] = wp.mat33(0.0)
        builder.color()
        self.model = builder.finalize(device=self.device)

        self.solver = TubeVBDSolver(self.model, iterations=vbd_iterations)
        self.solver.set_tube_contact(vessel_centerline, R, bodies,
                                     kappa=kappa, d_hat=d_hat,
                                     barrier_mode=barrier_mode,
                                     deformable_wall=deformable_wall,
                                     hgo_params=hgo_params, mu_along=mu_along,
                                     mu_across=mu_across, gamma_fric_deg=gamma_fric_deg)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.body_ids = wp.array(np.array(bodies, dtype=np.int32), dtype=wp.int32,
                                 device=self.device)

    def _actuate_base(self, insertion: float, twist: float):
        if insertion == 0.0 and twist == 0.0:
            return
        q = self.state_0.body_q.numpy()
        T = q[self.base]
        pos = T[:3] + insertion * self._fwd
        rot = T[3:7]
        if twist != 0.0:
            ax = self._fwd / (np.linalg.norm(self._fwd) + 1e-12)
            s = np.sin(twist / 2)
            dq = np.array([ax[0] * s, ax[1] * s, ax[2] * s, np.cos(twist / 2)])
            rot = _quat_mul(dq, rot)
        q[self.base] = np.concatenate([pos, rot])
        self.state_0.body_q = wp.array(q, dtype=wp.transform, device=self.device)

    def step(self, dt: float = 5.0e-3, substeps: int = 5,
             insertion: float = 0.0, twist: float = 0.0, preload=(0.0, 0.0, 0.0)):
        for _ in range(substeps):
            self.state_0.clear_forces()
            self._actuate_base(insertion / substeps, twist / substeps)
            if any(preload):
                wp.launch(add_world_force, dim=self.body_ids.shape[0],
                          inputs=[self.body_ids, float(preload[0]), float(preload[1]),
                                  float(preload[2]), 1],
                          outputs=[self.state_0.body_f], device=self.device)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def body_positions(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.bodies, :3]

    def node_radii(self) -> np.ndarray:
        return np.array([self.contact_frame.project(p).r for p in self.body_positions()])

    def wall_max_deflection(self) -> float:
        return self.solver.wall_max_deflection()


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])
