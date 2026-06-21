"""Energy-based stepper for rod + contact.

Semi-implicit integration where the conservative force is the negative gradient
(autograd) of the total potential energy (internal elastic + contact barrier),
plus non-conservative friction and a proximal driving boundary condition (the
device is actuated only at its proximal end -- translate/insert, doc §1.2).

Because the force is `-d(energy)/dx` via autograd and the whole rollout is a torch
graph, gradients of any scalar loss w.r.t. physical parameters flow exactly --
which is what makes system identification (recover friction/stiffness) work
(doc §3.5.7: gradients are reliable for calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod
from lumen.physics.wall import WallShell


@dataclass
class SimConfig:
    dt: float = 5.0e-3
    steps: int = 200
    insertion_rate: float = 0.0     # Dirichlet: mm/step advance of the proximal node
    anchor_base: bool = True        # Dirichlet velocity BC at the proximal node
    push_force: float = 0.0         # Neumann: constant push at the base (you push a catheter)
    preload_force: float = 0.0      # constant wall-ward press (+x) per node; sustains normal load


class Solver:
    def __init__(self, geom: ContactGeometry, contact: ContactParams | None = None,
                 cfg: SimConfig | None = None, wall: WallShell | None = None):
        self.geom = geom
        self.cp = contact or ContactParams()
        self.cfg = cfg or SimConfig()
        self.wall = wall

    def _deformed_R(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Node lumen radius using the deformable wall: R0(s) + w(s,theta)."""
        proj = self.geom.project(x)
        R0 = self.geom._R_of_s(proj["s"])
        return R0 + self.wall.sample(proj["s"], proj["theta"], w)

    def _forces(self, rod: Rod, cp: ContactParams):
        """Conservative forces on rod (and wall, if present), -grad(total energy).

        Detached copy: the stiff elastic/contact force is treated as locally
        constant across the backward pass, so param gradients flow through the
        differentiable friction term, not this stiff graph (doc §3.5.7).
        Returns (F_rod, F_wall_or_None).
        """
        with torch.enable_grad():   # works even under an outer torch.no_grad()
            xd = rod.x.detach().requires_grad_(True)
            if self.wall is None:
                E = rod.internal_energy(xd) + self.geom.barrier_energy(xd, cp)
                (gx,) = torch.autograd.grad(E.sum(), xd)
                return -gx.detach(), None
            wd = self.wall.w.detach().requires_grad_(True)
            R_over = self._deformed_R(xd, wd)
            E = (rod.internal_energy(xd) + self.wall.energy(wd)
                 + self.geom.barrier_energy(xd, cp, R_override=R_over))
            gx, gw = torch.autograd.grad(E.sum(), [xd, wd])
            return -gx.detach(), -gw.detach()

    def step(self, rod: Rod, mu: torch.Tensor | None = None) -> Rod:
        """One overdamped (first-order) step. `mu` (per-batch) overrides cp.mu.

        Quasi-static dynamics: node velocity = total force / drag. Inertia is
        negligible for slow catheter motion, and gradient-flow on the energy is
        unconditionally robust with stiff contact (where explicit inertial
        integration blows up). drag = rod.params.damping.
        """
        cfg = self.cfg
        cp = self.cp
        if mu is not None:
            cp = ContactParams(kappa=cp.kappa, d_hat=cp.d_hat, mu=mu)
        c = rod.params.damping

        F, Fw = self._forces(rod, cp)                     # conservative forces
        base_dir = rod.x[:, 1] - rod.x[:, 0]
        base_dir = base_dir / torch.linalg.norm(base_dir, dim=-1, keepdim=True).clamp_min(1e-12)
        # Neumann driving: a constant push at the base, transmitted through the rod
        if cfg.push_force != 0.0:
            F = F.clone()
            F[:, 0] = F[:, 0] + cfg.push_force * base_dir
        if cfg.preload_force != 0.0:
            F = F.clone()
            F[:, :, 0] = F[:, :, 0] + cfg.preload_force   # constant +x wall-ward press
        v_free = F / c                                    # would-be velocity (no friction)
        Ff = self.geom.friction_force(rod.x, v_free, cp)  # opposes tangential motion
        v = (F + Ff) / c
        # Dirichlet driving BC (alternative): prescribe the base velocity
        if cfg.anchor_base:
            v = v.clone()
            v[:, 0] = (cfg.insertion_rate / cfg.dt) * base_dir
        x = rod.x + cfg.dt * v
        rod.x, rod.v = x, v
        # wall DOFs evolve by the same overdamped flow on their elastic + contact energy
        if self.wall is not None:
            self.wall.w = self.wall.w + cfg.dt * Fw / self.wall.params.drag
        return rod

    def rollout(self, rod: Rod, mu: torch.Tensor | None = None) -> Rod:
        for _ in range(self.cfg.steps):
            rod = self.step(rod, mu=mu)
        return rod
