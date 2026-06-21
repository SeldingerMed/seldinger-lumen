"""Device-as-sensor: recover mechanics from the projected image (doc §3.6).

Closes the differentiable loop action -> physics -> 3-D scene -> fluoroscopy and
runs it in inverse: given an observed (synthetic) fluoroscopy image of how the
device deflected, recover a physical parameter by matching a rendered image. The
gradient flows from image-space loss, through the renderer, through the physics
rollout, to the parameter -- no access to the 3-D device state, only its 2-D
shadow. That is the conceptual payoff: every deflection visible in fluoro is a
mechanical experiment on the vessel.

Honest limit (doc §3.6): identifiability from a single 2-D projection is
under-determined; this demo recovers one scalar from one clean synthetic view.

Run:  python -m lumen.physics.imaging
"""

from __future__ import annotations

import torch

from lumen.physics import sysid
from lumen.physics.contact import ContactParams
from lumen.physics.solver import Solver
from lumen.sensors.projective import ProjectiveRenderer


def render_final(solver, make_rod, renderer, mu):
    """Roll out the physics and render the final device state to an image."""
    final = solver.rollout(make_rod(), mu=mu).x
    return renderer.render(final)


def calibrate_from_image(mu_true=0.5, mu_init=0.05, steps=35, iters=20):
    """Recover friction purely from a projected image. Returns (mu_hat, loss)."""
    geom, make_rod, cfg, _ = sysid.sliding_experiment(mu_true, steps=steps)
    renderer = ProjectiveRenderer(dtype=torch.float64)
    solver = Solver(geom, contact=ContactParams(**sysid.CP), cfg=cfg)
    with torch.no_grad():
        observed_img = render_final(
            solver, make_rod, renderer,
            torch.tensor([mu_true], dtype=torch.float64))

    mu = torch.tensor([mu_init], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([mu], lr=0.5, max_iter=iters,
                            line_search_fn="strong_wolfe")
    last = {}

    def closure():
        opt.zero_grad()
        img = render_final(solver, make_rod, renderer, mu.clamp(1e-3, 2.0))
        loss = ((img - observed_img) ** 2).mean()
        loss.backward()
        last["loss"] = float(loss.detach())
        return loss

    opt.step(closure)
    return float(mu.detach().clamp(1e-3, 2.0)), last["loss"]


def main():
    mu_true = 0.45
    mu_hat, loss = calibrate_from_image(mu_true=mu_true)
    print(f"[device-as-sensor] mu_true={mu_true:.3f}  mu_hat={mu_hat:.3f}  "
          f"img_loss={loss:.3e}")


if __name__ == "__main__":
    main()
