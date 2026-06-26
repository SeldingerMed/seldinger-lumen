"""Physically-grounded stochastic contact gradient (doc §3.5.8): rehabilitating the dead
deterministic gradient with physical gap noise, and the sigma reach-vs-bias tradeoff.

    python examples/stochastic_contact_gradient.py

The deterministic contact reaction is exactly 0 outside the active band, so a calibrator
started there has a zero gradient and is stuck. Modelling the gap as jittered by physical
uncertainty (blood-film thickness, wall roughness, manufacturing tolerance) of scale sigma
makes E[reaction] smooth, so the randomized-smoothing gradient is alive and recovers the
offset. Larger sigma reaches further into the flat region but biases the optimum more — the
identifiability knob the doc asks reviewers to judge.
"""

from __future__ import annotations

from lumen.accurate.stochastic import (contact_reaction, deterministic_grad,
                                       recover_by_smoothed_descent)


def main():
    R, d_hat = 2.0, 0.3
    react = lambda th: contact_reaction(th, R, d_hat, kappa=1.0)
    theta_true = 1.85                                  # gap 0.15 < d_hat -> in contact
    f_target = float(react(theta_true))
    theta0 = 1.5                                        # gap 0.50 > d_hat -> OUTSIDE the band

    print(f"target reaction f={f_target:.3f} at theta_true={theta_true} (in contact)")
    print(f"start theta0={theta0} (gap {R - theta0:.2f} > d_hat {d_hat}: inactive)\n")
    print(f"deterministic gradient at the start: {deterministic_grad(react, theta0):.2e}  "
          f"(dead — no signal toward contact)\n")

    print(f"{'sigma':>6} {'recovered theta':>16} {'bias':>10}   (reach vs bias)")
    for sigma in (0.08, 0.12, 0.2, 0.3):
        out = recover_by_smoothed_descent(f_target, theta0, sigma=sigma, lr=0.5,
                                          iters=400, reaction=react)
        print(f"{sigma:>6.2f} {out['theta']:>16.4f} {out['theta'] - theta_true:>+10.4f}")
    print("\n=> the smoothed gradient recovers theta from the flat region where the "
          "deterministic one is dead; sigma trades reach for bias (doc §3.5.8)")


if __name__ == "__main__":
    main()
