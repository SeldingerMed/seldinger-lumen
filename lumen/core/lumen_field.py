"""The lumen field R(s, theta, t).

The boundary of the tubular cavity, as a radius that varies along arc-length,
circumference, and time. This single field is *shared* by the wall mechanics and
the contact barrier (doc §3.5.6): wall deformation is written as a change in R,
and the contact gap reads the same R. Pulsatility is just a temporal modulation
of R and therefore costs nothing extra in the contact computation.

P0 ships the field representation and evaluation only; wall mechanics and the
pulsatile driver attach in later phases.
"""

from __future__ import annotations

import numpy as np


class LumenField:
    """R(s, theta) sampled on a grid, with periodic interpolation in theta.

    Pass a 1-length theta axis for an axisymmetric lumen (a plain tube, or a
    stenosis profile that depends only on arc-length).
    """

    def __init__(self, s_grid: np.ndarray, theta_grid: np.ndarray, R: np.ndarray):
        self.s = np.asarray(s_grid, dtype=float)
        self.theta = np.asarray(theta_grid, dtype=float)
        self.R = np.asarray(R, dtype=float)
        if self.R.shape != (len(self.s), len(self.theta)):
            raise ValueError("R must have shape (len(s_grid), len(theta_grid))")

    @classmethod
    def cylinder(cls, length: float, radius: float, n: int = 2) -> "LumenField":
        s = np.linspace(0.0, length, n)
        return cls(s, np.array([0.0]), np.full((n, 1), float(radius)))

    @classmethod
    def stenosis(cls, length: float, radius: float, at: float,
                 severity: float = 0.6, width: float = 5.0, n: int = 64) -> "LumenField":
        """Axisymmetric narrowing: a Gaussian dip in radius centred at ``at``."""
        s = np.linspace(0.0, length, n)
        dip = severity * np.exp(-0.5 * ((s - at) / width) ** 2)
        return cls(s, np.array([0.0]), (radius * (1.0 - dip))[:, None])

    def eval(self, s: float, theta: float = 0.0) -> float:
        """Bilinear interpolation; nearest-clamp in s, periodic wrap in theta."""
        si = float(np.interp(s, self.s, np.arange(len(self.s))))
        i0 = int(np.clip(np.floor(si), 0, len(self.s) - 1))
        i1 = min(i0 + 1, len(self.s) - 1)
        fs = si - i0
        if len(self.theta) == 1:
            return float(self.R[i0, 0] * (1 - fs) + self.R[i1, 0] * fs)
        th = (theta - self.theta[0]) % (2 * np.pi)
        step = (self.theta[-1] - self.theta[0]) / (len(self.theta) - 1)
        tj = th / step
        j0 = int(np.floor(tj)) % len(self.theta)
        j1 = (j0 + 1) % len(self.theta)
        ft = tj - np.floor(tj)
        r0 = self.R[i0, j0] * (1 - ft) + self.R[i0, j1] * ft
        r1 = self.R[i1, j0] * (1 - ft) + self.R[i1, j1] * ft
        return float(r0 * (1 - fs) + r1 * fs)

    def gap(self, s: float, theta: float, r: float) -> float:
        """Contact gap g = R - r. Positive = clearance, <= 0 = penetration."""
        return self.eval(s, theta) - r
