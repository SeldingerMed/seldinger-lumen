"""Fluoroscopy realism seam (Layer 1, doc §4.1; answers the §307 open question).

The clean DRR is a noise-free line integral. Real fluoro adds detector physics:
finite dose -> Poisson photon (quantum) noise; a finite focal spot + detector MTF
-> blur; X-ray scatter -> a low-frequency additive glow that washes out contrast;
and beam hardening -> a polychromatic beam makes long/dense paths read as LESS
attenuating than the monochromatic value. The bible's open question (§307) is *how
much* of this is needed before sim-to-real calibration is trustworthy — to be
answered empirically, incrementally, up the DiffDRR->DDGS ladder.

So this is a calibratable SEAM, not a fixed pipeline: every effect is off by
default (``degrade`` with default params is the identity), and the calibration
loop dials each one in to match real DSA.

Works attenuation -> attenuation (line-integral in, degraded line-integral out) so
it is drop-in for the perception/registration code, which expects the device as the
bright/high-A region. The detector physics is applied in the photon-count domain
internally (where it is physical) and converted back via Beer–Lambert.

ponytail: a few-line separable Gaussian instead of a scipy dependency. The noise
is stochastic but its expectation is smooth (doc §3.5.8 stochastic-contact
philosophy), so finite-difference calibration through ``degrade`` is well-posed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RealismParams:
    """Detector-physics knobs; all defaults are the no-op (degrade == identity).

    Each is independently calibratable against real DSA (doc §3.6, §307)."""
    i0: float | None = None       # incident photons/pixel; None = infinite dose (no quantum noise)
    psf_sigma: float = 0.0        # detector blur (focal spot + MTF), pixels
    scatter_frac: float = 0.0     # scatter-to-primary ratio (low-frequency additive glow)
    scatter_sigma: float = 8.0    # scatter glow blur width, pixels
    beam_hardening: float = 0.0   # cupping coefficient; A' = A/(1+bh·A) (concave, monotone)
    read_noise: float = 0.0       # additive Gaussian electronic noise, counts
    seed: int | None = None       # RNG seed for the stochastic terms (reproducible)

    def __post_init__(self):
        if self.i0 is not None and self.i0 <= 0:      # else exp/log on a non-positive dose -> NaN/inf
            raise ValueError(f"i0 (photon budget) must be positive or None, got {self.i0}")


def _gaussian_kernel(sigma):
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur1d(m, k):
    return np.convolve(np.pad(m, len(k) // 2, mode="reflect"), k, mode="valid")


def _gaussian_blur(img, sigma):
    """Separable reflect-padded Gaussian; output same shape as input."""
    if sigma <= 0:
        return img
    k = _gaussian_kernel(sigma)
    out = np.apply_along_axis(_blur1d, 1, img, k)
    return np.apply_along_axis(_blur1d, 0, out, k)


def degrade(A, params: RealismParams | None = None):
    """Apply detector physics to a line-integral image ``A``; return a degraded ``A``.

    Order is physical: beam hardening (attenuation domain) -> Beer–Lambert to photon
    counts -> additive scatter glow -> detector blur -> Poisson quantum noise (+ read
    noise) -> back to attenuation. With default params this is the identity."""
    p = params or RealismParams()
    A = np.asarray(A, float)
    if not (p.beam_hardening or p.scatter_frac or p.psf_sigma or p.read_noise) and p.i0 is None:
        return A                                      # all effects off -> exact identity (and cheap)
    Ad = A / (1.0 + p.beam_hardening * A) if p.beam_hardening else A
    i0 = float(p.i0) if p.i0 is not None else 1.0    # arbitrary scale when noiseless
    intensity = i0 * np.exp(-Ad)                      # primary transmitted photons
    if p.scatter_frac:                               # glow leaks across the detector
        intensity = intensity + p.scatter_frac * _gaussian_blur(intensity, p.scatter_sigma)
    if p.psf_sigma:
        intensity = _gaussian_blur(intensity, p.psf_sigma)
    if p.i0 is not None:                             # finite dose -> quantum noise
        rng = np.random.default_rng(p.seed)
        intensity = rng.poisson(np.clip(intensity, 0.0, None)).astype(float)
        if p.read_noise:
            intensity = intensity + rng.normal(0.0, p.read_noise, intensity.shape)
    intensity = np.clip(intensity, 1e-8 * i0, None)  # avoid log(0)
    return -np.log(intensity / i0)


if __name__ == "__main__":  # self-check: each knob does what it claims
    rng = np.random.default_rng(0)
    A = np.zeros((48, 48)); A[20:28, 20:28] = 3.0          # a bright "device" block on a flat field

    assert np.allclose(degrade(A), A), "default params must be the identity"

    noisy = degrade(A, RealismParams(i0=200.0, seed=1))    # finite dose roughens the flat field
    assert noisy[:10, :10].std() > 1e-3, "low dose should add quantum noise"

    blurred = degrade(A, RealismParams(psf_sigma=2.0))     # PSF spreads the block edge
    assert blurred.max() < A.max() and blurred[19, 24] > A[19, 24], "PSF should spread the edge"

    scattered = degrade(A, RealismParams(scatter_frac=0.5))  # scatter washes out contrast
    assert np.ptp(scattered) < np.ptp(A), "scatter should reduce contrast"

    hardened = degrade(A, RealismParams(beam_hardening=0.2))  # cupping lowers the peak A
    assert hardened.max() < A.max(), "beam hardening should lower deep attenuation"

    print("realism self-check ok")
