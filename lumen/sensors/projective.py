"""Differentiable projective (X-ray / fluoroscopy) renderer.

The clinical observation in endovascular work is a 2-D projection of a 3-D scene
formed by X-ray attenuation (doc §1.2). This is a minimal, fully-differentiable
projective renderer: it splats the radio-opaque device onto an image plane so the
whole forward map -- action -> physics -> 3-D scene -> synthetic fluoroscopy -- is
one torch graph (doc §3.6). That differentiability is what enables the
"device-as-sensor" inverse: recover patient-specific mechanics from how the device
deflects in the image.

This is the generic, synthetic coupling (open). Photoreal-DSA realism (scatter,
beam hardening; the DiffDRR -> DDGS-CT ladder, doc §4.1) and calibration against
*real* DSA are the private / later layers; the seam is the same renderer
interface. Orthographic projection here stands in for a distant X-ray source; a
pinhole/cone-beam variant slots in without changing the loop.
"""

from __future__ import annotations

import torch

# which world axis is the projection (viewing) direction -> which two are kept
_KEEP = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}


class ProjectiveRenderer:
    """Orthographic attenuation renderer: 3-D nodes -> 2-D image via Gaussian splats."""

    def __init__(self, height=48, width=48, bounds=(-6.0, 6.0, 0.0, 36.0),
                 view="y", sigma=1.2, amplitude=1.0, dtype=torch.float64):
        self.iu, self.iv = _KEEP[view]
        self.sigma, self.amp, self.dtype = sigma, amplitude, dtype
        umin, umax, vmin, vmax = bounds
        u = torch.linspace(umin, umax, width, dtype=dtype)
        v = torch.linspace(vmin, vmax, height, dtype=dtype)
        V, U = torch.meshgrid(v, u, indexing="ij")          # [H, W]
        self.U, self.V = U, V

    def render(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes [B, N, 3] (or [N, 3]) -> attenuation image [B, H, W].

        Differentiable in node positions, so gradients flow from image-space loss
        back through the physics that placed the nodes.
        """
        if nodes.ndim == 2:
            nodes = nodes.unsqueeze(0)
        u = nodes[..., self.iu]                              # [B, N]
        v = nodes[..., self.iv]
        du = self.U[None, None] - u[..., None, None]         # [B, N, H, W]
        dv = self.V[None, None] - v[..., None, None]
        blobs = self.amp * torch.exp(-(du ** 2 + dv ** 2) / (2 * self.sigma ** 2))
        return blobs.sum(dim=1)                              # [B, H, W]
