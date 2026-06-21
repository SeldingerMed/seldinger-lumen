"""Generic occlusion / clot patch -- the interface, not the constitutive model.

For the lead indication (thrombectomy), clot is a local segment where the lumen R
collapses toward zero and the contact barrier becomes an adhesive, frictional
patch coupled to the device (doc §3.4.4). This module ships the *open, generic*
mechanic:

  * R-collapse: just an occluded lumen field (use LumenField.stenosis with high
    severity) -- no new code needed; the contact solver sees the narrowed R.
  * adhesion: engaged device nodes bond to their engagement position, so the clot
    resists being pushed through and is dragged along on retraction (capture +
    retrieval, the generic shape of a stent-retriever pass).

The calibrated clot constitutive / failure model (INSIST / Luraghi: hybrid
FEA-SPH thrombus, fragmentation, aspiration) is private and plugs in by replacing
`adhesion_energy` -- this stub defines the seam, not the medicine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Occlusion:
    s_center: float            # arc-length of the clot
    capture_radius: float = 4.0
    k_adhesion: float = 3.0e2  # bond stiffness (generic; real model replaces this)

    def engaged_mask(self, s: torch.Tensor) -> torch.Tensor:
        """1.0 for nodes within capture_radius of the clot, else 0.0. [B, N]."""
        return (torch.abs(s - self.s_center) < self.capture_radius).to(s.dtype)

    def adhesion_energy(self, x: torch.Tensor, x_ref: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """Sticky bond of engaged nodes to their engagement positions -> [B].

        A quadratic well; its gradient is the adhesive restoring force resisting
        the device pulling out of (or pushing through) the clot.
        """
        d2 = ((x - x_ref) ** 2).sum(dim=-1)                 # [B, N]
        return 0.5 * self.k_adhesion * (mask * d2).sum(dim=-1)
