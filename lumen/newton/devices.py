"""Endovascular device models (doc §3.4.1). Currently: the stent-retriever.

A stent-retriever is a self-expanding braided device deployed across the clot; its
struts embed in the thrombus (engagement) and, on retraction, drag the clot out
of the vessel. Here it is a reduced coupling to ``lumen.newton.clot.ClotField``:
the deployed span and radial force set an *engagement strength*, and on retraction
the clot is dragged proximally (``ClotField.retrieve``) unless the engagement is
too weak (slip) or the hold exceeds the clot's cohesion (fragmentation).

The full braided FE stent (Solitaire/Trevo beam-element model, doc §3.4.4) is the
accurate-tier device; this is the fast-tier engagement coupling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stentriever:
    deployed_center: float       # arc-length of the deployment centre [mm]
    span: float = 20.0           # deployed length [mm]
    radial_force: float = 0.2    # outward (chronic) radial force (sim units)
    n_struts: int = 6            # struts engaging the clot
    embedment: float = 0.5       # fraction of radial force that grips the clot

    def engagement_strength(self, clot) -> float:
        """Grip force on the clot = radial_force·struts·embedment · (span∩clot overlap)."""
        return self.engagement_strength_for_mask(clot.s_grid, clot.mask)

    def engagement_strength_for_mask(self, s_grid, mask) -> float:
        """Grip force for one env's live clot mask on a shared arc-length grid."""
        s = s_grid[mask]
        if s.size == 0:
            return 0.0
        a, b = self.deployed_center - self.span / 2, self.deployed_center + self.span / 2
        clo, chi = float(s.min()), float(s.max())
        overlap = max(0.0, min(b, chi) - max(a, clo)) / max(chi - clo, 1e-9)
        return self.radial_force * self.n_struts * self.embedment * overlap


@dataclass
class FlowDiverter:
    """A braided flow diverter laid across an aneurysm neck (doc §3.4.1, §3.4.3).

    Reused here as a flow-physics module (not a mechanical braid — that is the
    accurate-tier FE device): a porous tube whose metal coverage throttles the neck
    inflow. ``diversion`` is the effective neck blockage = the metal coverage times
    how much of the neck the deployed span actually overlaps (placement matters —
    a diverter that misses the neck does nothing). Feeds ``AneurysmSac.update`` as
    the resistance-raising factor.

    ponytail: the deployed placement (``deployed_center``/``span``) is PRESCRIBED,
    not read from the live guidewire/catheter rod — coupling the diverter's actual
    deployed position to the rod sim is the accurate-tier extension (§3.4.4)."""

    deployed_center: float          # arc-length of the deployment centre [mm]
    span: float = 20.0              # deployed length [mm]
    metal_coverage: float = 0.35    # fraction of the surface that is metal (porosity = 1−this)

    def diversion(self, aneurysm) -> float:
        """Effective neck coverage in [0, metal_coverage] = metal_coverage · (span∩neck)."""
        a, b = self.deployed_center - self.span / 2, self.deployed_center + self.span / 2
        nlo = aneurysm.s_neck - aneurysm.neck_width / 2
        nhi = aneurysm.s_neck + aneurysm.neck_width / 2
        overlap = max(0.0, min(b, nhi) - max(a, nlo)) / max(nhi - nlo, 1e-9)
        return self.metal_coverage * min(overlap, 1.0)
