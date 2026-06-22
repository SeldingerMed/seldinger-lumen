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
        s = clot.s_grid[clot.mask]
        if s.size == 0:
            return 0.0
        a, b = self.deployed_center - self.span / 2, self.deployed_center + self.span / 2
        clo, chi = float(s.min()), float(s.max())
        overlap = max(0.0, min(b, chi) - max(a, clo)) / max(chi - clo, 1e-9)
        return self.radial_force * self.n_struts * self.embedment * overlap
