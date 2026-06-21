"""Observation models -- the swappable *sensor* field.

The doc's repurposability axis (§3.9): the same physics is observed through
different sensors depending on the procedure.

  * projective   X-ray / fluoroscopy / DSA   (endovascular)   -- DiffDRR-class
  * luminal      forward RGB camera          (endoscopy)
  * wave         ultrasound / OCT            (intravascular imaging)

P0 defines the interface only. A ``Sensor`` takes a 3D scene (centerline + lumen
field + instrument state) and returns a differentiable observation; concrete
renderers attach in Layer-1 work. Keeping this as a thin protocol is what lets a
bronchoscopy profile swap an RGB sensor in for the X-ray one with no core change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Sensor(Protocol):
    """A differentiable map from scene state to an observation."""

    def render(self, scene) -> "object":
        """Return an observation (e.g. an array) for the given scene state."""
        ...


__all__ = ["Sensor"]
