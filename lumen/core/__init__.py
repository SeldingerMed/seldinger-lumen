"""Modality-agnostic physics core.

Nothing in this package names a vessel, an airway, or any specific anatomy. It
operates on a centerline + lumen field + instrument, observed through a sensor.
Domain specifics live in ``lumen.profiles``; observation models in
``lumen.sensors``.

P0 ships ``frame`` (tube-intrinsic coordinates) and ``lumen_field`` (R(s,theta)).
Later phases add: ``instrument`` (Cosserat/VBD rod + torsion), ``wall`` (reduced
anisotropic shell sharing R), ``contact`` (tube-intrinsic narrowphase + barrier
Warp kernels), ``coupling`` (flow), and ``solver`` (Newton custom solver).
"""

from lumen.core.frame import CenterlineFrame, Projection
from lumen.core.lumen_field import LumenField

__all__ = ["CenterlineFrame", "Projection", "LumenField"]
