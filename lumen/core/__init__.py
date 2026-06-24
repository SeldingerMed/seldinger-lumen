"""Modality-agnostic geometry core.

Nothing here names a vessel, an airway, or any specific anatomy. It provides the
tube-intrinsic frame (``frame``) and the shared lumen field ``R(s,theta)``
(``lumen_field``) that the Newton solver (``lumen.newton``) builds on. Domain
specifics live in ``lumen.profiles``.
"""

from lumen.core.frame import CenterlineFrame, Projection
from lumen.core.lumen_field import LumenField
from lumen.core.tree import TreeProjection, VascularTree

__all__ = ["CenterlineFrame", "Projection", "LumenField", "VascularTree", "TreeProjection"]
