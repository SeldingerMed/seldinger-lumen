"""lumen -- a differentiable, GPU-parallel contact & coupling solver for a
continuum instrument inside a deformable lumen.

The core is modality-agnostic: it models "a slender device in a deformable tube
observed through some sensor." Endovascular intervention is one *profile*; a
lumen is equally an airway, a bowel, or a ureter. New modalities are added under
``lumen.profiles`` without touching the core.

Distributed on PyPI as ``lumensim`` (the import name stays ``lumen``).
"""

__version__ = "0.0.0"

from lumen.core.frame import CenterlineFrame, Projection
from lumen.core.lumen_field import LumenField

__all__ = ["CenterlineFrame", "Projection", "LumenField", "__version__"]
