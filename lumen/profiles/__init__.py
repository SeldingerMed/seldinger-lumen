"""Domain profiles -- the repurposing surface.

A profile bundles the three things you swap to point the solver at a new
procedure (doc §3.9): default anatomy/lumen parameters, an instrument spec, and a
sensor choice. Core physics is untouched.

Shipped:
  * endovascular   guidewire/catheter in vasculature, X-ray sensor

Planned (each is a new directory here, nothing in core changes):
  * bronchoscopy   scope in airways, forward RGB sensor
  * gi_endoscopy   endoscope in bowel, forward RGB sensor
  * ureteroscopy   ureteroscope in ureter, forward RGB sensor
"""

from dataclasses import dataclass


@dataclass
class Profile:
    """Descriptor for a procedure domain."""

    name: str
    instrument: str        # which instrument spec (see lumen.core.instrument, later)
    sensor: str            # "projective" | "luminal" | "wave"
    notes: str = ""


__all__ = ["Profile"]
