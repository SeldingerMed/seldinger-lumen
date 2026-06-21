"""Endovascular profile: continuum guidewire/catheter in vasculature.

This is the lead procedure (doc §9), but only a *profile* over the generic core.
It supplies generic device geometry and selects the projective (X-ray) sensor.

Important: this profile contains only generic, publishable parameters. Any
real-data calibration -- HGO wall fitting, clot models, patient asset pipelines,
trained policies -- stays in the private seldinger repos and is layered on top of
this open profile, never committed here.
"""

from lumen.profiles import Profile

ENDOVASCULAR = Profile(
    name="endovascular",
    instrument="guidewire",
    sensor="projective",
    notes="Lead procedure: mechanical thrombectomy. Generic params only.",
)

# Nominal device geometry (generic, mm). Calibration to real devices is private.
GUIDEWIRE = {
    "diameter_mm": 0.36,      # 0.014 in
    "length_mm": 1800.0,
    "torsional_dof": True,
}
MICROCATHETER = {
    "inner_diameter_mm": 0.43,
    "outer_diameter_mm": 0.70,
    "length_mm": 1500.0,
}

__all__ = ["ENDOVASCULAR", "GUIDEWIRE", "MICROCATHETER"]
