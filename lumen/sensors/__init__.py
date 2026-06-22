"""Layer 1 — sensor / observation models (doc §3.6, §4).

Renders the Layer-0 scene to the *native* clinical modality so calibration and the
device-as-sensor loop close on the real signal (projective X-ray), not RGB. The
sensor is the third modality swap point (anatomy + instrument + sensor, doc §3.9):
FluoroSensor (endovascular X-ray) first; a luminal-RGB sibling later.
"""

from lumen.sensors.carm import CArm
from lumen.sensors.fluoro import FluoroSensor

__all__ = ["FluoroSensor", "CArm"]
