"""Layer 1 — sensor / observation models (doc §3.6, §4).

Renders the Layer-0 scene to the *native* clinical modality so calibration and the
device-as-sensor loop close on the real signal (projective X-ray), not RGB. The
sensor is the third modality swap point (anatomy + instrument + sensor, doc §3.9):
FluoroSensor (endovascular X-ray) plus a luminal-RGB sibling (LuminalCamera, L1.4)
that proves the swap — same scene, different observation model.
"""

from lumen.sensors.carm import CArm
from lumen.sensors.device_as_sensor import (device_on_wall, estimate_wall_stiffness,
                                            identifiability, sensitivity, wall_yield)
from lumen.sensors.fluoro import FluoroSensor
from lumen.sensors.luminal import LuminalCamera
from lumen.sensors.perception import detect_device_tip, device_centroid
from lumen.sensors.realism import RealismParams, degrade
from lumen.sensors.registration import apply_se3, register

__all__ = ["FluoroSensor", "CArm",
           "register", "apply_se3",                      # L1.1 registration
           "estimate_wall_stiffness", "sensitivity", "identifiability",
           "device_on_wall", "wall_yield",               # L1.2 device-as-sensor
           "detect_device_tip", "device_centroid",       # L1.3 perception front-end
           "RealismParams", "degrade", "LuminalCamera"]  # L1.4 realism seam + 2nd modality
