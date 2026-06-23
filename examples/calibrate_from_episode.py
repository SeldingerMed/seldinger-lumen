"""L2.3 — sim2sim wall-stiffness calibration from a captured episode.

    python examples/calibrate_from_episode.py

Generates a wall-probe episode at a known stiffness, saves it, reloads it, and runs
the device-as-sensor inverse to recover the stiffness from the stored fluoro frames —
reporting the recovery error against the ground truth in meta.notes. Shows mono vs
biplanar. The math is numpy; needs warp/newton importable.
"""

from __future__ import annotations

import tempfile

from lumen.data import EpisodeDataset, calibrate_from_episode, probe_episode
from lumen.sensors import FluoroSensor
from lumen.sensors.device_as_sensor import device_on_wall


def main():
    true_C10 = 6.0e3
    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=90, nu=44, nv=44)
    nodes = device_on_wall(true_C10)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))

    for name, carms in (("mono", [cx]), ("biplanar", [cx, cy])):
        with tempfile.TemporaryDirectory() as d:
            probe_episode(true_C10, sensor, carms=carms, notes={"case": name}).save(d)
            ep = EpisodeDataset(d)[0]                       # reload — carms/sensor from the manifest
            res = calibrate_from_episode(ep, init_C10=2.0e3, iters=20)
            print(f"{name:9s}  views={res['n_views']}  true={res['true_C10']:.0f}  "
                  f"recovered={res['recovered_C10']:.0f}  rel_error={res['rel_error']:.3%}")


if __name__ == "__main__":
    main()
