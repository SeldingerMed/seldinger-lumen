"""Recover a guidewire pose from a synthetic fluoroscopy frame (Layer 1 L1.1).

    python examples/register_pose.py
"""

from __future__ import annotations

import numpy as np

from lumen.sensors import FluoroSensor
from lumen.sensors.registration import apply_se3, register


def main():
    a = np.linspace(0, 1.2, 16)
    wire = np.stack([4 * np.sin(a), np.zeros(16), np.linspace(-10, 10, 16)], axis=1)
    sensor = FluoroSensor(res=48, n_samples=120, nu=64, nv=64)
    carm = sensor.default_carm(wire, axis=(1, 0, 0))            # view +x; in-plane = y,z
    true = np.array([0.0, 3.0, -2.5, 0.20, 0.0, 0.0])          # in-plane translate + roll
    target, _ = sensor.render(apply_se3(wire, true), carm=carm)

    pose, hist = register(target, wire, sensor, carm, iters=25)
    print(f"image loss {hist[0]:.4f} -> {hist[-1]:.6f}")
    print(f"true  (ty, tz, rx): {true[1]:+.2f} {true[2]:+.2f} {true[3]:+.2f}")
    print(f"recov (ty, tz, rx): {pose[1]:+.2f} {pose[2]:+.2f} {pose[3]:+.2f}")
    print("depth tx is ambiguous from one view — pass [carm1, carm2] for biplanar")


if __name__ == "__main__":
    main()
