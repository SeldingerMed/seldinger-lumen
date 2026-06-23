"""Device-as-sensor: recover wall stiffness from fluoroscopy + report identifiability
(Layer 1 L1.2).

    python examples/estimate_stiffness.py
"""

from __future__ import annotations

from lumen.sensors import FluoroSensor
from lumen.sensors.device_as_sensor import device_on_wall, estimate_wall_stiffness, sensitivity


def main():
    sensor = FluoroSensor(res=48, n_samples=120, nu=64, nv=64)
    nodes = device_on_wall(4.0e3)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))            # bulge +x is depth for view +x
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))            # ...and in-plane for view +y

    true = 6.0e3
    targets = [sensor.render(device_on_wall(true), carm=c)[0] for c in (cx, cy)]
    est, hist = estimate_wall_stiffness(targets, sensor, [cx, cy], init_C10=2e3, iters=25)
    print(f"true C10 = {true:.0f}   recovered = {est:.0f}   ({100 * abs(est - true) / true:.1f}%)")
    print("identifiability — image change per +10% C10 (bigger = better determined):")
    print(f"  mono, depth-aligned displacement : {sensitivity(true, sensor, cx, bulge_dir=(1, 0, 0)):.2e}")
    print(f"  biplanar                         : {sensitivity(true, sensor, [cx, cy], bulge_dir=(1, 0, 0)):.2e}")
    print("=> stiffness from a single 2-D view is under-determined; biplanar resolves it (doc §3.6)")


if __name__ == "__main__":
    main()
