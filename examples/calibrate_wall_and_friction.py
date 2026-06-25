"""Device-as-sensor, the full M2 pair: jointly recover wall stiffness AND friction from
biplanar fluoroscopy, and report whether they are separably identifiable (Layer 1 L1.2).

    python examples/calibrate_wall_and_friction.py

Wall stiffness shows up as a lateral BULGE (the device pressed into a soft wall yields
more); friction shows up as an axial stick-slip LAG (a pushed device's tip lags when the
wall grips). The two displace the device in near-orthogonal directions, so a joint inverse
recovers both — and the Fisher conditioning number tells you when a single view confounds
them and biplanar resolves it (doc §3.6, the M2 'bounded identifiability' gate).
"""

from __future__ import annotations

from lumen.sensors import FluoroSensor
from lumen.sensors.device_as_sensor import (device_wall_and_friction,
                                            estimate_wall_and_friction, joint_identifiability)


def main():
    sensor = FluoroSensor(res=48, n_samples=120, nu=64, nv=64)
    nodes = device_wall_and_friction(4.0e3, 0.4)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))            # +x foreshortens the bulge
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))            # +y sees both in-plane

    true_C10, true_mu = 6.0e3, 0.5
    targets = [sensor.render(device_wall_and_friction(true_C10, true_mu), carm=c)[0]
               for c in (cx, cy)]
    C10, mu, _ = estimate_wall_and_friction(targets, sensor, [cx, cy],
                                            init_C10=2e3, init_mu=0.2, iters=40)
    print(f"wall stiffness C10: true {true_C10:.0f}  recovered {C10:.0f}  "
          f"({100 * abs(C10 - true_C10) / true_C10:.1f}%)")
    print(f"friction      mu : true {true_mu:.2f}   recovered {mu:.3f}   "
          f"({abs(mu - true_mu):.3f} abs)")

    print("\njoint identifiability (cond: high = confounded; lam_min: worst-determined "
          "direction, higher = better):")
    ji = joint_identifiability(true_C10, true_mu, sensor,
                               {"mono (+x)": [cx], "mono (+y)": [cy], "biplanar": [cx, cy]})
    for view, r in ji.items():
        print(f"  {view:11s}: cond = {r['cond']:10.1f}   lam_min = {r['lam_min']:.2e}   "
              f"cross-coupling = {r['corr']:+.2f}")
    print("=> the bulge-aligned mono view (+x) confounds stiffness and friction (cond -> huge); "
          "biplanar rescues it and strictly improves the worst-determined direction (doc §3.6)")


if __name__ == "__main__":
    main()
