"""Train an IMAGE-OBSERVATION navigation policy (Layer 1 L1.3).

    python examples/train_fluoro_nav.py

The policy sees only the synthetic fluoro (device tip detected in the rendered image),
not privileged simulator state — perception in the real modality (doc §3.6). CEM over
the batched sim, no torch. This quick smoke run is sized for CPU; increase `pop`,
`iters`, or the FluoroSensor resolution for a stronger search.
"""

from __future__ import annotations

import numpy as np

from lumen.assets import procedural
from lumen.rl.cem import train_cem
from lumen.rl.fluoro_nav import fluoro_env_factory
from lumen.sensors import FluoroSensor


def main():
    asset = procedural.straight_tube(80.0, 2.0)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    sensor = FluoroSensor(mu_device=1.0, res=20, n_samples=50, nu=24, nv=24)
    factory = fluoro_env_factory(sensor, view_axis=(1, 0, 0))
    print("training an image-observation policy (obs = device tip detected in fluoro)...",
          flush=True)
    _, hist = train_cem(np.asarray(pts), float(np.asarray(lumen.R).mean()),
                        lumen_field=lumen, env_factory=factory, warm_start=(2, -3.0),
                        pop=12, iters=7, device="cpu",
                        log=lambda r: print(f"  iter {r['iter']:2d}  "
                                            f"success={r['success_rate']:.2f}",
                                            flush=True))
    print(f"final success rate (image-based control): {hist[-1]['success_rate']:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
