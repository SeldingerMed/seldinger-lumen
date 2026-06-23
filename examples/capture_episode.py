"""L2.1 — capture procedural episodes into the Layer-2 schema.

    python examples/capture_episode.py [out_dir]

Generates a few procedural cases (straight / stenotic), runs the guidewire to the
target while recording the paired fluoro observation each step, and writes one
`lumen-episode/0` directory per case under <out_dir>. Reloads them and prints a
summary. Needs the full stack (newton + warp).
"""

from __future__ import annotations

import sys

from lumen.assets import procedural
from lumen.data import Episode, rollout_episode, validate
from lumen.sensors import FluoroSensor, LuminalCamera


def main(out_dir="episodes"):
    sensor = FluoroSensor(res=32, nu=64, nv=64, n_samples=96)
    cases = {
        "straight_fluoro": dict(asset=procedural.straight_tube(80.0, 2.0), sensor=sensor),
        "stenosis_fluoro": dict(asset=procedural.stenotic_tube(80.0, 2.0, severity=0.6), sensor=sensor),
        "straight_luminal": dict(asset=procedural.straight_tube(80.0, 4.0),
                                 sensor=LuminalCamera(nu=64, nv=64), modality="luminal"),
    }
    for name, kw in cases.items():
        ep = rollout_episode(max_steps=30, asset_ref=f"{name}.asset.json",
                             notes={"case": name, "true_C10": 4000.0}, **kw)
        validate(ep)
        path = f"{out_dir}/{name}"
        ep.save(path)
        back = Episode.load(path)
        validate(back, root=path)
        obs0 = back.steps[0].load_obs(path)
        print(f"{name:18s}  steps={back.outcome.steps:2d}  success={back.outcome.success!s:5s}  "
              f"final_dist={back.outcome.final_dist:6.2f}  obs{obs0.shape} -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "episodes")
