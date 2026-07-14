"""Emit the procedural demo assets used by docs/benchmarks.

    python examples/make_demo_assets.py [out_dir]
"""

from pathlib import Path
import sys

from lumen.assets import procedural

DEFAULT_OUT = Path(__file__).parent / "assets"


def main(out_dir=None):
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    procedural.straight_tube().save(str(out / "straight_tube.json"))
    procedural.stenotic_tube().save(str(out / "stenotic_tube.json"))
    procedural.tortuous_tube().save(str(out / "tortuous_tube.json"))
    procedural.bifurcation().save(str(out / "bifurcation.json"))
    procedural.tortuous_tree().save(str(out / "tortuous_tree.json"))
    print(f"wrote demo assets to {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
