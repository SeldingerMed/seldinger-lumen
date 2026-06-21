"""Emit the procedural demo assets used by docs/benchmarks."""

from pathlib import Path

from lumen.assets import procedural

OUT = Path(__file__).parent / "assets"


def main():
    OUT.mkdir(exist_ok=True)
    procedural.straight_tube().save(str(OUT / "straight_tube.json"))
    procedural.stenotic_tube().save(str(OUT / "stenotic_tube.json"))
    procedural.bifurcation().save(str(OUT / "bifurcation.json"))
    print(f"wrote demo assets to {OUT}")


if __name__ == "__main__":
    main()
