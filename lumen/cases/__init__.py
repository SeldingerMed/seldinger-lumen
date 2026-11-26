"""Real anatomy that ships with Lumen.

Each case is a vessel tree and wall surface derived from one openly licensed scan (the scan
itself is not included). Every case folder holds:

* ``case.json``: title, body region, modality, licence, attribution and where the scan is from;
* ``asset.json.gz``: the navigable lumen (centrelines, radii, branch points, entry);
* ``names.json``: which branch is which artery, for aiming at a vessel by name;
* ``surface.npz``: the vessel surface from the scan, simplified, for drawing.

Use them anywhere an import folder works, by name::

    from lumen import cases
    env = cases.make_env("head-cta-aneurysm", "right middle cerebral artery")

or ``lumen view head-cta-aneurysm``. ``lumen cases`` lists them with their credits.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path


def _root() -> Path:
    return Path(str(resources.files("lumen.cases")))


def names() -> list[str]:
    """Ids of the shipped cases."""
    root = _root()
    return sorted(p.name for p in root.iterdir() if (p / "case.json").is_file())


def case_folder(name: str) -> Path | None:
    """The folder of a shipped case, or None when ``name`` is not one."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    p = _root() / name
    return p if (p / "case.json").is_file() else None


def card(name: str) -> dict:
    folder = case_folder(name)
    if folder is None:
        raise KeyError(f"no shipped case called {name!r} (have: {', '.join(names())})")
    return json.loads((folder / "case.json").read_text())


def catalogue() -> list[dict]:
    """Every shipped case's card, with its id."""
    return [{"id": n, **card(n)} for n in names()]


def load(name: str):
    """``(asset, names)`` for a shipped case."""
    from lumen.scan.pipeline import load_imported

    folder = case_folder(name)
    if folder is None:
        raise KeyError(f"no shipped case called {name!r} (have: {', '.join(names())})")
    return load_imported(folder)


def make_env(name: str, target: str, **kwargs):
    """A navigation environment on a shipped case, aimed at a vessel by name or node id."""
    from lumen.scan.pipeline import make_env as _make

    folder = case_folder(name)
    if folder is None:
        raise KeyError(f"no shipped case called {name!r} (have: {', '.join(names())})")
    return _make(folder, target, **kwargs)
