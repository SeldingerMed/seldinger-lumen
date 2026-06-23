#!/usr/bin/env python3
"""Open-core firewall check (run in CI; fails the build on violation).

Two hard boundaries keep this repo cleanly Apache-2.0 releasable:

  1. No CathSim. CathSim is CC-BY-NC-SA-4.0; importing, depending on, or deriving
     assets from it would contaminate this repo's license. The whole point of
     lumen is to be the clean-room generic solver, so this must never appear.

  2. No patient data. Every committed asset must be procedurally generated
     (provenance == "procedural"). Patient-derived geometry belongs in the
     private seldinger repos behind the asset-schema seam, never here.

Exit non-zero with a clear message if either boundary is crossed.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BANNED = ("cathsim",)
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".github"}


# banned tokens are checked in code + packaging + scripts/configs (the real
# contamination surface: imports, deps, build/run scripts). Markdown/NOTICE may
# name CathSim to *document* the boundary, so .md is excluded.
CODE_SUFFIXES = (".py", ".toml", ".cfg", ".txt", ".sh", ".yaml", ".yml", ".ipynb")


def _iter_source():
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in CODE_SUFFIXES:
            yield p


def check_no_banned() -> list[str]:
    bad = []
    self_path = pathlib.Path(__file__).resolve()
    for p in _iter_source():
        if p.resolve() == self_path:
            continue  # this checker names the banned token by necessity
        text = p.read_text(errors="ignore").lower()
        for token in BANNED:
            if token in text:
                bad.append(f"  banned token {token!r} in {p.relative_to(ROOT)}")
    return bad


def _provenances(obj):
    """Yield every value under any 'provenance' key, recursively (top-level OR nested,
    e.g. an episode manifest's meta.provenance) — a hand-edited file can't hide it deeper."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "provenance" and isinstance(v, str):
                yield v
            else:
                yield from _provenances(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _provenances(item)


def check_provenance() -> list[str]:
    bad = []
    # scan EVERY .json anywhere in the repo (not a fixed allow-list of dirs)
    for p in ROOT.rglob("*.json"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            d = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        for prov in _provenances(d):
            if prov != "procedural":
                bad.append(f"  non-procedural asset committed: {p.relative_to(ROOT)} "
                           f"(provenance={prov!r})")
                break                                      # one report per file is enough
    return bad


def main() -> int:
    problems = check_no_banned() + check_provenance()
    if problems:
        print("FIREWALL VIOLATION -- this repo must stay Apache-2.0 / patient-free:")
        print("\n".join(problems))
        return 1
    print("firewall ok: no CathSim, no patient-derived assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
