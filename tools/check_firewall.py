#!/usr/bin/env python3
"""Open-core firewall check (run in CI; fails the build on violation).

Two hard boundaries keep this repo cleanly Apache-2.0 releasable:

  1. No CathSim. CathSim is CC-BY-NC-SA-4.0; importing, depending on, or deriving
     assets from it would contaminate this repo's license. The whole point of
     lumen is to be the clean-room generic solver, so this must never appear.

  2. No patient data outside the open cases. Every committed asset must be procedurally
     generated (provenance == "procedural"), with one exception: the few cases shipped in
     lumen/cases/<id>/, derived from public, de-identified scans whose licence allows
     redistribution (CC0 / CC BY / CC BY-SA). Each such folder must carry a case.json naming
     that licence, a citation and the source. Anything else patient-derived stays out.

Exit non-zero with a clear message if either boundary is crossed.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BANNED = ("cathsim",)
OPEN_LICENCES = ("CC0", "CC BY 4.0", "CC BY 3.0", "CC BY-SA 4.0")
CASES = ROOT / "lumen" / "cases"
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


# the files a shipped case is made of; nothing else in a case folder gets its card's approval
CASE_FILES = {"asset.json.gz": "imported(scan)", "names.json": None, "case.json": None,
              "surface.npz": None}
# derived geometry that carries no provenance field: allowed only as a file of an open case
GEOMETRY_SUFFIXES = (".npz", ".stl", ".vtp", ".vtk", ".obj", ".ply", ".nii", ".nii.gz",
                     ".nrrd", ".mha", ".mhd", ".dcm")


def _open_case_problem(p: pathlib.Path, prov: str | None = None) -> str | None:
    """None when ``p`` is one of the files of a shipped open case (with the provenance such a
    file has) whose card names an open licence, a citation and a source; otherwise why not."""
    try:
        rel = p.resolve().relative_to(CASES.resolve())
    except ValueError:
        return "not in lumen/cases/"
    if len(rel.parts) != 2:
        return "not directly inside a case folder"
    if rel.parts[1] not in CASE_FILES:
        return f"not one of the files of a case ({', '.join(sorted(CASE_FILES))})"
    if prov is not None and prov != CASE_FILES[rel.parts[1]]:
        return f"a case's {rel.parts[1]} holds {CASE_FILES[rel.parts[1]]!r} geometry only"
    card_path = CASES / rel.parts[0] / "case.json"
    try:
        card = json.loads(card_path.read_text())
    except (ValueError, OSError):
        return f"no readable {card_path.relative_to(ROOT)}"
    if card.get("licence") not in OPEN_LICENCES:
        return f"licence {card.get('licence')!r} is not one of {', '.join(OPEN_LICENCES)}"
    for key in ("citation", "source_url"):
        if not str(card.get(key) or "").strip():
            return f"case.json has no {key}"
    return None


def _read_json(p: pathlib.Path):
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def check_provenance() -> list[str]:
    bad = []
    # scan EVERY .json and .json.gz anywhere in the repo (not a fixed allow-list of dirs)
    for p in [*ROOT.rglob("*.json"), *ROOT.rglob("*.json.gz")]:
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            d = _read_json(p)
        except (ValueError, OSError, EOFError):
            continue
        for prov in _provenances(d):
            if prov != "procedural":
                why = _open_case_problem(p, prov)
                if why is not None:
                    bad.append(f"  non-procedural asset committed: {p.relative_to(ROOT)} "
                               f"(provenance={prov!r}; {why})")
                break                                      # one report per file is enough
    return bad


def check_geometry_files() -> list[str]:
    """Surface and image files carry no provenance field, so they are allowed only as the
    files of an open case: anywhere else they could be patient-derived and nothing would say."""
    bad = []
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
            continue
        if not p.name.lower().endswith(GEOMETRY_SUFFIXES):
            continue
        why = _open_case_problem(p)
        if why is not None:
            bad.append(f"  geometry file outside the open cases: {p.relative_to(ROOT)} ({why})")
    return bad


def main() -> int:
    problems = check_no_banned() + check_provenance() + check_geometry_files()
    if problems:
        print("FIREWALL VIOLATION -- this repo must stay Apache-2.0, with patient-derived "
              "geometry only in openly licensed lumen/cases/:")
        print("\n".join(problems))
        return 1
    print("firewall ok: no CathSim; patient-derived geometry only in openly licensed cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
