"""Pack an ``lumen import`` output folder as a case that ships with Lumen.

    python tools/pack_case.py IMPORT_FOLDER CASE_ID --title ... --region ... --modality ... \
        --licence "CC BY 4.0" --citation "..." --source-url ... [--source-doi ...] [--scan ...]

Writes ``lumen/cases/CASE_ID/`` (see ``lumen.cases``). Only derived geometry goes in: the
lumen model, the vessel names, and a simplified vessel surface. The scan stays out.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_TRIANGLES = 120_000
ALLOWED_LICENCES = ("CC0", "CC BY 4.0", "CC BY-SA 4.0", "CC BY 3.0")


def _round(obj, nd=3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    return obj


def pack(folder: Path, case_id: str, card: dict, dest_root: Path | None = None) -> Path:
    from lumen.assets.mesh import save_npz, simplify_to, weld
    from lumen.assets.surface_import import read_stl
    from lumen.scan.pipeline import load_imported

    if card.get("licence") not in ALLOWED_LICENCES:
        raise ValueError(f"licence {card.get('licence')!r} does not allow shipping derived data "
                         f"(allowed: {', '.join(ALLOWED_LICENCES)})")
    asset, names = load_imported(folder)
    dest = (dest_root or ROOT / "lumen" / "cases") / case_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    raw = json.dumps(_round(asset.to_dict()), separators=(",", ":")).encode()
    (dest / "asset.json.gz").write_bytes(gzip.compress(raw, 9, mtime=0))
    (dest / "names.json").write_text(json.dumps(names, indent=1))
    v, f = weld(np.asarray(read_stl(folder / "surface.stl"), np.float32))
    v, f = simplify_to(v, f, MAX_TRIANGLES)
    save_npz(dest / "surface.npz", v, f)
    report = json.loads((folder / "report.json").read_text())
    vessels = sorted(n for n, row in names.get("vessels", {}).items() if row.get("target_node"))
    wall = report.get("wall_error_mm", {}).get("source_to_lumen_mm", {})
    full = {**card, "vessels": vessels,
            "derived": {"by": "lumen import", "segments": len(asset.edges),
                        "surface_triangles": int(len(f)),
                        "wall_fit_p95_mm": round(float(wall.get("p95", float("nan"))), 2),
                        "region": report.get("region", {}).get("regions"),
                        "segmenter": [u.get("segmenter") for u in
                                      (report.get("segmentation") or {}).get("used") or []]}}
    (dest / "case.json").write_text(json.dumps(full, indent=1))
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description="Pack an import folder as a shipped Lumen case.")
    p.add_argument("folder", type=Path)
    p.add_argument("case_id")
    for k in ("title", "region", "modality", "licence", "citation", "source_url"):
        p.add_argument("--" + k.replace("_", "-"), required=True)
    p.add_argument("--source-doi")
    p.add_argument("--scan", help="which scan in the source dataset")
    a = p.parse_args(argv)
    card = {"title": a.title, "region": a.region, "modality": a.modality, "licence": a.licence,
            "citation": a.citation, "source_url": a.source_url}
    if a.source_doi:
        card["source_doi"] = a.source_doi
    if a.scan:
        card["scan"] = a.scan
    dest = pack(a.folder, a.case_id, card)
    size = sum(f.stat().st_size for f in dest.iterdir())
    print(f"{dest} ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
