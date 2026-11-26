"""``lumen import``: a scan (or a vessel mask) in, a navigable anatomy folder out.

The output folder holds

* ``asset.json``   the anatomy the solver navigates (``lumen.assets.schema.Asset``)
* ``surface.stl``  the vessel surface in patient millimetres, for viewing
* ``names.json``   vessel names: which edges carry which name and the node to aim for
* ``report.json``  how the import went: region, segmenters used, wall error, edge table
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from lumen.assets import scan_import as sc
from lumen.assets.schema import Asset


def name_edges(asset: Asset, volume, min_fraction: float = 0.3) -> dict:
    """Give each edge the name of the label most of its centreline runs through."""
    inv = np.linalg.inv(np.asarray(volume.affine, float))
    data = np.asarray(volume.data)
    shape = np.asarray(data.shape)
    out = {}
    for e in asset.edges:
        p = np.asarray(e.centerline_mm, float)
        ijk = np.rint((inv @ np.c_[p, np.ones(len(p))].T).T[:, :3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        lab = np.zeros(len(p), int)
        lab[ok] = data[tuple(ijk[ok].T)]
        hit = lab[lab > 0]
        if len(hit) < max(1, min_fraction * len(p)):
            continue
        top = int(np.bincount(hit).argmax())
        name = volume.names.get(top, "")
        if name:
            out[e.id] = name
    return out


def _distances(asset: Asset) -> dict:
    """Route length (mm) from the device entry to every node."""
    adj: dict = {}
    for e in asset.edges:
        L = float(e.s_grid[-1])
        adj.setdefault(e.node_a, []).append((e.node_b, L))
        adj.setdefault(e.node_b, []).append((e.node_a, L))
    dist = {asset.device_spawn.node_id: 0.0}
    todo = [asset.device_spawn.node_id]
    for n in todo:
        for m, L in adj.get(n, []):
            if m not in dist:
                dist[m] = dist[n] + L
                todo.append(m)
    return dist


def vessel_table(asset: Asset, edge_names: dict) -> dict:
    """For each named vessel: its edges, and the node at its far end (the natural target)."""
    dist = _distances(asset)
    table: dict = {}
    for e in asset.edges:
        name = edge_names.get(e.id)
        if not name:
            continue
        row = table.setdefault(name, {"edges": [], "target_node": None, "_far": -1.0})
        row["edges"].append(e.id)
        for n in (e.node_a, e.node_b):
            if dist.get(n, -1.0) > row["_far"]:
                row["_far"], row["target_node"] = dist[n], n
    for row in table.values():
        row["route_mm"] = round(row.pop("_far"), 1)
    return table


def resolve_target(names: dict, asset: Asset, target: str) -> str:
    """A node id from a node id or a vessel name (case-insensitive, a unique part is enough)."""
    if target in {n.id for n in asset.nodes}:
        return target
    vessels = names.get("vessels", {})
    want = " ".join(target.lower().replace("_", " ").split())
    hits = [v for v in vessels if v.lower() == want] or [v for v in vessels if want in v.lower()]
    if len(hits) == 1:
        node = vessels[hits[0]].get("target_node")
        if node is None:
            # never fall back to some other leaf: a run aimed at the wrong vessel looks like a result
            raise ValueError(f"vessel {hits[0]!r} has no node the wire can reach from the entry")
        return node
    known = ", ".join(sorted(vessels)) or "none"
    if not hits:
        raise ValueError(f"no vessel or node called {target!r}; named vessels: {known}")
    raise ValueError(f"{target!r} matches several vessels ({', '.join(sorted(hits))}); be more specific")


def load_imported(folder: str | Path) -> tuple[Asset, dict]:
    """Read an import folder back: ``(asset, names)``."""
    folder = Path(folder)
    asset = Asset.load(str(folder / "asset.json"))
    try:
        names = json.loads((folder / "names.json").read_text())
    except OSError:
        names = {"edges": {}, "vessels": {}}
    return asset, names


def _merge(volumes: list):
    """Several label volumes on the same scan grid -> one, with distinct label values."""
    from lumen.assets.scan_import import LabelVolume

    first = volumes[0]
    data = np.zeros(first.data.shape, np.int16)
    names: dict = {}
    for v in volumes:
        if v.data.shape != first.data.shape or not np.allclose(v.affine, first.affine, atol=1e-3):
            raise ValueError("segmenters returned different grids for one scan")
        for val, name in sorted(v.names.items()):
            sel = (np.asarray(v.data) == val) & (data == 0)
            if not sel.any():
                continue                                      # no voxels left: list no vessel
            names[len(names) + 1] = name
            data[sel] = len(names)
    return LabelVolume(data, first.affine, names)


def import_scan(scan_path=None, out_dir="lumen-import", *, mask=None, entry_mm=None,
                series=None, assume_yes: bool = False, ask=None, say=print, keep_work=False) -> dict:
    """Run the whole import and return the report (also written to ``report.json``)."""
    from lumen.assets import surface_import as si
    from lumen.scan import segmenters as sg
    from lumen.scan.io import read_scan
    from lumen.scan.region import BODY, HEAD, detect_region

    if mask is not None and scan_path is not None:
        raise ValueError("give a scan or a vessel mask, not both: with a mask the scan is not used")
    if mask is None and scan_path is None:
        raise ValueError("give a scan, or a vessel mask with --mask")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"input": {"scan": bool(scan_path), "mask": bool(mask)}}
    if mask is not None:
        volume = sc.load_label_volume(mask)
        report["segmentation"] = {"source": "mask supplied by the user"}
    else:
        if scan_path is None:
            raise ValueError("give a scan (DICOM folder, NIfTI or NRRD) or a vessel mask")
        say("Reading the scan ...")
        scan = read_scan(scan_path, series=series)
        region = detect_region(scan)
        shown = {(HEAD,): "the head", (BODY,): "the neck or trunk",
                 (HEAD, BODY): "the head and the neck or trunk"}[region.regions]
        report["scan"] = {"format": scan.source, "modality": scan.modality,
                          "shape": [int(v) for v in scan.data.shape],
                          "spacing_mm": [round(float(v), 4) for v in scan.spacing_mm]}
        report["region"] = {"regions": list(region.regions), "basis": region.basis,
                            "confident": region.confident, "head_mm": round(region.head_mm),
                            "body_mm": round(region.body_mm)}
        why = f"This scan shows {shown}." if region.confident else \
            "Lumen could not tell which part of the body this scan shows, so it will try both models."
        work = out / "work"
        try:
            volumes, used, errors = [], [], []
            needed = sg.for_regions(region.regions)
            missing = [seg for seg in needed if not sg.installed(seg)]
            if not missing:
                say(why)
            # ask for everything this scan needs before any work starts, then install in the
            # background: a second model installs while the first is already finding arteries
            installs, declined = {}, []
            for seg in missing:
                try:
                    sg.agree(seg, why=why, assume_yes=assume_yes, ask=ask)
                except sg.Declined:
                    declined.append(seg.key)
                    if len(declined) == len(needed):
                        raise
                    continue
                installs[seg.key] = sg.BackgroundInstall(seg, sg._install)
            if installs:
                titles = " and ".join(seg.title for seg in missing if seg.key in installs)
                say(f"Installing {titles} in the background (this happens once) ...")
            digest = hashlib.sha256(np.ascontiguousarray(scan.data).tobytes()).hexdigest()
            for seg in needed:
                if seg.key in declined:
                    errors.append(f"{seg.title} was not installed, so {seg.finds} are not included")
                    continue
                if seg.key in installs:
                    try:
                        installs[seg.key].wait()
                    except sg.SegmenterError as exc:
                        # not the same as finding nothing: the user can fix this and run again
                        errors.append(f"{seg.title} could not be installed, so {seg.finds} are not "
                                      f"included; run the import again to retry "
                                      f"({str(exc).splitlines()[0]})")
                        continue
                cached = work / seg.key / "labels.nii.gz"
                stamp = work / seg.key / "scan.sha256"
                try:
                    if cached.exists() and stamp.exists() and stamp.read_text().strip() == f"{digest} {seg.commit}":
                        # a previous import of this same scan finished this model and then failed
                        # later: reuse it. A different scan, or a different model version, is run fresh.
                        labels = sc.load_label_volume(cached)
                        say(f"Reusing the {seg.title} result from the previous attempt ...")
                    elif seg.region == HEAD:
                        cut = region.detail.get("head_from_mm") if BODY in region.regions else None
                        labels = sg.run_head(scan, work / seg.key, say=say, head_from_mm=cut)
                        sc.save_label_volume(labels, cached)
                        stamp.write_text(f"{digest} {seg.commit}")
                    else:
                        labels = sg.run_body(scan, work / seg.key, say=say)
                        sc.save_label_volume(labels, cached)
                        stamp.write_text(f"{digest} {seg.commit}")
                    volumes.append(labels)
                    used.append({"segmenter": seg.title, "repo": seg.repo, "commit": seg.commit,
                                 "licence": seg.licence})
                except sg.SegmenterError as exc:
                    # with two models on one scan, one of them finding nothing is not a failure
                    errors.append(f"{seg.title}: {str(exc).splitlines()[0]}")
            if not volumes:
                raise sg.SegmenterError("could not find the arteries in this scan: " + "; ".join(errors))
            volume = _merge(volumes) if len(volumes) > 1 else volumes[0]
            report["segmentation"] = {"source": "segmenter", "used": used, "notes": errors}
            if volume.data.shape == scan.data.shape and np.allclose(volume.affine, scan.affine, atol=1e-2):
                from lumen.scan.refine import refine_labels

                say("Sharpening the vessel walls against the scan ...")
                refined, sharpened = refine_labels(scan, volume)
                volume = sc.LabelVolume(refined, volume.affine, volume.names)
                report["segmentation"]["sharpened"] = sharpened
            else:
                report["segmentation"]["sharpened"] = {
                    "vessels": {}, "bridged": [],
                    "why": "labels are on a different grid from the scan; walls not sharpened"}
        finally:
            # copies of the scan must not survive a failure; the model outputs may, so a retry
            # after a tree-build error does not re-run the models
            if not keep_work:
                for f in work.rglob("scan.nii.gz"):
                    f.unlink(missing_ok=True)
                if not any(work.rglob("*")):
                    shutil.rmtree(work, ignore_errors=True)
    say("Building the vessel tree ...")
    asset, conv, tris = sc.asset_from_vessel_mask(volume, entry_mm=entry_mm)
    edge_names = name_edges(asset, volume)
    names = {"edges": edge_names, "vessels": vessel_table(asset, edge_names)}
    asset.save(str(out / "asset.json"))
    si.write_stl(out / "surface.stl", tris)
    (out / "names.json").write_text(json.dumps(names, indent=1))
    for row in conv["edges"]:
        row["name"] = edge_names.get(row["id"], "")
    report.update(conv)
    report["vessels"] = names["vessels"]
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def summary_lines(report: dict, out_dir) -> list[str]:
    """A short plain-language account of an import, for the terminal."""
    w = report["wall_error_mm"]["source_to_lumen_mm"]
    lines = [f"Anatomy written to {out_dir}",
             f"  {len(report['edges'])} vessel segments, entry at "
             f"({', '.join(f'{v:.0f}' for v in report['entry_mm'])}) mm",
             (f"  fitted wall vs the segmented surface: median {w['p50']:.2f} mm, "
              f"95% within {w['p95']:.2f} mm" if "p50" in w else
              "  fitted wall vs the segmented surface: no samples to compare")]
    if report.get("vessels"):
        lines.append("  named vessels you can target:")
        for name, row in sorted(report["vessels"].items(), key=lambda kv: kv[1]["route_mm"]):
            lines.append(f"    {name}  (node {row['target_node']}, {row['route_mm']:.0f} mm from entry)")
    else:
        lines.append("  no vessel names (the mask carried none); target a node id from asset.json")
    for note in (report.get("segmentation") or {}).get("notes") or []:
        lines.append(f"  note: {note}")
    return lines


def make_env(folder: str | Path, target: str, **kwargs):
    """A navigation environment on imported anatomy, aimed at a vessel by name or a node id.

    ``kwargs`` go to ``lumen.envs.tree_nav.TreeNavEnv`` (``max_steps``, ``device``, ...).
    """
    from lumen.envs.tree_nav import TreeNavEnv

    asset, names = load_imported(folder)
    return TreeNavEnv(asset, target_node=resolve_target(names, asset, target), **kwargs)
