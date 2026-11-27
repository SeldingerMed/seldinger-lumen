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


# Arteries a wire is brought up through. Where the scan cut one of them, the cut end is a way in.
ACCESS_VESSELS = ("internal carotid", "common carotid", "vertebral")


def access_nodes(asset: Asset, edge_names: dict, caudal_mm: float = 12.0) -> list[str]:
    """Where a wire can come into the scanned anatomy: the device entry first, then every free
    end of a carotid or vertebral artery at the bottom of the scanned region, where the scan cut
    the artery on its way up from the heart. A head scan holds both carotids and often the
    vertebrals; an operator reaches the right middle cerebral artery up the right carotid, not
    across the circle of Willis from the left. A short labelled spur inside the head, or a
    carotid cut at the top of a body scan (blood leaves there), is not a way in. Needs a patient
    frame (+z toward the head); other frames keep only the device entry."""
    deg: dict = {}
    for e in asset.edges:
        for n in (e.node_a, e.node_b):
            deg[n] = deg.get(n, 0) + 1
    out = [asset.device_spawn.node_id]
    frame = str(getattr(asset.frame, "name", "")).lower()
    if "lps" not in frame and "ras" not in frame:
        return out
    z_low = min(float(np.min(np.asarray(e.centerline_mm, float)[:, 2])) for e in asset.edges)
    pos = {n.id: np.asarray(n.position_mm, float) for n in asset.nodes}
    for e in asset.edges:
        name = " ".join(str(edge_names.get(e.id, "")).lower().replace("_", " ").split())
        if not any(w in name for w in ACCESS_VESSELS):
            continue
        for n in (e.node_a, e.node_b):
            if deg.get(n) == 1 and n not in out and pos[n][2] - z_low < caudal_mm:
                out.append(n)
    return out


def _shortest(asset: Asset, sources=None) -> tuple[dict, dict, dict]:
    """Route length (mm) to every node from the nearest of ``sources`` (default: the device
    entry), which source that is, and the edge each node is reached by."""
    import heapq

    adj: dict = {}
    for e in asset.edges:
        L = float(e.s_grid[-1])
        adj.setdefault(e.node_a, []).append((e.node_b, L, e.id))
        adj.setdefault(e.node_b, []).append((e.node_a, L, e.id))
    sources = list(sources) if sources else [asset.device_spawn.node_id]
    dist, src, via = {}, {}, {}
    heap = [(0.0, i, n, None) for i, n in enumerate(sources)]
    heapq.heapify(heap)
    while heap:
        d, i, n, e_in = heapq.heappop(heap)
        if n in dist:
            continue
        dist[n], src[n], via[n] = d, sources[i], e_in
        for m, L, eid in adj.get(n, []):
            if m not in dist:
                heapq.heappush(heap, (d + L, i, m, eid))
    return dist, src, via


def _distances(asset: Asset, sources=None) -> tuple[dict, dict]:
    """Route length (mm) to every node from the nearest of ``sources`` (default: the device
    entry), and which source that is."""
    dist, src, _ = _shortest(asset, sources)
    return dist, src


def vessel_table(asset: Asset, edge_names: dict, access: list | None = None) -> dict:
    """For each named vessel: its edges, a target node at its far end and the way in
    (``start_node``) a wire is brought up from, with the route length from there. The route
    must run through the vessel itself: from each way in, the target is the vessel's farthest
    end reached through one of its own edges, and the way in with the shortest such route is
    taken. (The nearest end of the nearest way in can lie across the circle of Willis: the
    left anterior cerebral artery's far end reached up the right carotid, through the right
    one.)"""
    access = list(access or access_nodes(asset, edge_names))
    trees = {a: _shortest(asset, [a]) for a in access}
    nearest, nearest_src = _distances(asset, access)
    edge_ends = {e.id: (e.node_a, e.node_b) for e in asset.edges}
    table: dict = {}
    for e in asset.edges:
        name = edge_names.get(e.id)
        if name:
            table.setdefault(name, {"edges": []})["edges"].append(e.id)
    for row in table.values():
        own = set(row["edges"])
        ends = {n for eid in own for n in edge_ends[eid]}
        best = None                                    # (route length, way in, target)
        for a, (dist, _, via) in trees.items():
            far = None
            for n in ends:
                if n not in dist or dist[n] <= 0.0:
                    continue
                # does the way from a to n pass through one of the vessel's own edges?
                m, through = n, False
                while via.get(m) is not None and not through:
                    eid = via[m]
                    through = eid in own
                    u, v = edge_ends[eid]
                    m = u if v == m else v
                if through and (far is None or dist[n] > dist[far]):
                    far = n
            if far is not None and (best is None or dist[far] < best[0]):
                best = (dist[far], a, far)
        if best is None:                               # not reached through itself: nearest end
            far = max((n for n in ends if n in nearest), key=lambda n: nearest[n], default=None)
            best = (nearest.get(far, -1.0), nearest_src.get(far), far)
        row["target_node"] = best[2]
        row["route_mm"] = round(best[0], 1)
        row["start_node"] = best[1]
    return table


def route_start(asset: Asset, names: dict, target_node: str) -> str:
    """The way in nearest ``target_node`` (the device entry when the scan has only one). A
    target that is itself a way in is reached from another one."""
    access = names.get("access") or [asset.device_spawn.node_id]
    others = [a for a in access if a != target_node]
    _, src = _distances(asset, others or access)
    if target_node not in src:
        raise ValueError(f"no way in reaches {target_node!r} (ways in: {', '.join(access)})")
    return src[target_node]


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
    packed = folder / "asset.json.gz"
    if not (folder / "asset.json").exists() and packed.exists():   # a shipped case
        import gzip

        asset = Asset.from_dict(json.loads(gzip.decompress(packed.read_bytes())))
    else:
        asset = Asset.load(str(folder / "asset.json"))
    try:
        names = json.loads((folder / "names.json").read_text())
    except (OSError, ValueError):                         # missing or damaged: no names
        return asset, {"edges": {}, "vessels": {}}
    if not isinstance(names, dict):                       # valid JSON of the wrong shape
        return asset, {"edges": {}, "vessels": {}}
    if "access" not in names:          # written before scans had several ways in: work them out
        edges = names.get("edges") if isinstance(names.get("edges"), dict) else {}
        names["access"] = access_nodes(asset, edges)
        worked = vessel_table(asset, edges, names["access"])
        saved = names.get("vessels") if isinstance(names.get("vessels"), dict) else {}
        # rows saved by hand keep their targets and their names; they gain a way in
        for name, row in saved.items():
            if isinstance(row, dict) and row.get("target_node"):
                fresh = dict(worked.get(name, {}))
                fresh.update(row)
                try:
                    fresh["start_node"] = route_start(asset, names, row["target_node"])
                except ValueError:
                    fresh["start_node"] = asset.device_spawn.node_id
                worked[name] = fresh
        names["vessels"] = worked
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
                series=None, assume_yes: bool = False, ask=None, say=print, keep_work=False,
                modality: str | None = None) -> dict:
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
    report: dict = {"input": {"scan": bool(scan_path), "mask": bool(mask)}}
    if mask is not None:
        volume = sc.load_label_volume(mask)
        out.mkdir(parents=True, exist_ok=True)             # only once the input has been read
        report["segmentation"] = {"source": "mask supplied by the user"}
    else:
        if scan_path is None:
            raise ValueError("give a scan (DICOM folder, NIfTI or NRRD) or a vessel mask")
        say("Reading the scan ...")
        scan = read_scan(scan_path, series=series, modality=modality)
        out.mkdir(parents=True, exist_ok=True)
        region = detect_region(scan)
        shown = {(HEAD,): "the head", (BODY,): "the neck or trunk",
                 (HEAD, BODY): "the head and the neck or trunk"}[region.regions]
        report["scan"] = {"format": scan.source, "modality": scan.modality,
                          "shape": [int(v) for v in scan.data.shape],
                          "spacing_mm": [round(float(v), 4) for v in scan.spacing_mm]}
        if scan.shifted_by:
            report["scan"]["shifted_to_hounsfield_by"] = scan.shifted_by
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
            h = hashlib.sha256(np.ascontiguousarray(scan.data).tobytes())
            h.update(np.ascontiguousarray(scan.affine, dtype=np.float64).tobytes())  # same voxels, other placement
            digest = h.hexdigest()
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
    # an entry chosen on the command line is the way in; otherwise every cut carotid or
    # vertebral end at the bottom of a head scan is one
    access = [asset.device_spawn.node_id] if entry_mm is not None else access_nodes(asset, edge_names)
    names = {"edges": edge_names, "access": access, "vessels": vessel_table(asset, edge_names, access)}
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
            if row.get("target_node") is None or row["route_mm"] < 0:
                lines.append(f"    {name}  (not reachable from the entry)")
            else:
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
    node = resolve_target(names, asset, target)
    if "start_node" not in kwargs:
        # a vessel asked for by name is reached the way its row says (through the vessel);
        # a node id from the nearest way in
        row = next((r for r in names.get("vessels", {}).values() if isinstance(r, dict)
                    and r.get("target_node") == node and r.get("start_node") not in (None, node)),
                   None)
        by_name = target not in {n.id for n in asset.nodes}
        kwargs["start_node"] = (row["start_node"] if by_name and row
                                else route_start(asset, names, node))
    return TreeNavEnv(asset, target_node=node, **kwargs)
