"""Reusable first-run workflows used by installed CLIs and example scripts."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np


def render_fluoro_example(out="fluoro.png") -> None:
    """Render the canonical biplanar fluoro demo and write preview artifacts."""
    from lumen.sensors import FluoroSensor, write_avi, write_png

    out = Path(out)
    a = np.linspace(0, np.pi / 2, 40)
    wire = np.stack([30 * np.sin(a) + 3 * np.sin(6 * a), 2 * np.cos(3 * a),
                     30 * (1 - np.cos(a))], axis=1)
    vessel = np.stack([30 * np.sin(a), np.zeros_like(a),
                       30 * (1 - np.cos(a))], axis=1)
    sensor = FluoroSensor(mu_device=1.2, res=96, n_samples=260)
    views = sensor.render_biplanar(wire, radius=0.6, contrast_nodes=vessel,
                                   contrast_radius=2.0, mu_contrast=0.16)
    write_png(out, np.flipud(views[0]["image"]))
    stem = out.parent / out.stem
    write_png(stem.parent / f"{stem.name}_lateral.png", np.flipud(views[1]["image"]))
    write_png(stem.parent / f"{stem.name}_device_mask.png",
              np.flipud(views[0]["masks"]["device"].astype(float)))
    write_png(stem.parent / f"{stem.name}_vessel_mask.png",
              np.flipud(views[0]["masks"]["vessel"].astype(float)))
    write_avi(stem.parent / f"{stem.name}_biplanar.avi", [np.flipud(v["image"]) for v in views],
              fps=2)
    tip = views[0]["keypoints"]["tip"]["uv"]
    tip = (tip[0], views[0]["image"].shape[0] - 1 - tip[1])
    print(f"wrote {out}, {stem}_lateral.png, masks, and {stem}_biplanar.avi; "
          f"tip keypoint view0=({tip[0]:.1f}, {tip[1]:.1f})")


def render_demo_package(out_dir="lumen_demo", *, scene: str = "stenotic",
                        steps: int = 60, size: int = 480, seed: int = 0) -> dict:
    """Write a compact demo media bundle and manifest.

    This intentionally composes existing first-run renderers instead of carrying a
    separate social-video renderer. The output is enough for a quick launch clip:
    a navigation animation/poster, a synthetic fluoro frame/masks, and a manifest
    with the navigation safety metrics needed to describe what was generated.
    """
    from lumen.viz import play

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nav = play(scene=scene, policy="forward", steps=steps, seed=seed,
               size=size, out=str(out / "navigation"))
    with contextlib.redirect_stdout(io.StringIO()):
        render_fluoro_example(str(out / "fluoro.png"))
    files = {
        "navigation_video": out / "navigation.avi",
        "navigation_poster": out / "navigation.png",
        "fluoro_ap": out / "fluoro.png",
    }
    checks = {name: path.is_file() and path.stat().st_size > 0 for name, path in files.items()}
    manifest = {
        "ok": bool(all(checks.values()) and nav["safe"]),
        "scene": scene,
        "steps_requested": int(steps),
        "seed": int(seed),
        "size": int(size),
        "navigation": nav,
        "checks": checks,
        "media": {name: path.name for name, path in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_demo_package(demo_dir="lumen_demo") -> dict:
    """Verify a demo bundle written by :func:`render_demo_package`."""
    root = Path(demo_dir)
    manifest_path = root / "manifest.json"
    problems: list[str] = []
    if not manifest_path.is_file():
        return {"ok": False, "problems": [f"missing {manifest_path}"], "checks": {}}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "problems": [f"invalid JSON in {manifest_path}: {e}"],
            "checks": {},
            "manifest": str(manifest_path),
        }
    media = manifest.get("media", {})
    checks = {}
    for name, rel in sorted(media.items()):
        path = root / str(rel)
        ok = path.is_file() and path.stat().st_size > 0
        checks[name] = ok
        if not ok:
            problems.append(f"missing or empty media: {rel}")
    nav = manifest.get("navigation", {})
    if not isinstance(nav, dict):
        problems.append("manifest navigation summary is not an object")
    elif not nav.get("safe", False):
        problems.append("navigation rollout is not marked safe")
    if not manifest.get("ok", False):
        problems.append("manifest ok flag is false")
    return {
        "ok": not problems,
        "problems": problems,
        "checks": checks,
        "manifest": str(manifest_path),
    }


def _display01(frame):
    arr = np.asarray(frame, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _label_overlay(frame, device_mask, vessel_mask):
    gray = _display01(frame)
    rgb = np.repeat((0.55 * gray)[..., None], 3, axis=2)
    if vessel_mask is not None:
        vessel = np.asarray(vessel_mask, bool)
        rgb[vessel, 1] = 0.85
    if device_mask is not None:
        device = np.asarray(device_mask, bool)
        rgb[device, 0] = 1.0
        rgb[device, 1:] *= 0.2
    return rgb


def _display_scale(frame, min_height: int = 192):
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return arr
    h = int(arr.shape[0])
    if h <= 0 or h >= min_height:
        return arr
    scale = int(np.ceil(min_height / h))
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def write_preview_sheet(ep, root: Path) -> tuple[Path, Path, Path | None]:
    """Write visual QA previews for one captured case bundle."""
    from lumen.sensors import write_png

    obs_steps = [s for s in ep.steps if s.obs_ref]
    if not obs_steps:
        raise ValueError("episode has no observation sidecars to preview")
    picks = [0, len(obs_steps) // 2, len(obs_steps) - 1]
    frames = [obs_steps[i].load_obs(root) for i in picks]
    preview = root / "preview.png"
    sheet = root / "preview_contact_sheet.png"
    write_png(preview, _display_scale(frames[0]))
    write_png(sheet, _display_scale(np.concatenate(frames, axis=1)))
    masks = [obs_steps[i].load_annotation(root, "device_mask") for i in picks]
    mask_sheet = None
    if all(m is not None for m in masks):
        mask_sheet = root / "device_mask_contact_sheet.png"
        write_png(mask_sheet, _display_scale(np.concatenate([m.astype(float) for m in masks], axis=1)))
    vessel_masks = [obs_steps[i].load_annotation(root, "vessel_mask") for i in picks]
    if all(m is not None for m in vessel_masks):
        write_png(root / "vessel_mask_contact_sheet.png",
                  _display_scale(np.concatenate([m.astype(float) for m in vessel_masks], axis=1)))
    if all(m is not None for m in masks) or all(m is not None for m in vessel_masks):
        overlays = [_label_overlay(frame, dev, vessel)
                    for frame, dev, vessel in zip(frames, masks, vessel_masks)]
        write_png(root / "label_overlay_contact_sheet.png",
                  _display_scale(np.concatenate(overlays, axis=1)))
    return preview, sheet, mask_sheet


def capture_examples(out_dir="episodes") -> None:
    """Capture the canonical procedural case-bundle corpus."""
    from lumen.assets import procedural
    from lumen.data import CaseBundle, Episode, rollout_episode, validate
    from lumen.sensors import FluoroSensor, LuminalCamera

    out_root = Path(out_dir)
    sensor = FluoroSensor(res=32, nu=64, nv=64, n_samples=96)
    cases = {
        "straight_fluoro": dict(asset=procedural.straight_tube(80.0, 2.0), sensor=sensor),
        "stenosis_fluoro": dict(asset=procedural.stenotic_tube(80.0, 2.0, severity=0.6), sensor=sensor),
        "straight_luminal": dict(asset=procedural.straight_tube(80.0, 4.0),
                                 sensor=LuminalCamera(nu=64, nv=64), modality="luminal"),
    }
    for name, kw in cases.items():
        path = out_root / name
        print(f"capturing {name:18s} -> {path}", flush=True)
        ep = rollout_episode(max_steps=30, asset_ref=f"{name}.asset.json", label=name,
                             notes={"case": name, "true_C10": 4000.0}, **kw)
        validate(ep)
        ep.save(path)
        back = Episode.load(path)
        validate(back, root=path)
        bundle = CaseBundle.load(path)
        obs0 = back.steps[0].load_obs(path)
        preview, sheet, mask_sheet = write_preview_sheet(back, path)
        tip_ok = back.outcome.metrics["tip_target"]["success"]
        wall_risk = back.outcome.metrics["wall_safety"]["perforation_risk"]
        mask_msg = f"  mask_sheet={mask_sheet}" if mask_sheet else ""
        vessel_sheet = path / "vessel_mask_contact_sheet.png"
        vessel_msg = f"  vessel_sheet={vessel_sheet}" if vessel_sheet.exists() else ""
        overlay_sheet = path / "label_overlay_contact_sheet.png"
        overlay_msg = f"  overlay={overlay_sheet}" if overlay_sheet.exists() else ""
        print(f"{name:18s}  steps={back.outcome.steps:2d}  success={back.outcome.success!s:5s}  "
              f"final_dist={back.outcome.final_dist:6.2f}  obs{obs0.shape}  "
              f"calib={bundle.calibration['type']}  tip_target={tip_ok!s:5s}  "
              f"wall_risk={wall_risk!s:5s}  preview={preview}  sheet={sheet}"
              f"{mask_msg}{vessel_msg}{overlay_msg}",
              flush=True)
