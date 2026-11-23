"""Clip 6: replayable episode. The hero navigation episode is captured into Lumen's
Episode format (lumen.data.schema: manifest + per-step sidecars for the fluoroscopy
image, the solver's rod node positions, the device mask and the tip keypoint), written
to disk, loaded back through EpisodeDataset/replay(), and scrubbed: play, rewind, pause,
play. Every displayed image and 3-D state comes from the reloaded sidecars, not memory.

Two checks are stamped into the manifest and on screen:
  * reload fidelity: every reloaded image and node array is byte-identical to capture;
  * re-simulation: replaying the recorded actions through a fresh env from the same seed
    reproduces the recorded rod node positions (max deviation in mm).

    python -m tools.xseries.clip_replay
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (FluoroCam, View3D, arch_rollout, corner_tag, encode, env_versions,
                                     font, git_dirty, git_sha, gray_to_rgb, label, out_root, probe,
                                     sha256, write_manifest)

LZ = Image.Resampling.LANCZOS
TARGET, INS = "lcca_out", 0.45


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def capture(out: Path, every: int, nu: int, seed: int):
    from lumen.data.schema import Episode, EpisodeMeta, Outcome, Step
    asset, target, frames, meta, rpath = arch_rollout(TARGET, INS, seed=seed)
    cam = FluoroCam(asset, axis=(0.0, 1.0, 0.0), nu=nu, fov=1.0)
    idx = list(range(0, len(frames), every))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    steps, hashes = [], []
    for j, fi in enumerate(idx):
        fr = frames[fi]
        img = np.flipud(cam.image(fr.extra["wire"], seed=fi)).copy()
        m = np.flipud(cam.masks(fr.extra["wire"])["device"]).astype(np.uint8)
        kp = cam.project(fr.nodes[-1])
        tip_px = ([float(kp["uv"][0]) + 0.5, float(nu - 1 - kp["uv"][1]) + 0.5]   # stored image (row-up) coords
                  if kp["present"] else None)
        tag = f"{j:05d}"
        st = Step(t=round(fr.info["step"] * meta["dt_s_per_step"], 6),
                  action={"insertion": 1.0 if fi else 0.0, "twist": 0.0},
                  kinematics={"tip_mm": [float(v) for v in fr.nodes[-1]], "sim_step": int(fr.info["step"]),
                              "max_pen_mm": float(fr.info.get("max_pen", 0.0)),
                              "node_positions_ref": f"nodes_{tag}.npy"},
                  annotations={"tip_px": tip_px, "device_mask_ref": f"mask_{tag}.npy",
                               "wire_polyline_ref": f"wire_{tag}.npy"},
                  obs_modality="fluoro", obs_ref=f"fluoro_{tag}.npy")
        st.obs = img
        st.node_positions = np.asarray(fr.nodes, np.float64)
        st.annotation_arrays = {"device_mask": m, "wire_polyline": np.asarray(fr.extra["wire"], np.float64)}
        steps.append(st)
        hashes.append((_h(img), _h(st.node_positions)))
    last = frames[-1].info
    ep = Episode(meta=EpisodeMeta(asset_ref="asset.json", dt=meta["dt_s_per_step"] * every,
                                  device={"radius_mm": meta["device_radius_mm"], "rod_bodies": meta["rod_bodies"]},
                                  sensor={"modality": "fluoro", **cam.params},
                                  labels={"task": "branch_navigation", "target": TARGET, "env": "TreeNavEnv"},
                                  notes={"seed": seed, "max_insertion_mm_per_step": INS, "record_every": every},
                                  provenance="procedural"),
                 steps=steps,
                 outcome=Outcome(success=bool(last.get("success")), final_dist=float(last.get("dist", 0.0)),
                                 steps=len(steps), label="hero lcca"),
                 asset=asset)
    root = out / "episode" / f"hero-lcca-seed{seed}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    ep.save(str(root))
    return asset, target, frames, meta, rpath, idx, hashes, root


def reload_and_verify(root: Path, hashes):
    from lumen.data.replay import EpisodeDataset, replay
    ds = EpisodeDataset(str(root.parent))
    ep = ds[0]
    assert not isinstance(ep, list)
    loaded = []
    ok_img = ok_nodes = 0
    for (t, action, kin, obs, ann), st, (hi, hn) in zip(replay(ep, include_annotations=True), ep.steps, hashes):
        nodes = st.load_nodes(getattr(ep, "root", str(root)))
        ok_img += _h(obs) == hi
        ok_nodes += _h(nodes) == hn
        loaded.append({"t": t, "obs": obs, "nodes": nodes, "wire": ann["wire_polyline"],
                       "mask": ann["device_mask"], "tip_px": ann["tip_px"], "kin": kin,
                       "obs_ref": st.obs_ref})
    return loaded, {"episodes_in_dataset": len(ds), "steps": len(loaded),
                    "obs_byte_identical": int(ok_img), "nodes_byte_identical": int(ok_nodes)}


def resimulate(frames, idx, seed):
    """Replay the recorded actions through a fresh env from the same seed."""
    from lumen.envs.tree_nav import TreeNavEnv
    from tools.xseries.anatomy import femoral_arch
    env = TreeNavEnv(femoral_arch(), target_node=TARGET, max_steps=1500, max_insertion=INS)
    env.reset(seed=seed)
    got = {0: np.asarray(env.sim.body_positions(), float).copy()}
    for k in range(1, len(frames)):
        env.step(np.array([1.0, 0.0]))
        got[k] = np.asarray(env.sim.body_positions(), float).copy()
    dev = [float(np.abs(got[fi] - frames[fi].nodes).max()) for fi in idx]
    return {"max_node_deviation_mm": max(dev), "steps_compared": len(dev)}


def schedule(n, hold):
    """play -> pause -> rewind (fast) -> pause -> play to end -> hold."""
    a = int(0.62 * (n - 1))
    b = int(0.30 * (n - 1))
    seq = [(k, "PLAY") for k in range(0, a + 1)]
    seq += [(a, "PAUSE")] * 24
    seq += [(k, "REWIND") for k in range(a, b - 1, -3)]
    seq += [(b, "PAUSE")] * 24
    seq += [(k, "PLAY") for k in range(b, n)]
    seq += [(n - 1, "END")] * hold
    return seq


def timeline(w, h, k, n, mode, t, ref, checks):
    img = Image.new("RGB", (w, h), (11, 14, 19))
    d = ImageDraw.Draw(img)
    x0, x1, y = 40, w - 40, 30
    d.rounded_rectangle((x0, y - 5, x1, y + 5), 5, fill=(40, 48, 60))
    px = x0 + (x1 - x0) * k / max(n - 1, 1)
    d.rounded_rectangle((x0, y - 5, px, y + 5), 5, fill=(92, 232, 238))
    d.ellipse((px - 11, y - 11, px + 11, y + 11), fill=(235, 240, 245))
    icon = {"PLAY": "play", "PAUSE": "pause", "REWIND": "rewind", "END": "end"}[mode]
    col = {"PLAY": (112, 211, 141), "PAUSE": (247, 181, 64), "REWIND": (247, 181, 64), "END": (190, 198, 208)}[mode]
    d.text((x0, y + 26), f"{icon}   step {k + 1:3d} / {n}   t = {t:5.2f} s", font=font(22), fill=col)
    d.text((x1, y + 26), f"loaded from disk: {ref}", font=font(18), fill=(130, 140, 152), anchor="ra")
    d.text((x0, y + 62), checks, font=font(17), fill=(160, 168, 180))
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="record every Nth solver step")
    ap.add_argument("--nu", type=int, default=720)
    ap.add_argument("--hold", type=int, default=45)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    if a.every <= 0:
        raise ValueError("--every must be positive")
    only = [int(x) for x in a.only.split(",") if x]
    out = Path(a.out) if a.out else out_root() / "clip6-replay"
    out.mkdir(parents=True, exist_ok=True)
    asset, target, frames, meta, rpath, idx, hashes, root = capture(out, a.every, a.nu, a.seed)
    loaded, rel = reload_and_verify(root, hashes)
    sim = resimulate(frames, idx, a.seed)
    ep_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    dev = sim["max_node_deviation_mm"]
    dev_s = "0 mm (bit-exact)" if dev == 0.0 else f"{dev:.1e} mm"
    checks = (f"reload: {rel['obs_byte_identical']}/{rel['steps']} images and {rel['nodes_byte_identical']}/"
              f"{rel['steps']} states byte-identical     re-simulated from recorded actions: max node deviation "
              f"{dev_s}")
    print(rel, sim, flush=True)
    view = View3D(asset, size_px=(960, 900), elev=14.0, zoom=1.45)
    view_sq = View3D(asset, size_px=(540, 640), elev=14.0, zoom=1.75)
    n = len(loaded)
    seq = schedule(n, a.hold)
    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for i, (k, mode) in enumerate(seq):
        if only and i not in only:
            continue
        s = loaded[k]
        g = gray_to_rgb(s["obs"])
        tip = s["tip_px"]
        fl = Image.fromarray(g).resize((900, 900), LZ)
        sc = 900 / a.nu
        dd = ImageDraw.Draw(fl)
        if tip is not None:
            u, v = tip[0] * sc, tip[1] * sc
            dd.ellipse((u - 14, v - 14, u + 14, v + 14), outline=(247, 181, 64), width=3)
        v3 = Image.fromarray(view.render(s["wire"], target=target))
        L = Image.new("RGB", (1920, 1080), (7, 9, 13))
        L.paste(fl, (30, 58)); L.paste(v3, (960, 30))
        label(L, "Replayed image (fluoro sidecar)", (30, 16), 24)
        label(L, "Paired solver state (node sidecar), 3-D", (984, 16), 24)
        label(L, "tip keypoint", (52, 920), 18, (247, 181, 64), bg=(10, 13, 18))
        L.paste(timeline(1920, 130, k, n, mode, s["t"], s["obs_ref"], checks), (0, 950))
        corner_tag(L, top=True)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        S = Image.new("RGB", (1080, 1080), (7, 9, 13))
        S.paste(fl.resize((520, 520), LZ), (10, 120))
        S.paste(Image.fromarray(view_sq.render(s["wire"], target=target)).crop((0, 60, 540, 600)), (540, 110))
        label(S, "Replayed image", (14, 70), 24); label(S, "Paired solver state", (556, 70), 24)
        tl = timeline(1920, 130, k, n, mode, s["t"], s["obs_ref"], "")
        S.paste(tl.crop((0, 0, 1920, 50)).resize((1080, 28)), (0, 704))
        label(S, f"{mode.lower()}   step {k + 1} / {n}   t = {s['t']:.2f} s", (40, 752), 26,
              (112, 211, 141) if mode == "PLAY" else (247, 181, 64))
        label(S, f"reload: {rel['obs_byte_identical']}/{n} images, {rel['nodes_byte_identical']}/{n} states "
                 "byte-identical", (40, 806), 21, (160, 168, 180))
        label(S, f"re-simulated: max node deviation {dev_s}", (40, 842), 21,
              (160, 168, 180))
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if i % 60 == 0:
            print(f"frame {i}/{len(seq)}", flush=True)
    if only:
        print("preview written")
        return
    rec = {"clip": "6-replay",
           "description": "The hero navigation episode (descending aorta to left common carotid, constant advance "
           "0.45 mm/step) captured into Lumen's Episode format: a JSON manifest plus per-step sidecars for the "
           "synthetic fluoroscopy image, the Newton rod node positions, the device mask, the display wire "
           "polyline, and the tip keypoint. The episode is saved, reloaded through EpisodeDataset and "
           "lumen.data.replay.replay(include_annotations=True), and scrubbed (play, pause, rewind, play). Every "
           "displayed frame is read back from disk. Checks: reloaded arrays are byte-identical to capture; "
           "re-simulating the recorded actions from the same seed in a fresh env reproduces the recorded rod "
           "states (max deviation reported).",
           "reload_check": rel, "resimulation_check": sim, "episode_dir": str(root), "episode_bytes": ep_bytes,
           "record_every_solver_steps": a.every, "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_replay " + " ".join(sys.argv[1:] if argv is None else argv),
           "rollout_file": str(rpath), "sim": meta, "versions": env_versions(), "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip6-replay-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
