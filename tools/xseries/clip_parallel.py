"""Clip 5: batched parallel environments. One Newton model holds E independent guidewires
in the same procedural aortic arch (shared vessel, wire-vs-wall contact only), each driven
by its own action stream; a single solver call advances all of them. The grid shows every
environment live from one batched sim; the panel shows aggregate env-steps/s measured on
this machine with lumen.newton.throughput.measure_throughput at several batch sizes, plus
the live rate of the capture itself (solver time only, rendering excluded).

    python -m tools.xseries.clip_parallel
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (corner_tag, encode, env_versions, font, git_dirty, git_sha, label,
                                     out_root, probe, sha256, shaft_polyline, write_manifest)

TARGET = "lcca_out"
SUB = 4
DT = 5e-3 * SUB
MAX_INS, MAX_TW = 0.9, 1.0


def hardware():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        chip = platform.processor()
    import warp as wp
    dev = str(wp.get_preferred_device())
    return {"chip": chip or platform.machine(), "warp_device": dev,
            "label": f"{chip}, Warp {'CPU' if dev == 'cpu' else dev} backend"}


def scene():
    from lumen.envs.tree_nav import TreeNavEnv
    from tools.xseries.anatomy import femoral_arch
    asset = femoral_arch()
    env = TreeNavEnv(asset, target_node=TARGET, max_insertion=MAX_INS)
    env.reset(seed=0)
    return asset, env


def batched_sim(env, E):
    from lumen.newton.sim import NewtonGuidewireSim
    return NewtonGuidewireSim(np.asarray(env.route_frame.points), env.R, env._device_points(),
                              radius=env.device_radius_mm, kappa=3e3, d_hat=0.3, vbd_iterations=8,
                              device=env.device, tree=env.tree, n_envs=E,
                              route_centerline=np.asarray(env.route_frame.points))


def sweep(env, sizes, steps):
    from lumen.newton.throughput import measure_throughput
    rows = []
    for E in sizes:
        sim = batched_sim(env, E)
        r = measure_throughput(sim, steps=steps, warmup=2, dt=DT, substeps=SUB, insertion=MAX_INS)
        rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        print(rows[-1], flush=True)
        del sim
    return rows


class Proj:
    """Orthographic frontal projection (x right, z up) of the arch into a tw x th tile."""

    def __init__(self, asset, env, tw, th, pad=5):
        pts = np.concatenate([np.asarray(e.centerline_mm, float) for e in asset.edges])
        lo, hi = pts[:, [0, 2]].min(0) - 6, pts[:, [0, 2]].max(0) + 6
        span = hi - lo
        self.s = min((tw - 2 * pad) / span[0], (th - 2 * pad) / span[1])
        self.off = np.array([(tw - span[0] * self.s) / 2, (th - span[1] * self.s) / 2])
        self.lo, self.tw, self.th = lo, tw, th
        self.asset, self.env = asset, env

    @staticmethod
    def aspect(asset):
        pts = np.concatenate([np.asarray(e.centerline_mm, float) for e in asset.edges])
        span = pts[:, [0, 2]].max(0) - pts[:, [0, 2]].min(0) + 12
        return float(span[0] / span[1])

    def uv(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        u = self.off[0] + (p[:, 0] - self.lo[0]) * self.s
        v = self.th - self.off[1] - (p[:, 2] - self.lo[1]) * self.s
        return np.stack([u, v], 1)

    def background(self, scale=1):
        img = Image.new("RGB", (self.tw * scale, self.th * scale), (12, 15, 20))
        d = ImageDraw.Draw(img)
        for e in self.asset.edges:
            P = np.asarray(e.centerline_mm, float)
            Rm = float(np.asarray(e.R, float).mean())
            w = max(2, int(2 * Rm * self.s * scale))
            d.line([tuple(x) for x in self.uv(P) * scale], fill=(150, 66, 66), width=w, joint="curve")
        for e in self.asset.edges:
            P = np.asarray(e.centerline_mm, float)
            Rm = float(np.asarray(e.R, float).mean())
            w = max(1, int(2 * Rm * self.s * scale) - 2 * scale)
            d.line([tuple(x) for x in self.uv(P) * scale], fill=(70, 30, 34), width=w, joint="curve")
        return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--envs", type=int, default=60, help="environments in the displayed batch")
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--sizes", default="1,4,16,64,256")
    ap.add_argument("--bench-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=int, default=45)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    only = [int(x) for x in a.only.split(",") if x]
    out = Path(a.out) if a.out else out_root() / "clip5-batched-envs"
    out.mkdir(parents=True, exist_ok=True)
    hw = hardware()
    asset, env = scene()
    sizes = [int(x) for x in a.sizes.split(",") if x]
    rows = sweep(env, sizes, a.bench_steps) if not only else [
        {"n_envs": s, "env_steps_per_s": 0.0, "ms_per_step": 0.0, "us_per_env_step": 0.0, "device": "cpu"}
        for s in sizes]
    E = a.envs
    sim = batched_sim(env, E)
    rng = np.random.default_rng(a.seed)
    # per-env action streams: an advance rate and a slowly varying twist, both per env
    ins_rate = rng.uniform(0.35, 1.0, E)
    tw = np.zeros(E)
    target = np.asarray(env.route_frame.points[-1], float)
    done = np.zeros(E, bool)                  # an env stops advancing once its tip reaches the target
    states, live, dones = [sim.env_positions().copy()], [], [done.copy()]
    for k in range(a.steps):
        tw = np.clip(0.9 * tw + rng.normal(0, 0.35, E), -1, 1)
        t0 = time.perf_counter()
        sim.step(dt=DT, substeps=SUB, insertion=np.where(done, 0.0, ins_rate * MAX_INS),
                 twist=np.where(done, 0.0, tw * MAX_TW))
        pos = sim.env_positions()             # host read included in the live timing
        live.append(time.perf_counter() - t0)
        done |= np.linalg.norm(pos[:, -1] - target, axis=1) < 2.5
        states.append(pos.copy())
        dones.append(done.copy())
        if not np.isfinite(pos).all():
            raise RuntimeError(f"non-finite state at step {k}")
    live_rate = E * len(live) / sum(live)
    print({"grid_envs": E, "live_env_steps_per_s": round(live_rate, 1)}, flush=True)

    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    asp = Proj.aspect(asset)

    def layout(W, H):        # grid whose tile aspect best matches the arch
        best = None
        for c in range(1, E + 1):
            if E % c:
                continue
            r = E // c
            err = abs(np.log((W / c) / (H / r) / asp))
            if best is None or err < best[0]:
                best = (err, c, r)
        return best[1], best[2]
    cL, rL = layout(1040, 1020)
    cS, rS = layout(1000, 880)
    twL, thL, twS, thS = 1040 // cL, 1020 // rL, 1000 // cS, 880 // rS
    PL, PS = Proj(asset, env, twL, thL), Proj(asset, env, twS, thS)
    bgL, bgS = PL.background(), PS.background()
    seq = list(range(len(states))) + [len(states) - 1] * a.hold
    best = max(r["env_steps_per_s"] for r in rows) or 1.0
    for i, k in enumerate(seq):
        if only and i not in only:
            continue
        pos, dn = states[k], dones[k]
        L = Image.new("RGB", (1920, 1080), (7, 9, 13))
        S = Image.new("RGB", (1080, 1080), (7, 9, 13))
        for e in range(E):
            for P, bg, img, x0, y0, cols, lw in ((PL, bgL, L, 20, 30, cL, 3), (PS, bgS, S, 40, 70, cS, 3)):
                t = bg.copy()
                d = ImageDraw.Draw(t)
                wire = shaft_polyline(env, pos[e])
                uv = P.uv(wire)
                d.line([tuple(x) for x in uv], fill=(110, 240, 245), width=lw)
                tp = P.uv(pos[e][-1])[0]
                if dn[e]:
                    d.ellipse((tp[0] - 4, tp[1] - 4, tp[0] + 4, tp[1] + 4), fill=(112, 211, 141))
                else:
                    d.ellipse((tp[0] - 2, tp[1] - 2, tp[0] + 2, tp[1] + 2), fill=(235, 250, 252))
                img.paste(t, (x0 + (e % cols) * P.tw, y0 + (e // cols) * P.th))
        dL = ImageDraw.Draw(L)
        x0 = 1100
        label(L, f"{E} environments, one batched solver", (x0, 40), 34)
        label(L, "Same procedural aortic arch. Each wire has its own advance", (x0, 96), 20, (160, 168, 180))
        label(L, "rate and twist stream; one Newton call steps all of them.", (x0, 124), 20, (160, 168, 180))
        label(L, f"t = {k * DT:5.2f} s   step {k}", (x0, 176), 24, (190, 198, 208))
        label(L, f"at the left common carotid target: {int(dn.sum())}/{E}", (x0 + 330, 176), 24, (112, 211, 141))
        label(L, "Aggregate throughput, measured on this machine", (x0, 250), 24)
        label(L, hw["label"], (x0, 284), 20, (247, 181, 64))
        bx, by, bw = x0, 330, 760
        for j, r in enumerate(rows):
            yy = by + j * 58
            frac = r["env_steps_per_s"] / best
            dL.rectangle((bx + 120, yy, bx + 120 + int((bw - 300) * frac), yy + 38),
                         fill=(92, 232, 238) if r["n_envs"] == E else (60, 120, 130))
            dL.text((bx + 100, yy + 19), f"{r['n_envs']} env{'s' if r['n_envs'] > 1 else ''}", font=font(20),
                    fill=(190, 198, 208), anchor="rm")
            dL.text((bx + 130 + int((bw - 300) * frac), yy + 19), f"{r['env_steps_per_s']:,.0f} env-steps/s",
                    font=font(20), fill=(235, 240, 245), anchor="lm")
        yb = by + len(rows) * 58 + 20
        if len(rows) > 1 and rows[0]["env_steps_per_s"] > 0:
            gain = best / rows[0]["env_steps_per_s"]
            label(L, f"{gain:.0f}x the single-environment rate from batching alone", (x0, yb), 22, (112, 211, 141))
        label(L, f"this capture: {live_rate:,.0f} env-steps/s live ({E} envs, solver + state readback)",
              (x0, yb + 44), 19, (160, 168, 180))
        label(L, "The same kernels compile to CUDA through NVIDIA Warp.", (x0, yb + 80), 19, (160, 168, 180))
        corner_tag(L, top=False)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        label(S, f"{E} environments, one batched solver", (40, 14), 30)
        label(S, f"{hw['label']}:  {best:,.0f} env-steps/s peak, "
                 f"{max(rows, key=lambda r: r['env_steps_per_s'])['n_envs']} envs", (40, 1072 - 40), 20,
              (247, 181, 64))
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if i % 40 == 0:
            print(f"frame {i}/{len(seq)}", flush=True)
    if only:
        print("preview written")
        return
    rec = {"clip": "5-batched-envs",
           "description": f"{E} independent guidewire environments held in one Newton model (n_envs={E}) in the "
           "procedural femoral-approach aortic arch, sharing the vessel tree (contact is wire-vs-wall only), each "
           "driven by its own action stream (per-env advance rate uniform 0.35 to 1.0 of 0.9 mm/step and an AR(1) "
           "twist stream; an env stops advancing once its tip is within 2.5 mm of the left common carotid target, "
           "green tip), and advanced by one batched solver call per step. Throughput panel: aggregate "
           "env-steps/s from lumen.newton.throughput.measure_throughput (device-synchronized, after warmup) at "
           "each batch size on the stated hardware, same arch scene, insertion 0.9 mm/step. Live rate: the capture "
           "loop's own solver time including per-step state readback, rendering excluded.",
           "reached_target_final": int(dones[-1].sum()), "hardware": hw, "throughput_rows": rows, "live_env_steps_per_s": round(live_rate, 2),
           "grid_envs": E, "steps": a.steps, "dt_s_per_step": DT, "substeps": SUB, "seed": a.seed,
           "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_parallel " + " ".join(sys.argv[1:] if argv is None else argv),
           "versions": env_versions(), "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip5-batched-envs-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rows, rec["outputs"])


if __name__ == "__main__":
    main()
