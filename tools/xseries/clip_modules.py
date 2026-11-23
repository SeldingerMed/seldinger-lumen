"""Clip 7: Lumen's reduced-order procedure modules, driven live in the Newton sim.

Left, mechanical thrombectomy: a clot occludes the vessel (ClotField: finite-extent,
deformable, damageable occlusion field), a stent-retriever is engaged (Stentriever
engagement coupling) and the wire is pulled back with light aspiration. The clot is
dragged proximally, the occlusion falls and distal flow (NewtonFlow) recovers.

Right, flow diversion: the same saccular aneurysm (Aneurysm + AneurysmSac 0-D compliant
sac on a 1-D pulsatile FlowField) with and without a flow diverter laid across the neck
(FlowDiverter). The diverter throttles the neck inflow jet and lengthens sac turnover.

Every drawn quantity is read from the simulator state each step. The vessel strips are
schematic longitudinal sections (radial axis exaggerated, stated on screen).

    python -m tools.xseries.clip_modules
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (corner_tag, encode, env_versions, font, git_dirty, git_sha,
                                     label, out_root, probe, sha256, write_manifest)

DT = 0.025
M, LEN, R, NW = 60, 120.0, 2.0, 11
BG, PANEL = (7, 9, 13), (11, 14, 19)
WALL, BLOOD, CLOT, WIRE = (212, 96, 92), (46, 18, 22), (226, 70, 84), (92, 232, 238)
AMBER, GREEN, GREY = (247, 181, 64), (112, 211, 141), (190, 198, 208)


def _vessel():
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, LEN, M)], axis=1)


def _wire(z0=40.0):
    return np.stack([np.zeros(NW), np.zeros(NW), np.linspace(z0, z0 + 2 * (NW - 1), NW)], axis=1)


def run_thrombectomy(pull=1.0, aspiration=0.02, settle=4, max_steps=220):
    from lumen.newton.devices import Stentriever
    from lumen.newton.flow import NewtonFlow
    from lumen.newton.sim import NewtonGuidewireSim
    flow = NewtonFlow()
    st = Stentriever(deployed_center=62.0, radial_force=0.2, n_struts=6)
    sim = NewtonGuidewireSim(_vessel(), R, _wire(), radius=0.2, kappa=3e3, d_hat=0.3, flow=flow,
                             clot_segment=(55.0, 70.0), clot_height=1.6, stentriever=st, device="cpu")
    rec = []

    def snap(k, phase):
        c = sim.clot
        Q = float(flow.Q())
        rec.append({"k": k, "phase": phase, "o": np.asarray(c.o, float).copy(),
                    "D": np.asarray(c.D, float).copy(), "s": np.asarray(c.s_grid, float).copy(),
                    "wire": np.asarray(sim.body_positions(), float).copy(),
                    "occlusion": float(flow.occlusion), "Q": Q, "Qd": float(flow.downstream_Q()),
                    "distal_frac": float(flow.downstream_Q()) / max(Q, 1e-9),
                    "status": (sim.last_retrieval or {}).get("status") if hasattr(sim, "last_retrieval") else None})
    for k in range(settle):
        sim.step(dt=DT, substeps=2)
        snap(k, "occluded")
    for k in range(max_steps):
        sim.step(dt=DT, substeps=2, insertion=-pull, aspiration=aspiration)
        snap(settle + k, "retrieval")
        if flow.occlusion < 0.02:
            break
    for k in range(20):
        sim.step(dt=DT, substeps=2)
        snap(len(rec), "reperfused")
    return rec, {"pull_mm_per_step": pull, "aspiration": aspiration, "clot_segment_mm": [55.0, 70.0],
                 "clot_height_mm": 1.6, "vessel_radius_mm": R, "stentriever": vars(st)}


def run_aneurysm(diverter: bool, steps=120, s_neck=50.0):
    from lumen.newton.aneurysm import Aneurysm
    from lumen.newton.devices import FlowDiverter
    from lumen.newton.flow import FlowField, FlowFieldParams
    from lumen.newton.sim import NewtonGuidewireSim
    fd = FlowDiverter(deployed_center=s_neck, span=20.0, metal_coverage=0.35) if diverter else None
    an = Aneurysm(s_neck=s_neck)
    sim = NewtonGuidewireSim(_vessel(), R, _wire(10.0), radius=0.2, kappa=3e3, d_hat=0.3,
                             flow=FlowField(FlowFieldParams(heart_rate=1.2)), aneurysm=an,
                             flow_diverter=fd, device="cpu")
    sac = sim.aneurysm_sacs[0]
    rec = []
    for k in range(steps):
        sim.step(dt=DT, substeps=3)
        rec.append({"k": k, "neck_Q": float(sac.last_Q), "P_sac": float(sac.P_sac),
                    "peak": float(sim.sac_inflow_peak())})
    meta = {"diverter": vars(fd) if fd else None, "aneurysm": vars(an),
            "sac_inflow_peak": float(sim.sac_inflow_peak()), "sac_turnover_time": float(sim.sac_turnover_time()),
            "diversion": float(sim.sac_diversion())}
    return rec, meta


# ------------------------------------------------------------------------- drawing
class Strip:
    """Longitudinal section of a straight vessel: s horizontal, radius vertical (exaggerated)."""

    def __init__(self, x0, x1, yc, rpx):
        self.x0, self.x1, self.yc, self.rpx = x0, x1, yc, rpx

    def X(self, s):
        return self.x0 + (self.x1 - self.x0) * np.asarray(s, float) / LEN

    def Y(self, r):
        return self.yc - self.rpx * np.asarray(r, float) / R

    def vessel(self, d):
        d.rectangle((self.x0, self.Y(R), self.x1, self.Y(-R)), fill=BLOOD)
        d.line((self.x0, self.Y(R), self.x1, self.Y(R)), fill=WALL, width=4)
        d.line((self.x0, self.Y(-R), self.x1, self.Y(-R)), fill=WALL, width=4)


def draw_thrombectomy(img, box, r, t_hist, scale_note):
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    st = Strip(x0 + 30, x1 - 30, y0 + 190, 70)
    st.vessel(d)
    s, o, D = r["s"], r["o"], r["D"]
    for i in range(len(s) - 1):
        if o[i] <= 1e-3 and o[i + 1] <= 1e-3:
            continue
        xa, xb = st.X(s[i]), st.X(s[i + 1])
        oa = 0.5 * (o[i] + o[i + 1])
        dmg = 0.5 * (D[i] + D[i + 1])
        col = tuple(int(c * (1 - 0.5 * dmg) + 60 * 0.5 * dmg) for c in CLOT)
        d.rectangle((xa, st.Y(R), xb, st.Y(R - oa)), fill=col)        # clot from both walls inward
        d.rectangle((xa, st.Y(-(R - oa)), xb, st.Y(-R)), fill=col)
    w = r["wire"]
    pts = [(float(st.X(p[2])), float(st.Y(p[0]))) for p in w]
    d.line([(st.x0, st.Y(0.0))] + pts, fill=WIRE, width=4)
    d.ellipse((pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5), fill=WIRE)
    d.polygon([(st.x1 + 6, st.yc - 10), (st.x1 + 22, st.yc), (st.x1 + 6, st.yc + 10)], fill=GREY)
    d.text((st.x1 + 14, st.yc + 22), "flow", font=font(15), fill=GREY, anchor="ma")
    d.text((st.x0, st.Y(-R) + 14), "proximal", font=font(15), fill=(130, 140, 152), anchor="la")
    d.text((st.x1, st.Y(-R) + 14), "distal", font=font(15), fill=(130, 140, 152), anchor="ra")
    ratio = (st.rpx / R) / ((st.x1 - st.x0) / LEN)
    d.text((st.x0, st.Y(R) - 12), f"{scale_note}, radial axis x{ratio:.0f}", font=font(14),
           fill=(110, 120, 132), anchor="ld")
    # readouts
    occ, frac = r["occlusion"], r["distal_frac"]
    label(img, f"occlusion {100 * occ:4.0f}%", (x0 + 30, y0 + 300), 24, CLOT if occ > 0.3 else GREY)
    label(img, f"distal flow {100 * frac:4.0f}% of inflow", (x0 + 330, y0 + 300), 24,
          GREEN if frac > 0.8 else AMBER)
    phase = {"occluded": "clot occludes the vessel", "retrieval": "stent-retriever engaged, pulling back",
             "reperfused": "clot withdrawn, flow restored"}[r["phase"]]
    label(img, phase, (x0 + 30, y0 + 342), 20, (160, 168, 180))
    # chart
    _chart(img, (x0 + 10, y0 + 390, x1 - 10, y1 - 10), t_hist,
           [("occlusion", [h["occlusion"] for h in t_hist["all"]], CLOT),
            ("distal flow / inflow", [h["distal_frac"] for h in t_hist["all"]], GREEN)],
           ymax=1.05, yfmt=lambda v: f"{100 * v:.0f}%", ticks=[0, 0.25, 0.5, 0.75, 1.0])


def draw_aneurysm(img, box, recs, metas, k, t_hist):
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    qmax = max(max(abs(h["neck_Q"]) for h in rec) for rec in recs)
    for j, (rec, meta, name) in enumerate(zip(recs, metas, ("No device", "Flow diverter across the neck"))):
        yc = y0 + 120 + j * 150
        st = Strip(x0 + 30, x1 - 30, yc, 28)
        st.vessel(d)
        sn = meta["aneurysm"]["s_neck"]
        nw = meta["aneurysm"]["neck_width"]
        xa, xb = st.X(sn - nw / 2), st.X(sn + nw / 2)
        top = st.Y(R)
        d.ellipse((xa - 26, top - 92, xb + 26, top + 6), fill=BLOOD, outline=WALL, width=4)
        d.rectangle((xa + 2, top - 4, xb - 2, top + 4), fill=BLOOD)
        if meta["diverter"]:
            c, sp = meta["diverter"]["deployed_center"], meta["diverter"]["span"]
            for xx in np.arange(st.X(c - sp / 2), st.X(c + sp / 2), 7):
                d.line((xx, st.Y(R) + 2, xx + 7, st.Y(-R) - 2), fill=(170, 180, 192), width=1)
                d.line((xx + 7, st.Y(R) + 2, xx, st.Y(-R) - 2), fill=(170, 180, 192), width=1)
        q = rec[k]["neck_Q"]
        mag = abs(q) / max(qmax, 1e-9)
        xm = 0.5 * (xa + xb)
        if mag > 0.02:
            wpx = 2 + 12 * mag
            if q > 0:
                ya, yb = top + 4, top - 20 - 50 * mag
            else:
                ya, yb = top - 20 - 50 * mag, top + 4
            d.line((xm, ya, xm, yb), fill=AMBER, width=int(wpx))
            hd = 1 if yb < ya else -1
            d.polygon([(xm - wpx, yb + hd * 4), (xm + wpx, yb + hd * 4), (xm, yb - hd * 12)], fill=AMBER)
        label(img, name, (x0 + 30, yc - 118), 22, GREY)
        if j == 1:
            ratio = (st.rpx / R) / ((st.x1 - st.x0) / LEN)
            d.text((x1 - 30, st.Y(-R) + 12), f"section, radial axis x{ratio:.0f}", font=font(14),
                   fill=(110, 120, 132), anchor="ra")
        label(img, f"neck inflow peak {rec[k]['peak']:.2f}", (x1 - 30, yc - 118), 18, (160, 168, 180),
              anchor="ra")
    _chart(img, (x0 + 10, y0 + 390, x1 - 10, y1 - 10), t_hist,
           [("neck inflow, no device", [h["neck_Q"] for h in recs[0]], AMBER),
            ("neck inflow, flow diverter", [h["neck_Q"] for h in recs[1]], GREEN)],
           ymax=1.15 * qmax, ymin=-1.15 * qmax, yfmt=lambda v: f"{v:.0f}",
           ticks=[v for v in range(-20, 21, 5) if abs(v) <= 1.15 * qmax])
    d.text((x0 + 70, y0 + 372), "neck flow into the sac (+) and out (-), simulator units", font=font(14),
           fill=(110, 120, 132), anchor="la")


def _chart(img, box, t_hist, series, ymax, yfmt, ymin=0.0, ticks=None):
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    d.rectangle(box, fill=PANEL)
    L, Rm, T, B = 62, 14, 34, 34
    pl, pr, pt, pb = x0 + L, x1 - Rm, y0 + T, y1 - B
    d.rectangle((pl, pt, pr, pb), outline=(52, 62, 76))
    for yv in (ticks if ticks is not None else np.linspace(ymin, ymax, 5)):
        yy = pb - (pb - pt) * (yv - ymin) / (ymax - ymin)
        d.text((pl - 8, yy), yfmt(yv), font=font(14), fill=(130, 140, 152), anchor="rm")
    if ymin < 0:
    yspan = max(float(ymax) - float(ymin), 1e-9)
    if ymin < 0 < ymax:
        yz = pb - (pb - pt) * (0 - ymin) / yspan
        d.line((pl, yz, pr, yz), fill=(52, 62, 76))
    xl = pl
    for name, _, col in series:
        d.line((xl, y0 + 14, xl + 24, y0 + 14), fill=col, width=4)
        d.text((xl + 30, y0 + 14), name, font=font(15), fill=col, anchor="lm")
        xl += 50 + int(d.textlength(name, font=font(15)))
    n, kn = t_hist["n"], t_hist["k"]
    for _, vals, col in series:
        pts = [(pl + (pr - pl) * j / max(n - 1, 1), pb - (pb - pt) * (min(max(v, ymin), ymax) - ymin) / yspan)
               for j, v in enumerate(vals[:kn + 1])]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
    d.text((pl, pb + 18), f"simulated time  t = {kn * DT:4.2f} s", font=font(14), fill=(130, 140, 152),
           anchor="lm")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps-steps", type=int, default=2, help="video frames per simulated step")
    ap.add_argument("--hold", type=int, default=50)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    if a.fps_steps <= 0:
        raise ValueError("--fps-steps must be positive")
    if a.hold < 0:
        raise ValueError("--hold must be non-negative")
    only = [int(x) for x in a.only.split(",") if x]
    out = Path(a.out) if a.out else out_root() / "clip7-procedure-modules"
    out.mkdir(parents=True, exist_ok=True)
    thr, tmeta = run_thrombectomy()
    an0, m0 = run_aneurysm(False)
    an1, m1 = run_aneurysm(True)
    summ = {"thrombectomy": {"steps": len(thr), "occlusion_start": thr[0]["occlusion"],
                             "occlusion_end": thr[-1]["occlusion"],
                             "distal_flow_frac_start": round(thr[0]["distal_frac"], 3),
                             "distal_flow_frac_end": round(thr[-1]["distal_frac"], 3),
                             "retrieval_statuses": sorted({r["status"] for r in thr if r["status"]})},
            "aneurysm_no_device": {k: m0[k] for k in ("sac_inflow_peak", "sac_turnover_time", "diversion")},
            "aneurysm_flow_diverter": {k: m1[k] for k in ("sac_inflow_peak", "sac_turnover_time", "diversion")}}
    summ["aneurysm_change"] = {
        "inflow_peak_pct": (round(100 * (m1["sac_inflow_peak"] / m0["sac_inflow_peak"] - 1), 1)
                            if m0["sac_inflow_peak"] > 0 else None),
        "turnover_time_pct": (round(100 * (m1["sac_turnover_time"] / m0["sac_turnover_time"] - 1), 1)
                              if m0["sac_turnover_time"] > 0 else None)}
    print(summ, flush=True)
    n_steps = max(len(thr), len(an0))
    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    sched = [k for k in range(n_steps) for _ in range(a.fps_steps)] + [n_steps - 1] * a.hold
    for i, k in enumerate(sched):
        if only and i not in only:
            continue
        kt, ka = min(k, len(thr) - 1), min(k, len(an0) - 1)
        L = Image.new("RGB", (1920, 1080), BG)
        label(L, "Thrombectomy: stent-retriever pull-back", (40, 22), 30, GREY)
        label(L, "Aneurysm: flow diverter vs no device", (1000, 22), 30, GREY)
        draw_thrombectomy(L, (20, 80, 940, 1060), thr[kt], {"all": thr, "n": len(thr), "k": kt},
                          "longitudinal section")
        draw_aneurysm(L, (980, 80, 1900, 1060), (an0, an1), (m0, m1), ka, {"n": len(an0), "k": ka})
        d = ImageDraw.Draw(L)
        d.line((960, 80, 960, 1050), fill=(30, 36, 46), width=2)
        if k >= len(an0) - 1:
            ch = summ["aneurysm_change"]
            ip = f"{ch['inflow_peak_pct']:+.0f}%" if ch["inflow_peak_pct"] is not None else "n/a"
            tt = f"{ch['turnover_time_pct']:+.0f}%" if ch["turnover_time_pct"] is not None else "n/a"
            label(L, f"diverter: peak neck inflow {ip},  sac turnover time {tt}",
                  (1440, 400), 20, GREEN, anchor="ma", bg=(10, 13, 18))
        corner_tag(L, top=False)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        S = Image.new("RGB", (1080, 1080), BG)
        top = Image.new("RGB", (960, 1000), BG)
        draw_thrombectomy(top, (0, 0, 960, 1000), thr[kt], {"all": thr, "n": len(thr), "k": kt},
                          "section")
        bot = Image.new("RGB", (960, 1000), BG)
        draw_aneurysm(bot, (0, 0, 960, 1000), (an0, an1), (m0, m1), ka, {"n": len(an0), "k": ka})
        S.paste(top.crop((0, 60, 960, 380)).resize((1020, 340)), (30, 60))
        S.paste(bot.crop((0, 0, 960, 370)).resize((1020, 393)), (30, 500))
        label(S, "Thrombectomy: stent-retriever pull-back", (24, 14), 28, GREY)
        label(S, "Aneurysm: flow diverter vs no device", (24, 440), 28, GREY)
        if k >= len(an0) - 1:
            ch = summ["aneurysm_change"]
            label(S, f"diverter: peak neck inflow {ch['inflow_peak_pct']:+.0f}%,  turnover {ch['turnover_time_pct']:+.0f}%",
                  (540, 960), 24, GREEN, anchor="ma", bg=(10, 13, 18))
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if i % 60 == 0:
            print(f"frame {i}/{len(sched)}", flush=True)
    if only:
        print("preview written")
        return
    rec = {"clip": "7-procedure-modules",
           "description": "Lumen's reduced-order procedure modules running inside the Newton guidewire sim. Left: "
           "ClotField occlusion with a Stentriever engagement coupling; the wire is pulled back with light "
           "aspiration, the clot is dragged proximally and NewtonFlow distal flow recovers. Right: the same "
           "saccular Aneurysm (0-D compliant AneurysmSac on a 1-D pulsatile FlowField) with and without a "
           "FlowDiverter across the neck. Vessel strips are schematic longitudinal sections with the radial "
           "axis exaggerated; every drawn quantity is read from the simulator each step. Flow and inflow "
           "values are in simulator units; these are fast-tier reduced-order models, not CFD or FE devices.",
           "summary": summ, "thrombectomy_setup": tmeta, "aneurysm_no_device": m0, "aneurysm_flow_diverter": m1,
           "dt_s_per_step": DT, "video_frames_per_step": a.fps_steps,
           "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_modules " + " ".join(sys.argv[1:] if argv is None else argv),
           "versions": env_versions(), "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip7-procedure-modules-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
