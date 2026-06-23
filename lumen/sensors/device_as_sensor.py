"""L1.2 — device-as-sensor: recover wall mechanics from the device image (doc §3.6).

The conceptual payoff of the imaging loop: a device pressed against the wall deflects
by an amount that depends on the wall's stiffness, and that deflection is visible in
fluoroscopy — so running the wall mechanics in inverse mode through the image yields an
in-vivo wall-stiffness estimate (instrumented elastography).

Scope (honest, per the doc's own caveats §3.6):
  * The forward here is a QUASI-STATIC reduced coupling, not the full rod dynamics:
    the device rides the wall, and the wall's radial yield w(load; C10) is the REAL
    HGO constitutive solve (lumen.newton.hgo_wall). So the thing being inverted —
    wall stiffness through the imaged device displacement — is faithful; the
    device–wall mechanical coupling is simplified (a firm-contact rod buckles in the
    full sim; that stable-coupled version is future work).
  * Identifiability is the gate (doc §3.6, M2): wall stiffness from a SINGLE 2-D
    projection is under-determined — the device's wall-ward displacement is only
    partly in-plane. `identifiability()` quantifies this and shows biplanar resolves it.
"""

from __future__ import annotations

import numpy as np

from lumen.newton.hgo_wall import HGOParams, hgo_wall_pressure
from lumen.sensors._optim import fd_minimize


def wall_yield(load, R0, params: HGOParams):
    """Radial wall yield w solving HGO shell pressure(w) = load (the real constitutive
    inverse). Softer wall (smaller C10) -> larger w. Bisection on [0, 0.9·R0]."""
    lo, hi = 0.0, 0.9 * R0
    if hgo_wall_pressure(hi, R0, params) < load:      # L1: load exceeds wall capacity -> fail loud
        raise ValueError(f"load {load} exceeds HGO wall capacity at 0.9·R0 "
                         f"(no rupture model); lower the load or stiffen the wall")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if hgo_wall_pressure(mid, R0, params) < load:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def device_on_wall(C10, load=300.0, R0=2.0, k1_ratio=0.5, n=21, span=24.0,
                   bulge_dir=(1.0, 0.0, 0.0), contact_frac=0.5):
    """Device pressed against the wall: a wire along +z that bulges toward `bulge_dir`
    by the wall yield w(load; C10) over a central contact patch. Softer wall -> deeper
    bulge -> a visibly different fluoro. Returns node positions (n,3)."""
    params = HGOParams(C10=C10, k1=C10 * k1_ratio, k2=1.0, thickness=0.3)
    w = wall_yield(load, R0, params)
    z = np.linspace(-span / 2, span / 2, n)
    bump = np.exp(-(z / (contact_frac * span / 2)) ** 2)          # smooth contact patch
    d = np.asarray(bulge_dir, float)
    nrm = np.linalg.norm(d)
    if nrm < 1e-9:                                                # L2: a zero dir would silently
        raise ValueError("bulge_dir must be non-zero")           # render a straight wire
    d = d / nrm
    base = np.stack([np.zeros(n), np.zeros(n), z], axis=1)         # wire centred on z
    return base + np.outer(bump * w, d)                            # contact apex displaced by w


def estimate_wall_stiffness(targets, sensor, carms, init_C10=4.0e3, load=300.0,
                            R0=2.0, bulge_dir=(1.0, 0.0, 0.0), iters=30, **dev_kw):
    """Recover C10 from target fluoro image(s) by matching the imaged device bulge.
    Optimises in log-C10 (positive, well-scaled). Returns (C10, history)."""
    carms = [carms] if hasattr(carms, "rays") else list(carms)       # bare CArm -> [CArm]
    targets = [targets] if np.ndim(targets) == 2 else list(targets)  # bare image -> [image]
    if len(carms) != len(targets):                                   # H1: no silent zip-truncation
        raise ValueError(f"{len(carms)} carms vs {len(targets)} targets")

    def loss(x):
        nodes = device_on_wall(float(np.exp(x[0])), load=load, R0=R0,
                               bulge_dir=bulge_dir, **dev_kw)
        imgs = [sensor.render(nodes, carm=c)[0] for c in carms]
        return float(sum(np.mean((a - t) ** 2) for a, t in zip(imgs, targets)))

    x, hist = fd_minimize(loss, [np.log(init_C10)], scale=[0.6], iters=iters)
    return float(np.exp(x[0])), hist


def sensitivity(true_C10, sensor, carms, rel=0.1, load=300.0, R0=2.0,
                bulge_dir=(1.0, 0.0, 0.0), **dev_kw):
    """Identifiability proxy (∝ Fisher information): the image change for a +`rel`
    relative C10 step, summed over views. Large = well-determined. It shrinks with
    wall stiffness (stiff wall -> sub-pixel device displacement) and grows with views
    — the quantitative form of the doc's 'under-determined from 2-D; biplanar resolves
    it' caveat (§3.6)."""
    if hasattr(carms, "rays"):
        carms = [carms]
    a = [sensor.render(device_on_wall(true_C10, load=load, R0=R0, bulge_dir=bulge_dir, **dev_kw),
                       carm=c)[0] for c in carms]
    b = [sensor.render(device_on_wall(true_C10 * (1 + rel), load=load, R0=R0,
                                      bulge_dir=bulge_dir, **dev_kw), carm=c)[0] for c in carms]
    return float(sum(np.mean((x - y) ** 2) for x, y in zip(a, b)))


def identifiability(true_C10, sensor, carms_by_view, C10_grid, load=300.0, R0=2.0,
                    bulge_dir=(1.0, 0.0, 0.0), **dev_kw):
    """Loss(C10) curves for each view set (e.g. {'mono':[c1], 'biplanar':[c1,c2]}),
    with targets generated at `true_C10`. A sharper, single-minimum curve = more
    identifiable. Returns {view: loss_array}."""
    out = {}
    for view, carms in carms_by_view.items():
        carms = [carms] if hasattr(carms, "rays") else list(carms)   # L8: accept a bare CArm
        nodes_t = device_on_wall(true_C10, load=load, R0=R0, bulge_dir=bulge_dir, **dev_kw)
        targets = [sensor.render(nodes_t, carm=c)[0] for c in carms]
        losses = []
        for C10 in C10_grid:
            nodes = device_on_wall(float(C10), load=load, R0=R0, bulge_dir=bulge_dir, **dev_kw)
            imgs = [sensor.render(nodes, carm=c)[0] for c in carms]
            losses.append(float(sum(np.mean((a - t) ** 2) for a, t in zip(imgs, targets))))
        out[view] = np.array(losses)
    return out
