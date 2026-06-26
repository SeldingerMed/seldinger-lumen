"""L1.2 — device-as-sensor: recover wall mechanics from the device image (doc §3.6).

The conceptual payoff of the imaging loop: a device pressed against the wall deflects
by an amount that depends on the wall's stiffness, and that deflection is visible in
fluoroscopy — so running the wall mechanics in inverse mode through the image yields an
in-vivo wall-stiffness estimate (instrumented elastography).

Two mechanical observables, both imaged, complete the M2 "wall/friction" pair:
  * WALL stiffness (C10) -> a wall-ward BULGE (`device_on_wall`): a softer wall yields
    more, so the device pressed into it bulges deeper.
  * FRICTION (mu) -> an axial stick-slip LAG (`device_with_friction`): under a proximal
    push, distributed Coulomb/Dahl wall friction holds the distal device, so the tip
    advances less than the base (§3.5.5). High friction -> visible tip lag.
The two displace the device in (near-)orthogonal directions — lateral bulge vs axial
lag — so a joint inverse (`estimate_wall_and_friction`) can recover BOTH, and
`joint_identifiability` reports whether they are SEPARABLY identifiable from the given
views (the M2 "bounded identifiability" gate, as a Fisher conditioning number).

Scope (honest, per the doc's own caveats §3.6):
  * The forward here is a QUASI-STATIC reduced coupling, not the full rod dynamics:
    the device rides the wall, and the wall's radial yield w(load; C10) is the REAL
    HGO constitutive solve (lumen.newton.hgo_wall). So the thing being inverted —
    wall stiffness / friction through the imaged device displacement — is faithful; the
    device–wall mechanical coupling is simplified (a firm-contact rod buckles in the
    full sim; that stable-coupled version is future work).
  * Identifiability is the gate (doc §3.6, M2). Each parameter ALONE is mono-identifiable
    from a suitable view (the axial lag is visible from any side view; the bulge from any
    view not aligned with bulge_dir). It is the JOINT (C10, mu) that a single bulge-aligned
    view leaves under-determined — there the bulge foreshortens and the two confound.
    `joint_identifiability()` quantifies this (Fisher cond/lam_min) and shows biplanar
    resolves it.

Which identifiability tool to use: `*_sensitivity` is a scalar image-change proxy (cheap,
single param); `*_identifiability` returns a loss-vs-param CURVE (single param, shows the
minimum's sharpness); `joint_identifiability` returns the 2×2 Fisher Gram for the JOINT
(C10, mu) — the only one that reports parameter CONFOUNDING. Reach for the last when the
question is "can I separate stiffness from friction from these views".
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


# --- FRICTION arm (axial stick-slip lag) -------------------------------------
def _friction_advance(mu, push, normal_load, k_axial, z, span, contact_frac):
    """Per-node axial advance under a proximal push against distributed wall friction.

    QUASI-STATIC Coulomb: each node advances by `push` minus the cumulative friction drag
    of the contact DISTAL to it (`mu·normal_load/k_axial`), clamped at 0 (fully stuck). So
    the base advances ~push and the tip lags — the imaged signature of friction. NOTE this
    is a steady stick-slip *lag profile*, not the dynamic stick-slip oscillation (no
    velocity dependence / Dahl hysteresis) — consistent with the §3.6 quasi-static scope."""
    if span <= 0.0 or contact_frac <= 0.0 or k_axial <= 0.0:
        raise ValueError(f"span, contact_frac, k_axial must be > 0; got "
                         f"span={span}, contact_frac={contact_frac}, k_axial={k_axial}")
    contact = np.exp(-(z / (contact_frac * span / 2)) ** 2)        # contact load profile
    frac_distal = np.cumsum(contact[::-1])[::-1]
    frac_distal = frac_distal / frac_distal[0]                     # 1 at base -> ~0 at tip
    return np.maximum(0.0, push - (mu * normal_load / k_axial) * (1.0 - frac_distal))


def device_with_friction(mu, push=6.0, normal_load=300.0, k_axial=120.0, n=21, span=24.0,
                         axis=(0.0, 0.0, 1.0), contact_frac=0.5):
    """Device pushed proximally by `push` against wall friction `mu`: the tip lags by the
    accumulated quasi-static Coulomb drag, so the imaged device compresses toward the base.
    More friction -> larger tip lag. Returns node positions (n,3).

    Note: at mu->0 the device is a near-rigid translation by `push`, so the image barely
    changes with mu near 0 -> low-mu friction is weakly identifiable; mid-range mu is where
    the lag signal is strong (see friction_sensitivity)."""
    if mu < 0.0:                                                   # negative friction is unphysical
        raise ValueError(f"mu must be >= 0; got {mu}")
    d = np.asarray(axis, float)
    nrm = np.linalg.norm(d)
    if nrm < 1e-9:
        raise ValueError("axis must be non-zero")
    d = d / nrm
    z = np.linspace(-span / 2, span / 2, n)
    adv = _friction_advance(mu, push, normal_load, k_axial, z, span, contact_frac)
    base = np.stack([np.zeros(n), np.zeros(n), z], axis=1)
    return base + np.outer(adv, d)


def estimate_friction(targets, sensor, carms, init_mu=0.3, push=6.0, normal_load=300.0,
                      k_axial=120.0, iters=30, **dev_kw):
    """Recover the wall friction `mu` from target fluoro image(s) by matching the imaged
    stick-slip lag. Returns (mu, history). Same view contract as estimate_wall_stiffness."""
    if init_mu < 0.0:                                  # a negative seed sits in the clamped flat region
        raise ValueError(f"init_mu must be >= 0; got {init_mu}")
    carms = [carms] if hasattr(carms, "rays") else list(carms)
    targets = [targets] if np.ndim(targets) == 2 else list(targets)
    if len(carms) != len(targets):
        raise ValueError(f"{len(carms)} carms vs {len(targets)} targets")

    def loss(x):
        nodes = device_with_friction(max(0.0, float(x[0])), push=push, normal_load=normal_load,
                                     k_axial=k_axial, **dev_kw)
        imgs = [sensor.render(nodes, carm=c)[0] for c in carms]
        return float(sum(np.mean((a - t) ** 2) for a, t in zip(imgs, targets)))

    x, hist = fd_minimize(loss, [float(init_mu)], scale=[0.3], iters=iters)
    return max(0.0, float(x[0])), hist


def friction_sensitivity(true_mu, sensor, carms, rel=0.1, push=6.0, normal_load=300.0,
                         k_axial=120.0, **dev_kw):
    """Identifiability proxy for friction: the image change for a `rel`-scaled mu step,
    summed over views. Shrinks as the lag saturates (fully stuck) and grows with views.
    The step is `rel·max(mu, 0.1)` so the mu=0 boundary still gets a nonzero perturbation
    (a pure `rel·mu` step would report a spurious zero there)."""
    if hasattr(carms, "rays"):
        carms = [carms]
    common = dict(push=push, normal_load=normal_load, k_axial=k_axial, **dev_kw)
    step = rel * max(true_mu, 0.1)                     # nonzero even at the mu=0 boundary
    a = [sensor.render(device_with_friction(true_mu, **common), carm=c)[0] for c in carms]
    b = [sensor.render(device_with_friction(true_mu + step, **common), carm=c)[0]
         for c in carms]
    return float(sum(np.mean((x - y) ** 2) for x, y in zip(a, b)))


def friction_identifiability(true_mu, sensor, carms_by_view, mu_grid, push=6.0,
                             normal_load=300.0, k_axial=120.0, **dev_kw):
    """Loss(mu) curves per view set (targets at `true_mu`). Sharper single minimum = more
    identifiable. Returns {view: loss_array}. Mirrors the wall-stiffness identifiability."""
    out = {}
    common = dict(push=push, normal_load=normal_load, k_axial=k_axial, **dev_kw)
    for view, carms in carms_by_view.items():
        carms = [carms] if hasattr(carms, "rays") else list(carms)
        targets = [sensor.render(device_with_friction(true_mu, **common), carm=c)[0] for c in carms]
        losses = []
        for mu in mu_grid:
            imgs = [sensor.render(device_with_friction(float(mu), **common), carm=c)[0]
                    for c in carms]
            losses.append(float(sum(np.mean((a - t) ** 2) for a, t in zip(imgs, targets))))
        out[view] = np.array(losses)
    return out


# --- JOINT (C10, mu): the full M2 wall+friction inverse ----------------------
def device_wall_and_friction(C10, mu, load=300.0, R0=2.0, push=6.0, normal_load=300.0,
                             k_axial=120.0, k1_ratio=0.5, n=21, span=24.0,
                             bulge_dir=(1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                             contact_frac=0.5):
    """Device that both PRESSES the wall (lateral bulge ∝ wall yield w(C10)) and is PUSHED
    against friction (axial lag ∝ mu). The two displacements are ~orthogonal, which is
    what lets the joint inverse separate them. Returns node positions (n,3)."""
    if mu < 0.0:
        raise ValueError(f"mu must be >= 0; got {mu}")
    params = HGOParams(C10=C10, k1=C10 * k1_ratio, k2=1.0, thickness=0.3)
    w = wall_yield(load, R0, params)
    bd = np.asarray(bulge_dir, float)
    ax = np.asarray(axis, float)
    if np.linalg.norm(bd) < 1e-9 or np.linalg.norm(ax) < 1e-9:
        raise ValueError("bulge_dir and axis must be non-zero")
    bd, ax = bd / np.linalg.norm(bd), ax / np.linalg.norm(ax)
    if abs(float(np.dot(bd, ax))) > 0.95:              # else bulge and lag are collinear ->
        raise ValueError("bulge_dir must not be ~parallel to the friction axis; the wall "
                         "bulge and friction lag would be collinear and confounded in the "
                         "forward model itself (view-independent degeneracy)")
    z = np.linspace(-span / 2, span / 2, n)
    bump = np.exp(-(z / (contact_frac * span / 2)) ** 2)
    adv = _friction_advance(mu, push, normal_load, k_axial, z, span, contact_frac)
    base = np.stack([np.zeros(n), np.zeros(n), z], axis=1)
    return base + np.outer(bump * w, bd) + np.outer(adv, ax)


def estimate_wall_and_friction(targets, sensor, carms, init_C10=4.0e3, init_mu=0.3,
                               load=300.0, R0=2.0, push=6.0, normal_load=300.0,
                               k_axial=120.0, bulge_dir=(1.0, 0.0, 0.0), iters=40, **dev_kw):
    """Jointly recover (C10, mu) from target fluoro image(s) by matching the bulge AND the
    lag. Optimises log-C10 (positive, scaled) and mu. Returns (C10, mu, history)."""
    if init_C10 <= 0.0 or init_mu < 0.0:               # log(C10) needs C10>0; mu>=0
        raise ValueError(f"need init_C10 > 0 and init_mu >= 0; got {init_C10}, {init_mu}")
    carms = [carms] if hasattr(carms, "rays") else list(carms)
    targets = [targets] if np.ndim(targets) == 2 else list(targets)
    if len(carms) != len(targets):
        raise ValueError(f"{len(carms)} carms vs {len(targets)} targets")

    def loss(x):
        nodes = device_wall_and_friction(float(np.exp(x[0])), max(0.0, float(x[1])),
                                         load=load, R0=R0, push=push, normal_load=normal_load,
                                         k_axial=k_axial, bulge_dir=bulge_dir, **dev_kw)
        imgs = [sensor.render(nodes, carm=c)[0] for c in carms]
        return float(sum(np.mean((a - t) ** 2) for a, t in zip(imgs, targets)))

    x, hist = fd_minimize(loss, [np.log(init_C10), float(init_mu)], scale=[0.6, 0.3], iters=iters)
    return float(np.exp(x[0])), max(0.0, float(x[1])), hist


def joint_identifiability(true_C10, true_mu, sensor, carms_by_view, rel=0.1,
                          load=300.0, R0=2.0, push=6.0, normal_load=300.0, k_axial=120.0,
                          bulge_dir=(1.0, 0.0, 0.0), **dev_kw):
    """Are (C10, mu) SEPARABLY identifiable from each view set? Builds the image Jacobian
    w.r.t. (log C10, mu) by finite differences, forms the 2×2 Gram G=JᵀJ (Fisher up to a
    noise scale), and reports {view: {cond, lam_min, corr, G}}:
      * `lam_min` — smallest eigenvalue of G = how strongly the WORST-determined parameter
        combination is constrained. Monotone: adding a view can only raise it (G is a sum
        of per-view Grams), so biplanar always improves the hardest direction. This is the
        honest 'is it identifiable at all' number.
      * `cond` — condition number λmax/λmin. High -> ill-posed/confounded. NOT monotone in
        views (a view can lift λmax more than λmin), so biplanar's job is to rescue the
        CATASTROPHIC mono case (a bulge-aligned view, cond -> ∞), not to beat the best view.
      * `corr` — parameter cross-coupling in [-1,1]; near ±1 means the two params mimic each
        other in the image (confounded).
    The quantitative M2 'bounded identifiability' statement. The Jacobian uses CENTRAL
    differences (O(rel²) truncation) with a multiplicative log-C10 step and a relative mu
    step, so the columns are scale-consistent and `lam_min` is a trustworthy Fisher number
    (a one-sided difference would bias it ~O(rel))."""
    out = {}
    h_log = np.log1p(rel)                                  # log-C10 step: C10·(1±rel) ⇒ Δlog = log1p(±rel)
    for view, carms in carms_by_view.items():
        carms = [carms] if hasattr(carms, "rays") else list(carms)

        def img(C10, mu, carms=carms):
            nodes = device_wall_and_friction(C10, mu, load=load, R0=R0, push=push,
                                             normal_load=normal_load, k_axial=k_axial,
                                             bulge_dir=bulge_dir, **dev_kw)
            return np.concatenate([sensor.render(nodes, carm=c)[0].ravel() for c in carms])

        h_mu = rel * max(true_mu, 0.1)                     # relative mu step (nonzero at mu=0)
        mu_lo = max(true_mu - h_mu, 0.0)                   # keep mu >= 0
        d_logC10 = (img(true_C10 * (1 + rel), true_mu)              # central ∂img/∂(log C10)
                    - img(true_C10 / (1 + rel), true_mu)) / (2 * h_log)
        d_mu = (img(true_C10, true_mu + h_mu)                       # central ∂img/∂mu
                - img(true_C10, mu_lo)) / (true_mu + h_mu - mu_lo)
        J = np.stack([d_logC10, d_mu], axis=1)                          # (pixels, 2)
        G = J.T @ J
        evals = np.linalg.eigvalsh(G)
        cond = float(evals[-1] / max(evals[0], 1e-30))
        corr = float(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1] + 1e-30))
        out[view] = {"cond": cond, "lam_min": float(evals[0]), "corr": corr, "G": G}
    return out
