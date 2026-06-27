"""L2.3 — sim2sim calibration: recover wall stiffness from a captured episode (doc §3.6).

Closes the imaging loop on Layer-2 data: an episode carries the fluoro frame(s) of a
device pressed against the wall at an (unknown, to the solver) stiffness; running the
device-as-sensor inverse (L1.2) through those frames recovers C10. Because it is
sim2sim, the ground-truth C10 is stored in `meta.notes["calib"]`, so the harness
reports the exact recovery error — the honest demonstration of what real DSA data
will do through the SAME `Episode` seam (§3.6, the M2 identifiability gate).

Honest scope: this inverts the device-on-wall QUASI-STATIC forward model (L1.2), so
calibration runs on a **wall-probe** episode (`probe_episode`), not on an L2.1
navigation rollout — a navigation frame is a different scene and is not invertible by
this model. Navigation episodes are for perception/policy (Layer 3); wall-probe
episodes are for calibration. Both are the same `lumen-episode/0` schema.
"""

from __future__ import annotations

import warnings

import numpy as np

from lumen.data.schema import Episode, EpisodeMeta, Outcome, Step


def _fluoro_meta(sensor) -> dict:
    return {"modality": "fluoro", "mu_device": sensor.mu_device, "eps": sensor.eps,
            "res": sensor.res, "n_samples": sensor.n_samples, "margin": sensor.margin,
            "nu": sensor.nu, "nv": sensor.nv}


def _fluoro_from_meta(m: dict):
    from lumen.sensors import FluoroSensor
    return FluoroSensor(mu_device=m.get("mu_device", 1.0), eps=m.get("eps", 0.6),
                        res=m["res"], n_samples=m.get("n_samples", 192),
                        margin=m.get("margin", 8.0), nu=m["nu"], nv=m["nv"])


def probe_episode(true_C10, sensor, carms=None, view_axis=(0.0, 0.0, 1.0), load=300.0,
                  R0=2.0, bulge_dir=(1.0, 0.0, 0.0), asset_ref="", label="wall_probe",
                  notes=None, **dev_kw):
    """Generate a wall-probe calibration episode: a device pressed against the wall at
    `true_C10`, imaged from one or more C-arms (one step per view). The episode is
    self-contained for calibration — it stores the views, the fluoro sensor config, and
    the forward-model params (load/R0/bulge_dir/dev_kw) plus the ground-truth C10."""
    from lumen.sensors.device_as_sensor import device_on_wall

    nodes = device_on_wall(true_C10, load=load, R0=R0, bulge_dir=bulge_dir, **dev_kw)
    if carms is None:
        # L3: a view along the bulge axis is depth-ambiguous (the §3.6 under-determined case)
        if abs(float(np.dot(np.asarray(view_axis, float), np.asarray(bulge_dir, float)))) > 0.99:
            warnings.warn("default view_axis is ~parallel to bulge_dir; this mono view is "
                          "depth-ambiguous (under-determined) — add a second view", stacklevel=2)
        carms = [sensor.default_carm(nodes, axis=view_axis)]
    elif hasattr(carms, "rays"):
        carms = [carms]
    else:
        carms = list(carms)

    if not carms:
        raise ValueError("at least one C-arm must be provided for calibration")

    # a probe is N VIEWS of one static scene, not a time series — t indexes the view and
    # dt=0. notes["episode_kind"] lets replay/summarize tell probes from navigation rollouts.
    steps = [Step(t=float(i), action={"load": float(load)}, kinematics={"view": i},
                  obs_modality="fluoro", obs_ref=f"{i:03d}.npy",
                  obs=sensor.render(nodes, carm=c)[0])
             for i, c in enumerate(carms)]
    meta = EpisodeMeta(
        asset_ref=asset_ref, dt=0.0, device={"R0": float(R0)},
        sensor=_fluoro_meta(sensor),       # documented renderer shape only — carms live in calib
        calibration={"type": "carm", "views": [c.to_dict() for c in carms]},
        notes={**(notes or {}),
               "episode_kind": "wall_probe",
               "calib": {"true_C10": float(true_C10), "load": float(load), "R0": float(R0),
                         "bulge_dir": list(bulge_dir), "dev_kw": dict(dev_kw),
                         "carms": [c.to_dict() for c in carms]}})
    return Episode(meta=meta, steps=steps,
                   outcome=Outcome(success=True, final_dist=0.0, steps=len(steps), label=label))


def joint_probe_episode(true_C10, true_mu, sensor, carms=None, view_axis=(0.0, 1.0, 0.0),
                        load=300.0, R0=2.0, bulge_dir=(1.0, 0.0, 0.0), push=6.0,
                        normal_load=300.0, k_axial=120.0, axis=(0.0, 0.0, 1.0), asset_ref="",
                        label="wall_friction_probe", notes=None, **dev_kw):
    """Generate a wall+friction probe episode: a device that both presses the wall (bulge
    ∝ 1/C10) and is pushed against friction (axial lag ∝ mu), imaged from one or more
    C-arms. Stores both ground-truth params + the forward knobs so `calibrate_from_episode`
    recovers (C10, mu) jointly — the M2 'wall/friction' seam on Layer-2 data.

    The default `view_axis` (+y) sees BOTH the lateral bulge (x) and the axial lag (z)
    in-plane; a mono view aligned with either bulge_dir or the friction axis foreshortens
    that effect and confounds the joint (it warns)."""
    from lumen.sensors.device_as_sensor import device_wall_and_friction

    nodes = device_wall_and_friction(true_C10, true_mu, load=load, R0=R0, bulge_dir=bulge_dir,
                                     push=push, normal_load=normal_load, k_axial=k_axial,
                                     axis=axis, **dev_kw)
    if carms is None:
        va = np.asarray(view_axis, float)
        if (abs(float(np.dot(va, np.asarray(bulge_dir, float)))) > 0.99
                or abs(float(np.dot(va, np.asarray(axis, float)))) > 0.99):
            warnings.warn("mono view_axis is ~parallel to bulge_dir or the friction axis; it "
                          "foreshortens that effect and the joint (C10, mu) is under-"
                          "determined — add a second view", stacklevel=2)
        carms = [sensor.default_carm(nodes, axis=view_axis)]
    elif hasattr(carms, "rays"):
        carms = [carms]
    else:
        carms = list(carms)
    if not carms:
        raise ValueError("at least one C-arm must be provided for calibration")

    steps = [Step(t=float(i), action={"load": float(load), "push": float(push)},
                  kinematics={"view": i}, obs_modality="fluoro", obs_ref=f"{i:03d}.npy",
                  obs=sensor.render(nodes, carm=c)[0])
             for i, c in enumerate(carms)]
    meta = EpisodeMeta(
        asset_ref=asset_ref, dt=0.0, device={"R0": float(R0)},
        sensor=_fluoro_meta(sensor),
        calibration={"type": "carm", "views": [c.to_dict() for c in carms]},
        notes={**(notes or {}),
               "episode_kind": "wall_friction_probe",
               "calib": {"true_C10": float(true_C10), "true_mu": float(true_mu),
                         "load": float(load), "R0": float(R0), "bulge_dir": list(bulge_dir),
                         "axis": list(axis), "push": float(push), "normal_load": float(normal_load),
                         "k_axial": float(k_axial), "dev_kw": dict(dev_kw),
                         "carms": [c.to_dict() for c in carms]}})
    return Episode(meta=meta, steps=steps,
                   outcome=Outcome(success=True, final_dist=0.0, steps=len(steps), label=label))


def calibrate_from_episode(episode: Episode, root: str | None = None,
                           init_C10: float = 4.0e3, init_mu: float = 0.3, iters: int = 30,
                           noise_std: float = 0.0, noise_seed: int = 0,
                           identifiable_tol: float = 0.05) -> dict:
    """Recover wall stiffness from a wall-probe episode and report the sim2sim error.

    Reconstructs the forward model from the episode (sensor + C-arms + forward params),
    pulls the target frame(s) (in-memory `step.obs` if present, else the sidecar via
    `root`), runs the L1.2 inverse, and returns recovered/true C10 + rel_error + n_views.

    Noise-free recovery is necessary but NOT sufficient: the deterministic render makes
    loss(true_C10)=0 exactly, so even an under-determined mono view "recovers" trivially.
    Pass `noise_std>0` to probe identifiability honestly — it re-recovers from noised
    targets and reports `rel_error_noisy` + `identifiable` (rel_error_noisy <
    identifiable_tol). A mono out-of-plane view blows up under noise; biplanar holds
    (the §3.6 gate). This is the honest closure, not the noise-free number."""
    from lumen.sensors.carm import CArm
    from lumen.sensors.device_as_sensor import estimate_wall_stiffness

    calib = episode.meta.notes.get("calib")
    if calib is None:
        raise ValueError("not a calibration probe episode (no meta.notes['calib']); a "
                         "navigation episode can't be inverted by the device-on-wall model")
    carms_d = episode.meta.calibration.get("views") or calib.get("carms")
    if not carms_d:
        raise ValueError("calibration episode has no stored C-arm views (calib['carms'])")
    root = root or getattr(episode, "root", None)
    try:
        sensor = _fluoro_from_meta(episode.meta.sensor)
        carms = [CArm.from_dict(c) for c in carms_d]
    except (KeyError, TypeError) as e:
        raise ValueError(f"calibration episode is malformed (sensor/carms): {e}") from e

    def _target(s):
        if s.obs is not None:
            return np.asarray(s.obs)
        if root is None:
            raise ValueError("episode root needed to load obs sidecars (or pass an in-memory episode)")
        return s.load_obs(root)

    targets = [_target(s) for s in episode.steps]
    if len(targets) != len(carms):
        raise ValueError(f"{len(targets)} target frames vs {len(carms)} carms")

    if "true_mu" in calib:        # joint wall+friction probe -> recover (C10, mu) together
        return _calibrate_joint(calib, targets, sensor, carms, init_C10, init_mu, iters,
                                noise_std, noise_seed, identifiable_tol)

    fwd = dict(load=calib["load"], R0=calib["R0"], bulge_dir=tuple(calib["bulge_dir"]),
               **calib.get("dev_kw", {}))
    true_C10 = float(calib["true_C10"])

    def _recover(ts):
        rec, hist = estimate_wall_stiffness(ts, sensor, carms, init_C10=init_C10, iters=iters, **fwd)
        return float(rec), abs(rec - true_C10) / true_C10, hist

    recovered, rel_error, hist = _recover(targets)
    out = {"recovered_C10": recovered, "true_C10": true_C10, "rel_error": rel_error,
           "n_views": len(carms), "history": hist}
    if noise_std > 0:                              # honest identifiability probe (H1)
        rng = np.random.default_rng(noise_seed)
        noised = [t + rng.normal(0.0, noise_std, np.shape(t)) for t in targets]
        rec_n, rel_n, _ = _recover(noised)
        out.update(recovered_C10_noisy=rec_n, rel_error_noisy=rel_n, noise_std=noise_std,
                   identifiable=bool(rel_n < identifiable_tol))
    return out


def _calibrate_joint(calib, targets, sensor, carms, init_C10, init_mu, iters,
                     noise_std, noise_seed, identifiable_tol):
    """Joint (C10, mu) recovery branch of calibrate_from_episode (wall+friction probe).

    Reports BOTH identifiability notions, which are distinct: `cond`/`lam_min` are the
    DETERMINISTIC local Fisher gate (from joint_identifiability on the stored views), and
    `identifiable` is a single-sample (N=1, high-variance) Monte-Carlo probe — does one
    noisy recovery still land within `identifiable_tol`. Errors are RELATIVE for both C10
    and mu, so `identifiable_tol` means the same thing for each."""
    from lumen.sensors.device_as_sensor import estimate_wall_and_friction, joint_identifiability

    fwd = dict(load=calib["load"], R0=calib["R0"], bulge_dir=tuple(calib["bulge_dir"]),
               push=calib["push"], normal_load=calib["normal_load"], k_axial=calib["k_axial"],
               **({"axis": tuple(calib["axis"])} if "axis" in calib else {}),
               **calib.get("dev_kw", {}))
    true_C10, true_mu = float(calib["true_C10"]), float(calib["true_mu"])

    def _recover(ts):
        C10, mu, hist = estimate_wall_and_friction(ts, sensor, carms, init_C10=init_C10,
                                                   init_mu=init_mu, iters=iters, **fwd)
        return float(C10), float(mu), hist

    def _rel_mu(mu):                               # relative mu error (floor avoids /0 at true_mu=0)
        return abs(mu - true_mu) / max(true_mu, 1e-6)

    C10, mu, hist = _recover(targets)
    fisher = joint_identifiability(true_C10, true_mu, sensor, {"views": carms}, **fwd)["views"]
    out = {"recovered_C10": C10, "true_C10": true_C10, "rel_error_C10": abs(C10 - true_C10) / true_C10,
           "recovered_mu": mu, "true_mu": true_mu, "rel_error_mu": _rel_mu(mu),
           "abs_error_mu": abs(mu - true_mu), "n_views": len(carms), "history": hist,
           "fisher_cond": fisher["cond"], "fisher_lam_min": fisher["lam_min"],
           "fisher_corr": fisher["corr"]}
    if noise_std > 0:                              # single-sample MC identifiability probe (H1)
        rng = np.random.default_rng(noise_seed)
        noised = [t + rng.normal(0.0, noise_std, np.shape(t)) for t in targets]
        C10n, mun, _ = _recover(noised)
        rel_C10_n, rel_mu_n = abs(C10n - true_C10) / true_C10, _rel_mu(mun)
        out.update(recovered_C10_noisy=C10n, recovered_mu_noisy=mun, noise_std=noise_std,
                   rel_error_C10_noisy=rel_C10_n, rel_error_mu_noisy=rel_mu_n,
                   identifiable=bool(rel_C10_n < identifiable_tol and rel_mu_n < identifiable_tol))
    return out


if __name__ == "__main__":  # self-check (needs warp/newton importable; math is numpy)
    import tempfile

    from lumen.sensors import FluoroSensor

    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=90, nu=44, nv=44)
    # biplanar probe at a known stiffness -> recover it
    from lumen.sensors.device_as_sensor import device_on_wall
    nodes = device_on_wall(6e3)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))
    ep = probe_episode(6e3, sensor, carms=[cx, cy])
    res = calibrate_from_episode(ep, init_C10=2e3, iters=16)            # in-memory
    assert res["rel_error"] < 0.1, res
    with tempfile.TemporaryDirectory() as d:                            # round-trip then calibrate
        ep.save(d)
        from lumen.data.replay import EpisodeDataset
        loaded = EpisodeDataset(d)[0]
        res2 = calibrate_from_episode(loaded, init_C10=2e3, iters=16)
        assert res2["rel_error"] < 0.1 and res2["n_views"] == 2
    print(f"calibrate self-check ok (recovered {res['recovered_C10']:.0f} vs 6000)")
