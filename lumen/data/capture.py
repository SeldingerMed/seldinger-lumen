"""L2.1 — synthetic capture: turn a Layer-0 rollout into a Layer-2 Episode.

Drives a `NewtonGuidewireSim` and logs, per step, the device kinematics + the paired
Layer-1 observation (a fluoro frame or a luminal RGB frame) + the outcome, into the
`lumen-episode/0` schema. This is the OPEN end of the capture seam: it produces
`provenance="procedural"` episodes that exercise the whole pipeline (and feed the
sim2sim calibration harness, L2.3). A private patient-capture pipeline emits the
same `Episode` object — never in this repo.

`EpisodeRecorder` is the low-level driver (you call `record_step` yourself);
`rollout_episode` is the convenience that builds the sim from an `Asset`, runs a
policy to a target, and returns a finished `Episode`.

ponytail: the fluoro C-arm is sized to the whole VESSEL once and reused across steps
(the L1.3 lesson — a per-step device-sized view puts tip/target off-detector). Node
positions are logged every step as a sidecar; pass record_nodes=False to skip them.
"""

from __future__ import annotations

import numpy as np

from lumen.data.metrics import compute_clinical_metrics
from lumen.data.schema import Episode, EpisodeMeta, Outcome, Step, _is_bare_file_ref, validate


class SimDiverged(RuntimeError):
    """The Layer-0 sim produced non-finite node positions (blew up); the step is garbage."""


def _sensor_meta(sensor, modality) -> dict:
    if sensor is None:
        return {"modality": modality}
    if modality == "fluoro":   # res + n_samples are render knobs a replay must reproduce
        return {"modality": modality, "mu_device": sensor.mu_device, "eps": sensor.eps,
                "res": sensor.res, "n_samples": sensor.n_samples, "margin": sensor.margin,
                "nu": sensor.nu, "nv": sensor.nv}
    if modality == "luminal":
        return {"modality": modality, "fov_deg": sensor.fov_deg, "nu": sensor.nu,
                "nv": sensor.nv, "n_steps": sensor.n_steps, "max_dist": sensor.max_dist,
                "falloff": sensor.falloff, "ambient": sensor.ambient,
                "tissue_color": list(sensor.tissue_color),
                "texture_strength": sensor.texture_strength,
                "fold_strength": sensor.fold_strength,
                "artifact_strength": sensor.artifact_strength,
                "artifact_seed": sensor.artifact_seed}
    return {"modality": modality}


def _scope_intrinsics(sensor) -> dict:
    return {"model": "pinhole", "fov_deg": sensor.fov_deg, "nu": sensor.nu, "nv": sensor.nv}


def _scope_calibration(sensor) -> dict:
    return {"type": "scope", "intrinsics": _scope_intrinsics(sensor),
            "render": {k: v for k, v in _sensor_meta(sensor, "luminal").items()
                       if k != "modality"}}


def _clinical_step_signals(sim, frame, nodes, R):
    """Best-effort live sim signals for clinical endpoint metrics."""
    kin = {}
    wall = getattr(getattr(sim, "solver", None), "_wall", None)
    if wall is None:
        wall = getattr(getattr(sim, "solver", None), "_tree_wall", None)
    if wall is not None and getattr(wall, "wall_load", None) is not None:
        wl = np.asarray(wall.wall_load.numpy(), dtype=float)
        if wl.size:
            kin["wall_force_max"] = float(np.nanmax(wl))
            kin["wall_force_sum"] = float(np.nansum(wl))
    if R:
        kin["max_penetration"] = float(max(0.0, sim.node_radii().max() - R))
    if getattr(sim, "tree", None) is not None:
        try:
            kin["edge"] = sim.tree.project(nodes[-1]).edge_id
        except Exception:
            pass
    flow = getattr(sim, "flow", None)
    if flow is not None:
        for key, fn in (("flow_downstream_Q", flow.downstream_Q), ("flow_baseline_Q", flow.Q)):
            try:
                kin[key] = float(fn())
            except Exception:
                pass
    clot = getattr(sim, "clot", None)
    if clot is not None:
        kin["clot_damage_max"] = float(clot.max_damage())
        kin["clot_occlusion_max"] = float(np.max(clot.o)) if np.size(clot.o) else 0.0
    retrieval = getattr(sim, "last_retrieval", None)
    if isinstance(retrieval, dict):
        kin["retrieval_status"] = retrieval.get("status", "none")
    cath = sim.catheter_positions()
    if len(cath):
        cath_s = frame.project(cath[-1]).s
        tip_s = frame.project(nodes[-1]).s
        kin["catheter_tip_s"] = float(cath_s)
        kin["support_gap"] = float(tip_s - cath_s)
    return kin


class EpisodeRecorder:
    """Wrap a running `NewtonGuidewireSim`; `record_step(action)` advances it and logs
    one `Step`. `sensor` is the Layer-1 renderer for `modality` (a `FluoroSensor` for
    "fluoro", a `LuminalCamera` for "luminal"); "none" logs kinematics only."""

    def __init__(self, sim, sensor=None, modality="fluoro", lumen=None, every=1,
                 dt=5e-3, substeps=4, view_axis=(1.0, 0.0, 0.0), record_nodes=True,
                 meta: EpisodeMeta | None = None):
        if modality not in ("none", "fluoro", "luminal"):
            raise ValueError(f"modality must be 'none'|'fluoro'|'luminal', got {modality!r}")
        if modality != "none" and sensor is None:
            raise ValueError(f"modality={modality!r} needs a sensor (renderer)")
        if modality == "luminal" and lumen is None:
            raise ValueError("luminal modality needs the lumen field")
        if getattr(sim, "n_envs", 1) != 1:                # H1: capture is single-env; a batched
            raise ValueError(                             # sim's body_positions concatenates all
                f"EpisodeRecorder needs a single-env sim, got n_envs={sim.n_envs} "
                "(body_positions would mix envs; tip/render would be wrong)")
        self.sim = sim
        self.frame = sim.contact_frame              # the Layer-0 centerline frame
        self.sensor, self.modality, self.lumen = sensor, modality, lumen
        self.every = max(1, int(every))
        self.dt, self.substeps = dt, substeps
        self.record_nodes = record_nodes
        # L1.3 lesson: a fixed vessel-sized C-arm, not a per-step device-sized one.
        self.carm = (sensor.default_carm(np.asarray(self.frame.points), axis=view_axis)
                     if modality == "fluoro" else None)
        self.meta = meta or EpisodeMeta()
        if not self.meta.calibration:
            if modality == "fluoro":
                self.meta.calibration = {"type": "carm", "views": [self.carm.to_dict()],
                                         "active_view": 0, "view_axis": list(view_axis)}
            elif modality == "luminal":
                self.meta.calibration = _scope_calibration(sensor)
        self.steps: list = []
        self._t = 0.0

    def record_step(self, action) -> Step:
        if np.isscalar(action):
            action = {"insertion": float(action)}
        self.sim.step(dt=self.dt, substeps=self.substeps,
                      insertion=float(action.get("insertion", 0.0)),
                      twist=float(action.get("twist", 0.0)),
                      aspiration=float(action.get("aspiration", 0.0)))
        i = len(self.steps)
        nodes = np.asarray(self.sim.body_positions())
        if not np.all(np.isfinite(nodes)):               # M3: guard divergence at the source,
            raise SimDiverged(f"non-finite node positions at step {i}")  # not via a later validate
        pr = self.frame.project(nodes[-1])
        kin = {"tip_mm": [float(x) for x in nodes[-1]], "tip_s": float(pr.s),
               "tip_r": float(pr.r), "max_r": float(self.sim.node_radii().max())}
        kin.update(_clinical_step_signals(self.sim, self.frame, nodes, getattr(self.sim, "R", 0.0)))
        if self.record_nodes:
            kin["node_positions_ref"] = f"{i:04d}_nodes.npy"

        render = self.modality != "none" and (i % self.every == 0)
        obs_arr, obs_ref, modality = None, None, "none"
        if render:
            modality, obs_ref = self.modality, f"{i:04d}.npy"
            if self.modality == "fluoro":
                obs_arr, _ = self.sensor.render(nodes, carm=self.carm)
            else:                                   # luminal: camera at the tip
                obs_arr = self.sensor.render(self.frame, self.lumen, nodes)

        step = Step(t=self._t, action=dict(action), kinematics=kin,
                    obs_modality=modality, obs_ref=obs_ref, force=None,
                    obs=obs_arr, node_positions=(nodes if self.record_nodes else None))
        self.steps.append(step)
        self._t += self.dt
        return step

    def episode(self, outcome: Outcome) -> Episode:
        return Episode(meta=self.meta, steps=self.steps, outcome=outcome)


def _state_obs(frame, sim, R, L, target_s):
    """The NavEnv 5-D state obs (a privileged driving signal; the EPISODE stores the
    image obs). Lets a cem-trained `make_policy` drive the capture unchanged."""
    pr = frame.project(np.asarray(sim.body_positions())[-1])
    return np.array([pr.s / L, pr.r / R, np.sin(pr.theta), np.cos(pr.theta),
                     (target_s - pr.s) / L], dtype=np.float32)


def rollout_episode(asset, policy=None, sensor=None, modality="fluoro",
                    target_frac=0.7, max_insertion=2.0, max_steps=40, success_tol=2.5,
                    every=1, substeps=4, dt=5e-3, device=None, asset_ref="asset.json", label=None,
                    notes=None, sim_kwargs=None, record_nodes=True, view_axis=(1.0, 0.0, 0.0),
                    procedure="endovascular_navigation"):
    """Build a sim from `asset`, drive it to the target with `policy` (None = constant
    forward insertion), record every step, and return a validated `Episode`.

    `policy(obs)->action` consumes the 5-D STATE obs and returns insertion in [-1, 1]
    (compatible with a state-obs `lumen.rl.cem.make_policy`). An image-obs policy (L1.3)
    needs an image-obs rollout — future work; the episode here still STORES the image obs.
    Raises if the rollout can't produce a valid episode (e.g. the sim diverges on step 0)."""
    from lumen.core.frame import CenterlineFrame
    from lumen.newton.sim import NewtonGuidewireSim

    if not asset.edges:
        raise ValueError("asset has no edges to roll out")
    pts, lumen = asset.edge_arrays(asset.edges[0])
    vessel = np.asarray(pts)
    R = float(np.asarray(lumen.R).mean())
    frame = CenterlineFrame(vessel)
    L = float(frame.length)
    target_s = target_frac * L
    # device + contact knobs recorded in meta.device so a replay/calibration harness can
    # reproduce the sim (M1 — these affect the kinematics, not just the geometry).
    radius, n_nodes, node_spacing = 0.2, 10, 2.0
    kappa, d_hat, vbd_iterations = 3e3, 0.3, 8
    p0, t0, m1 = frame.points[0], frame.tangents[0], frame.m1[0]
    device_points = (p0 + 0.5 * R * m1)[None, :] + np.arange(n_nodes)[:, None] * node_spacing * t0[None, :]
    sim = NewtonGuidewireSim(vessel, R, device_points, radius=radius, kappa=kappa, d_hat=d_hat,
                             lumen_field=lumen, vbd_iterations=vbd_iterations, device=device,
                             **(sim_kwargs or {}))

    resolved_label = label if label is not None else asset.edges[0].id
    guidewire = {"radius": radius, "n_nodes": n_nodes, "node_spacing": node_spacing,
                 "kappa": kappa, "d_hat": d_hat, "vbd_iterations": vbd_iterations}
    meta = EpisodeMeta(asset_ref=asset_ref, dt=dt, provenance=asset.provenance,
                       device={**guidewire, "guidewire": guidewire},
                       sensor=_sensor_meta(sensor, modality),
                       labels={"procedure": procedure, "asset_edge": asset.edges[0].id},
                       notes={**(notes or {}),
                              "episode_kind": "navigation",     # vs "wall_probe" (calibration)
                              "procedure": procedure,
                              "target_s": target_s, "target_frac": target_frac,
                              "success_tol": success_tol,
                              "perforation_penetration_threshold": d_hat})
    rec = EpisodeRecorder(sim, sensor=sensor, modality=modality, lumen=lumen, every=every,
                          dt=dt, substeps=substeps, view_axis=view_axis,
                          record_nodes=record_nodes, meta=meta)

    success = False
    dist = abs(float(frame.project(np.asarray(sim.body_positions())[-1]).s) - target_s)
    for _ in range(max_steps):
        obs = _state_obs(frame, sim, R, L, target_s)
        a = 1.0 if policy is None else float(np.asarray(policy(obs)).reshape(-1)[0])
        try:
            step = rec.record_step({"insertion": float(np.clip(a, -1.0, 1.0)) * max_insertion})
        except SimDiverged:                            # recorder guards divergence; dist stays last-good
            break
        dist = abs(step.kinematics["tip_s"] - target_s)
        if dist <= success_tol:
            success = True
            break

    ep = rec.episode(Outcome(success=success, final_dist=float(dist), steps=len(rec.steps),
                             # explicit None check: keep an intentional "" label, don't fall back
                             label=resolved_label))
    if _is_bare_file_ref(asset_ref):
        ep.asset = asset
    ep.outcome.metrics = compute_clinical_metrics(ep)
    validate(ep)                                        # M2: fail loud rather than return a bad episode
    return ep


if __name__ == "__main__":  # self-check (needs newton+warp): capture, save, reload, validate
    import tempfile

    from lumen.assets import procedural
    from lumen.sensors import FluoroSensor

    asset = procedural.straight_tube(80.0, 2.0)
    ep = rollout_episode(asset, sensor=FluoroSensor(res=20, nu=24, nv=24, n_samples=40),
                         max_steps=6, every=1, notes={"true_C10": 4000.0})       # validates internally
    assert ep.steps and ep.outcome.steps == len(ep.steps)
    assert ep.steps[-1].kinematics["tip_s"] >= ep.steps[0].kinematics["tip_s"]   # advanced
    assert ep.meta.device["kappa"] == 3e3 and "n_samples" in ep.meta.sensor      # M1 + sensor knobs
    with tempfile.TemporaryDirectory() as d:
        ep.save(d)
        back = Episode.load(d)
        validate(back, root=d)                       # every referenced sidecar exists
        assert back.steps[0].load_obs(d).shape == (24, 24)
    print("capture self-check ok")
