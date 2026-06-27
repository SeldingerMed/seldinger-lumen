# Episode schema (`lumen-episode/0`)

The Layer 2 data standard: a captured intervention as a time-synchronized log of
**device kinematics + the paired observation + outcome**, in a declared coordinate
frame. There is no Open X-Embodiment for endovascular intervention; this is the
schema meant to become that standard.

It is the time-series sibling of the [asset schema](../lumen/assets/schema.py)
(`lumen-asset/0`, which carries *geometry*). Both are emitted by two ends behind
one seam:

- the open `lumen.data.capture` recorder → `provenance = "procedural"`
- a private patient-capture pipeline → `provenance = "patient(private)"`

Patient-derived episodes never live in this repo. The firewall
(`tools/check_firewall.py`) scans every `*.json` and fails the build on any
top-level `provenance != "procedural"`.

## On-disk layout

One directory per episode:

```text
<episode>/
  manifest.json        # scalars: meta, per-step kinematics/actions/outcome, sidecar refs
  asset.json           # lumen-asset/0 geometry when asset_ref is a local filename
  obs/
    000.npy            # paired observation for step 0 (fluoro grayscale or luminal RGB)
    000_nodes.npy      # device node positions (n,3) for step 0   [optional]
    000_device_mask.npy # CV supervision mask for fluoro capture  [optional]
    ...
```

Observations are stored as `.npy` — lossless and dependency-free for both grayscale
fluoroscopy and RGB luminal frames. A viewer PNG is an example-side extra, not part
of the canonical load path; `examples/capture_episode.py` writes `preview.png` from
the first observation and `preview_contact_sheet.png` from the first/mid/last
observations, plus `device_mask_contact_sheet.png` for fluoro labels, for quick
visual inspection. Sidecars are **lazy-loaded** (`Step.load_obs(root)` /
`Step.load_nodes(root)`) so a large corpus iterates without exhausting memory.

`obs_ref` and `node_positions_ref` must be **bare filenames** (no path components,
no `..`): they are resolved under `<episode>/obs/`, and `validate` / `load_*` reject
anything that would escape it (a manifest is a trust boundary — it can arrive from
elsewhere). `validate` also rejects duplicate refs across steps (a reused name would
clobber an earlier step's sidecar).

## Manifest structure

```jsonc
{
  "version": "lumen-episode/0",
  "provenance": "procedural",          // TOP-LEVEL — where the firewall looks
  "meta": {
    "frame": { "name": "voxel_scaled", "spacing_mm": [...], "origin_mm": [...] },
    "asset_ref": "straight.json",      // the lumen-asset/0 geometry this ran on
    "device": { "guidewire": { "radius": 0.2, ... }, ... }, // device definitions/knobs
    "sensor": { "modality": "fluoro", "nu": 128, ... },
    "calibration": {
      "type": "carm",
      "views": [{ "source": [...], "detector_center": [...], "width": 60.0, ... }]
    },
    "labels": { "procedure": "endovascular_navigation", "anatomy": "straight_tube" },
    "dt": 0.005,
    "notes": { "episode_kind": "navigation", "true_C10": 4000.0 },  // free-form; see below
    "provenance": "procedural",
    "version": "lumen-episode/0"
  },
  "steps": [
    {
      "t": 0.0,
      "action": { "insertion": 1.0, "twist": 0.0, "aspiration": 0.0 },
      "kinematics": { "tip_mm": [x,y,z], "tip_s": 0.0, "tip_r": 0.1, "max_r": 0.2,
                      "node_positions_ref": "000_nodes.npy" },
      "annotations": {
        "device_mask_ref": "000_device_mask.npy",
        "keypoints": {
          "tip": { "uv": [u, v], "present": true },
          "base": { "uv": [u, v], "present": true }
        }
      },
      "obs_modality": "fluoro",        // "fluoro" | "luminal" | "none"
      "obs_ref": "000.npy",
      "force": null                     // measured where instrumented; null for procedural
    }
  ],
  "outcome": {
    "success": true,
    "final_dist": 0.4,
    "steps": 40,
    "retrieval": null,
    "label": "straight",
    "metrics": {
      "tip_target": { "success": true, "final_dist": 0.4 },
      "branch_choice": { "target": "left", "final": "left", "correct": true },
      "wall_safety": { "max_wall_force": 1.2, "max_penetration": 0.0,
                       "perforation_risk": false },
      "clot": { "retrieval": "retrieve", "fragmentation": false,
                "distal_emboli_proxy": 0.0 },
      "flow": { "baseline_Q": 4.0, "final_Q": 3.6, "restoration": 0.9,
                "restored": true },
      "catheter_support": { "final_gap": 3.0, "supported": true }
    }
  }
}
```

## Case bundles

`Episode` is intentionally permissive enough to load older manifests and partial
records for repair. A **case bundle** is the stricter replayable directory contract:

- local `asset_ref` sidecar with `lumen-asset/0` geometry
- `meta.calibration` with C-arm views for fluoro or scope intrinsics for luminal
- `meta.device` device definitions/knobs
- step actions, observations, node positions, outcome, and labels
- optional per-step CV annotations such as fluoro device masks and projected keypoints

Use `CaseBundle.load(root)` when a consumer needs a self-contained case rather than
a loose episode. It validates the sidecars, loads the asset, attaches the episode
root, and exposes `bundle.replay()` so observations are lazy-loaded from the same
directory.

Use `lumen.data.annotation_coverage(ep)` or corpus-level `summarize(...)` before
loading arrays to check whether a bundle has CV-ready masks/keypoints. Keypoint
coverage reports present counts per name (`base`, `tip`, `nodes`) rather than only
metadata presence. Root-mode validation also checks known `device_mask` annotations
are 2-D bool/unsigned masks whose shape matches the paired observation, and present
keypoints are finite in-frame `(u, v)` coordinates.

## Clinical Metrics

`outcome.metrics` is the canonical endpoint summary produced by
`lumen.data.compute_clinical_metrics(ep)` and populated by synthetic capture. The
metrics are named to match review questions rather than generic RL reward terms:

- `tip_target` — target-band success and final tip-target distance
- `branch_choice` — target branch/edge, final branch/edge, and correctness
- `wall_safety` — peak wall force, penetration, normalized risk score, and perforation-risk flag
- `clot` — retrieval/slip/fragment status, fragmentation, damage, residual occlusion, and distal-emboli proxy
- `flow` — baseline distal flow, final distal flow, restoration fraction, and restored flag
- `catheter_support` — final/min/max guidewire-catheter support gap and unsupported lead

The recorder fills these from live sim signals when present: solver wall load,
tree edge projection, clot damage/retrieval status, downstream flow, and coaxial
catheter tip position. Missing subsystems degrade to `null`/`false` rather than
inventing a measurement.

## Episode kinds

`notes["episode_kind"]` discriminates how an episode was produced (default
`"navigation"` for episodes predating the field):

- **`"navigation"`** — a rollout (L2.1): `steps` are timesteps, `outcome.success`/
  `final_dist` are navigation metrics. `summarize` computes its rates over these.
- **`"wall_probe"`** — a calibration probe (L2.3): `steps` are *views* of one static
  scene (so `t` indexes the view and `dt=0`), not a time series. The calibration
  ground truth and forward-model block lives in `notes["calib"]`; the canonical C-arm
  geometry lives in `meta.calibration["views"]`. `summarize` excludes probes from
  navigation rates.

## What it deliberately does NOT carry

Wall mechanics (HGO stiffness/fragility) are the **calibration target**, not an
input — they stay private (§8), exactly as in the asset schema. An episode carries
geometry references, observations, kinematics, and outcome; recovering the physics
from those is the job of the calibration harness (`lumen.data.calibrate`).

## Python API

```python
from lumen.data import CaseBundle, Episode, EpisodeMeta, Step, Outcome, replay, validate

ep = Episode(meta=EpisodeMeta(asset_ref="straight.json", dt=5e-3), steps=[...], outcome=...)
validate(ep)            # shape / monotonic-time / finite / provenance / version checks
ep.save("episodes/ep_0001")
back = Episode.load("episodes/ep_0001")
frame = back.steps[10].load_obs("episodes/ep_0001")   # lazy sidecar read
mask = back.steps[10].load_annotation("episodes/ep_0001", "device_mask")
bundle = CaseBundle.load("episodes/ep_0001")           # stricter self-contained replay contract
for t, action, kinematics, obs, annotations in replay(bundle.episode, include_annotations=True):
    mask = annotations.get("device_mask")               # lazy CV label array, if present
```

Versioning: `SCHEMA_VERSION` bumps on any breaking change to the manifest shape;
`validate` pins the version so stale episodes fail loudly rather than silently.
