# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Lumen is pre-1.0 and uses
[Semantic Versioning](https://semver.org/) for every published version.

## [Unreleased]

### Added
- `lumen view`: a self-contained HTML viewer with a rotatable 3D view, a simulated C-arm image
  (AP, lateral, LAO/RAO, cranial/caudal), and step-by-step replay of recorded runs
  (`--run`, `--save-run`, `--replay`). Works on shipped cases, imported scans and generated
  anatomy. Three looks for the 3D view (`--look monitor|anatomy|glass`, switchable on the
  page).
- Five open real cases in `lumen/cases/` (head CTA with aneurysm, head and neck CTA, head
  TOF-MRA, aortic CTA, abdominal CTA), derived from CC BY 4.0 and CC0 public datasets with
  attribution in NOTICE. `lumen cases` lists them; `lumen.cases.make_env` drives one.
- Import reads unsigned CT files (stored with 1024 added) as Hounsfield units, and re-grids
  thin MRA arteries finer so small branches stay connected. An intracranial MRA slab that
  fills its picture is recognised as a head, and is segmented with the head model.
- Import checks each vessel cross-section against the vessel either side of it: a section cut
  along a neighbouring vessel, or a ray that runs along the wall of a thin artery, is rebuilt
  from its neighbours. Aneurysms, stenoses and a vessel widening into the aorta are kept.

### Fixed
- The wire stays in the vessel on imported anatomy. Five causes, each fixed: past the open far
  end of a scanned artery the wall force switched off (the wall now continues past it); where
  a thin branch left a wide vessel, a wire node inside the wide vessel was judged against the
  thin branch's wall (each node now belongs to the branch where it has the most room); a
  scanned artery cut off at the scan edge closed to a point (its radius now holds over that
  cap, except at the entry, where the wire starts); the pushed end was driven with unlimited force into a jam (feeding now pauses while
  the wire presses into the wall, `feed_stop_pen_mm`, default 0.2 mm); and the wall was soft
  enough for a taut wire to cut across a tight bend (`TreeNavEnv` wall stiffness 3e3 -> 6e3,
  now the `wall_stiffness` argument). These change `TreeNavEnv` dynamics on generated anatomy
  too.
- Segmentation no longer fails with a bare `EOFError` when the temp folder cannot hold a Unix
  socket (exFAT/FAT32 drives, network shares, long macOS paths).
- Vessel names live only in the `.names.json` sidecar beside a label volume (#151).
- `lumen import`: turn a real CTA or MRA (DICOM folder, NIfTI, NRRD), or a vessel mask, into
  anatomy the solver can navigate. Lumen works out the body region from the picture, picks the
  segmentation model (head: the RSNA 2025 intracranial model; neck to pelvis: TotalSegmentator;
  both when a scan shows both), asks before installing one into its own private environment,
  names the branches and writes an accuracy report. Runs on the user's machine. Needs the new
  `scan` extra. See `docs/import.md`.
- A two-view registration challenge with overlapping procedural forks, calibrated projections and offline pose scoring.
- Open-source project setup: CI (tests + firewall + lint across Python 3.10–3.12),
  DCO sign-off check, CONTRIBUTING / Code of Conduct / SECURITY, issue & PR
  templates, CODEOWNERS, Dependabot, and a GitHub Pages site.
- `FlowField`: a 1-D resistive-network blood-flow model along the centerline
  (pressure field `P(s)`, velocity field `v(s)`, aspiration as a pressure sink).
- Layer 1 sensor stack: `FluoroSensor` (differentiable DRR fluoroscopy), 2D/3D
  `register`, device-as-sensor wall-stiffness estimation, and an image-based RL
  observation (`FluoroBatchedNav`).
- Layer 1 (L1.4) realism seam `RealismParams` / `degrade`: calibratable detector
  physics on the DRR — Poisson photon noise, detector PSF blur, scatter glow, and
  beam hardening — off by default, threaded through `FluoroSensor.render(realism=…)`.
- Layer 1 (L1.4) second observation modality `LuminalCamera`: a forward-looking
  endoscopic RGB view from the device tip over the shared `R(s,θ)` lumen field,
  proving the sensor-swap invariant (same scene, different sensor).
- Layer 2 data standard & capture (`lumen.data`): the `lumen-episode/0` schema
  (`Episode` — kinematics + paired observation + outcome; `docs/EPISODE_SCHEMA.md`);
  synthetic capture (`EpisodeRecorder` / `rollout_episode`); corpus iteration and
  replay (`EpisodeDataset` / `replay` / `summarize`); and sim2sim wall-stiffness
  calibration that closes the §3.6 loop on an episode (`probe_episode` /
  `calibrate_from_episode`). Firewall-guarded like the asset seam; the real corpus
  stays private.
- Layer 4 deployment seam (`lumen.deployment`) with fail-closed safety envelopes and
  calibrated force/torque benchtop trace validation, including a deterministic torque
  whip proxy.
- Package version is now `0.2.0`; a CPU Docker image and CI Gymnasium environment
  checker make the install and registered-environment contracts executable.
- Replay-certified benchmark scorecards now bind every ranked metric to a SHA-256
  action/outcome certificate, with a replay-verification CLI and leaderboard API.
- Optional `lumen.rl` adapters now provide SB3 `Monitor` environments and
  CleanRL-compatible seeded thunks/vector environments without making either trainer a
  core dependency.

### Changed
- Leaner README and corrected `ARCHITECTURE.md` references (`tube_vbd.py`).
- Cardiac pulsatility now uses one rectified waveform for pressure and lumen radius,
  updates wall cell areas with the radius, and resets phase/coupling state between
  episodes; the clot remains non-pulsatile.
- Wall-load peaks, pressure peaks, and load impulses now accumulate across every
  converged public-step substep on device before diagnostic synchronization.
### Fixed
- Contact load was double-counted by the HGO wall and the clot; the wall now skips
  clot cells.
- Stent-retriever fragmentation now scales with the timestep via the damage law.
- Friction tangent Hessian carries the correct `1/dt` factor.
- NaN/inf guards in the clot update and the NavEnv divergence path.

_Earlier history predates the public release and lives in the git log._
