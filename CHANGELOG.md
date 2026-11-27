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
- Which vessel holds a point near a junction is decided as the contact physics decides it:
  by the distance to the nearest point of each vessel's axis, so past a vessel's end, the
  distance to the end point (it was measured across the axis carried on beyond the end).
  On the aortic arch the two agreed for 42 of 100 points near the junctions and now agree
  for all. The wall gap and direction are unchanged. Wall contact near junctions changes,
  so the benchmark suites move to `lumen-bench/6` and `lumen-bench/heldout/4` (#153).
- A tree-navigation step whose simulation blew up is marked off the route, so the benchmark's
  clinical target check cannot count it as a reach. Containment works before the solver keeps
  a previous state. An asset whose device spawn names a node it does not have is refused when
  its vessel tree is built.
- Imported vessels no way in reaches are left out of `names.json` instead of being listed with
  no target. An older `names.json` row kept by hand gets the route length of its own target
  and, when its edge list is empty, the vessel's named edges.
- A pushed wire now carries the push to its tip on real anatomy. The rod solver ran 8
  iterations for a 16-node wire and a push travels about one node per iteration, so the tip
  got about a third of each push and the wire bowed and prolapsed; `TreeNavEnv` now runs
  twice as many iterations as nodes (`vbd_iterations`), passing about 0.8 of a push. The
  solver has no damping, so after a pull-back the wire coasted and folded round its held
  end; with `quasi_static` (default on) the wire stops when the hand stops or reverses, as in
  blood. The solver substep is 2.5 ms (`substep_ms`), with lighter bend damping, a 45 degree
  shaped tip and the wire's rest shape laid in its starting vessel, which together pass 0.99 of
  a push through a carotid-siphon S-bend.
- The tip's place along the route is tracked inside the vessels, and progress, reward and
  success use it: a tip in the wrong branch that happens to lie near the route's centreline
  earns nothing (`info["tip_mm"]`, `info["furthest_mm"]`). On a real scan with junctions every
  few millimetres, a point just past a join no longer counts as the end of the vessel before.
- A head scan whose carotids or vertebrals were cut at the bottom of the picture offers those
  cut ends as ways in: each named vessel records the start the wire is brought up from, so the
  right middle cerebral artery is reached up the right carotid (`names.json` `access`). The
  entry end of a scanned artery now holds its width like every other cut end.
- `FollowPolicy` (used by `lumen view --run`) judges each push by its own progress, backs off
  to 5 mm short of the branch point it stuck at, turns the shaped tip 30 degrees and tries
  again. On the same 24 wire routes of the shipped real cases, a wire steered this way now
  reaches 10 (5 with the earlier physics and a scripted push-and-retry operator), with mean
  progress 0.82 (0.67) and at most 0.47 mm into the wall (0.30).
- The benchmark bifurcation (`make_tree_nav`) starts the wire with its shaped tip turned
  toward the target branch (`TreeNavEnv(aim_tip=True)`), as an operator turns it before the
  fork, so a wire that is only pushed enters either branch with no wall penetration. Elsewhere
  the tip starts in one fixed direction (`aim_tip=False`, the default) and a policy chooses the
  branch by turning the wire.
- Scoring no longer flatters the wire. The deepest wall penetration of each step is counted
  before containment puts a node back (it was measured after, so a node 0.6 mm through the
  wall could be scored as safe). A tip read further along the route than one step can carry it
  is believed only if the wire itself lies along the route back to where the tip was;
  waiting there no longer makes the reading count. A point past the end of a vessel is no
  longer inside it. The benchmark's clinical target check uses the tracked tip and
  requires it to be in the route.
- A named vessel's route runs through that vessel: the left anterior cerebral artery of the
  shipped aneurysm CTA was reached up the right carotid through the right one, and is now
  reached up the left. A target that is itself a way in is reached from another, a target no
  way in reaches raises `ValueError`, `lumen import --entry` makes the given entry the only
  way in, and an older `names.json` keeps its hand-edited targets and names.
- A route shorter than the default 30 mm wire gets a shorter wire instead of failing.
- These change `TreeNavEnv` dynamics, so the benchmark suites move to `lumen-bench/5` and
  `lumen-bench/heldout/3`.
- The wire stays in the vessel on imported anatomy. Five causes, each fixed: past the open far
  end of a scanned artery the wall force switched off (the wall now continues past it); where
  a thin branch left a wide vessel, a wire node inside the wide vessel was judged against the
  thin branch's wall (each node now belongs to the branch where it has the most room); a
  scanned artery cut off at the scan edge closed to a point (its radius now holds over that
  cap, except at the entry, where the wire starts); the pushed end was driven with unlimited force into a jam (feeding now pauses while
  the wire presses into the wall, `feed_stop_pen_mm`, default 0.2 mm); and the wall was soft
  enough for a taut wire to cut across a tight bend (`TreeNavEnv` wall stiffness 3e3 -> 6e3,
  now the `wall_stiffness` argument). These change `TreeNavEnv` dynamics on generated anatomy
  too, so the benchmark suites move to `lumen-bench/4` and `lumen-bench/heldout/2`; older
  scores and replays are refused rather than compared.
- The Newton tree contact kernel picks a node's vessel with the same branch-blended radius
  R(s, theta) that `VascularTree.project` and the wall force use (it used a mean radius per
  centerline point), finding the nearest point of each vessel's axis first, so a wider section
  further along a vessel cannot take over a node in a narrowing. The wall-fit figure in import
  reports is measured under that same rule.
- Import no longer reads an untagged MR image as CT when its tissue happens to sit around
  1024: unsigned CT is recognised by the dip between fat and soft tissue, which MR does not
  have. `lumen import --modality CT|MR` overrides the reading.
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
