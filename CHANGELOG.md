# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Lumen is pre-1.0 and uses
[Semantic Versioning](https://semver.org/) for every published version.

## [Unreleased]

### Added
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
### Fixed
- Contact load was double-counted by the HGO wall and the clot; the wall now skips
  clot cells.
- Stent-retriever fragmentation now scales with the timestep via the damage law.
- Friction tangent Hessian carries the correct `1/dt` factor.
- NaN/inf guards in the clot update and the NavEnv divergence path.

_Earlier history predates the public release and lives in the git log._
