# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is pre-1.0 and
does not yet promise [SemVer](https://semver.org/) stability.

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

### Changed
- Leaner README and corrected `ARCHITECTURE.md` references (`tube_vbd.py`).
- Cardiac pulsatility modulates the vessel wall only, not the clot occlusion.

### Fixed
- Contact load was double-counted by the HGO wall and the clot; the wall now skips
  clot cells.
- Stent-retriever fragmentation now scales with the timestep via the damage law.
- Friction tangent Hessian carries the correct `1/dt` factor.
- NaN/inf guards in the clot update and the NavEnv divergence path.

_Earlier history predates the public release and lives in the git log._
