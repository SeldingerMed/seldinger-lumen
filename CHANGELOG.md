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
