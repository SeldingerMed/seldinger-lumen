# Contributing to lumen

Thanks for your interest! This is the open core of a medical-AI stack, so we hold a
few hard lines (license, no patient data) alongside the usual quality bar. Reading
this once will make your PR sail through.

## TL;DR

1. Sign your commits off: `git commit -s` (we require the [DCO](#developer-certificate-of-origin-dco)).
2. Keep `pytest` and the firewall check green.
3. One focused change per PR; fill in the PR template.
4. A maintainer (@txmed82) reviews every PR before it merges.

## Development setup

```bash
git clone https://github.com/SeldingerMed/seldinger-lumen
cd seldinger-lumen
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# the Newton engine, pinned to the commit our SolverVBD fork tracks (same as CI)
pip install "git+https://github.com/newton-physics/newton@6dfe7303d9ca50f7505cac31bee9885c813d89d7"
```

Newton is pre-1.0 and `TubeVBDSolver` forks one of its internals, so we pin a
known-good commit (`NEWTON_REF` in `.github/workflows/ci.yml`). Bumping it is a
deliberate PR with the test suite re-run, not an automatic upgrade.

Run the checks the way CI does:

```bash
pytest -q                       # the full suite (Newton-dependent tests skip if newton is absent)
python tools/check_firewall.py  # license + no-patient-data firewall
ruff check .                    # lint
```

The NumPy-only geometry core (`lumen.core`) and most logic tests run without Newton;
the Newton-backed tests skip cleanly if `warp`/`newton` aren't installed.

## The two firewalls (non-negotiable)

`tools/check_firewall.py` runs in CI and **will fail your PR** if either is violated:

1. **No [CathSim](https://github.com/robotvisionlabs/cathsim).** It's CC-BY-NC-SA-4.0;
   importing or vendoring any of it would contaminate our Apache-2.0 license. Don't
   reference it in code, tests, or assets.
2. **No patient data.** Every committed asset must be procedurally generated
   (`provenance="procedural"`). Real anatomy, scans, or measurements stay in the
   private Seldinger repos behind the `lumen.assets.schema` seam. Never commit a
   real centerline, mask, or patient-derived parameter.

If you need real-data behavior, calibrate privately and contribute the *procedural*
analog or the *interface*, not the data.

## What makes a good PR

- **Scope:** one logical change. Split unrelated fixes.
- **Tests:** non-trivial logic ships with a check. We favor one small, fast,
  assertion-based test that fails if the logic breaks over heavy fixtures. Match the
  style in `tests/` (kernel-level checks for kernels, analytic comparisons for
  physics).
- **Physics faithfulness:** this solver follows a specific architecture. If a change
  alters the contact, wall, friction, clot, or flow *model* (not just a bug fix),
  say why in the PR and cite the equation/source. Don't silently swap a constitutive
  law or a coupling.
- **No new runtime dependencies** without discussion — the core is intentionally
  NumPy + Warp/Newton only.
- **Docs:** update `README.md` / `ARCHITECTURE.md` if you change public behavior or
  layout.

## Adding a new modality or extension

The whole point of the architecture is that you extend it *without touching the core*
(see [ARCHITECTURE.md](ARCHITECTURE.md)). For extension PRs specifically:

- **New modality** (airway, bowel, ureter, …): add a directory under
  `lumen/profiles/<name>/`. It must not modify `lumen.core.*` or another profile. A
  profile bundles the anatomy field + instrument choice + (future) sensor.
- **New device / constitutive model:** add it alongside the existing ones
  (`lumen/newton/devices.py`, `clot.py`, etc.) behind the same interfaces; keep the
  defaults literature-grounded and cite them.
- **Accurate-tier oracle** (STARK/SymX, ppf-contact-solver): wire it into the
  drop-in slot in `lumen/newton/crossval.py`; don't reimplement IPC.
- An extension PR that requires a core change should first open an **issue** so we
  can keep the core's invariants intact.

## Developer Certificate of Origin (DCO)

We use the [DCO](https://developercertificate.org/) instead of a CLA. It's a
one-line certification that you wrote the patch (or have the right to submit it).
Add it by committing with `-s`:

```bash
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Every commit in a PR must be signed off; a CI check enforces it. Forgot? Amend
(`git commit --amend -s`) or rebase with `git rebase --signoff main`.

## Review & merge

- Every PR requires review from a maintainer (enforced via `CODEOWNERS` + branch
  protection). Please be patient — this is a small team.
- CI (tests + firewall + lint + DCO) must be green.
- We squash-merge; write a clear PR title.

## Reporting bugs / proposing features

Use the issue templates. For anything security- or data-sensitive, see
[SECURITY.md](SECURITY.md) — do **not** open a public issue.

By contributing, you agree your contributions are licensed under Apache-2.0.
