<!-- Thanks for contributing! Keep PRs focused — one logical change. -->

## What & why

<!-- What does this change, and why? Link any related issue (Fixes #123). -->

## Type

- [ ] Bug fix
- [ ] New feature / extension (new `lumen/profiles/<x>/`, device, or constitutive model)
- [ ] Physics-model change (alters contact / wall / friction / clot / flow behavior)
- [ ] Docs / tooling / CI
- [ ] Other:

## Physics faithfulness (if you changed a model)

<!-- Cite the equation/paper for any constitutive or coupling change, and say what
you verified. Skip if this is a pure bug fix or non-physics change. -->

## Checklist

- [ ] Commits are **signed off** (`git commit -s`, DCO)
- [ ] `pytest -q` passes locally
- [ ] `python tools/check_firewall.py` passes (no CathSim, no patient data)
- [ ] `ruff check .` passes
- [ ] Added/updated a **test** for non-trivial logic
- [ ] Updated `README.md` / `ARCHITECTURE.md` if public behavior or layout changed
- [ ] No new runtime dependency (or it was discussed in an issue first)
- [ ] If this is an extension, it does **not** modify `lumen/core/*` or another profile
