# Engineering review request: task 132 / issue #53

Reviewer engine required by supervisor: openai-codex/gpt-5.5 / ChatGPT 5.5.

Repo: /Users/colin/Desktop/projects/seldinger/.worktrees/seldinger-lumen-task132
Repo boundary: SeldingerMed/seldinger-lumen, Apache-2.0 open-core repo.
Issue: https://github.com/SeldingerMed/seldinger-lumen/issues/53
Task: Implement batched coaxial guidewire + catheter assemblies.

Summary of intended change:
- Removes the n_envs>1 constructor guard for coaxial catheter assemblies.
- Builds one contiguous guidewire+catheter assembly per env in the Newton model.
- Allocates per-env catheter base actuation arrays and accepts per-env catheter insertion/twist actions.
- Passes per-env assembly sizing into wall contact and coaxial guidewire-catheter coupling so bodies map to the correct env.
- Restricts coaxial coupling to the same env's catheter centerline and preserves two-way reaction force behavior.
- Updates solver support docs/tests and adds a two-env regression test that drives guidewire and catheter bases independently.

Local checks run:
- python -m pytest tests/test_newton_coaxial.py::test_batched_coaxial_envs_are_independent_under_per_env_actions tests/test_newton_coaxial.py::test_coaxial_builds_with_two_rods tests/test_newton_coaxial_coupling.py -q => 5 passed
- python -m pytest tests/test_newton_coaxial.py tests/test_newton_coaxial_coupling.py tests/test_newton_batched.py tests/test_solver_support_docs.py -q => 23 passed
- python -m ruff check . => All checks passed
- python -m pytest -q => 354 passed
- git diff --check => clean

Risk / secrets / PHI / license notes:
- No secrets, credentials, private URLs, PHI, patient data, or external assets added.
- Apache-2.0 repo; code is original implementation within existing Newton/Warp abstractions.
- Main risk is body-id-to-env indexing for batched catheter rods and per-env coupling; tests assert body layout and independent per-env guidewire/catheter motion.

Intended PR body:

## Summary
- Builds batched coaxial guidewire+catheter assemblies as contiguous per-env Newton body blocks.
- Adds per-env catheter actuation arrays and scopes coaxial coupling to each env's catheter centerline while preserving two-way reactions.
- Updates the solver support matrix and adds a two-env coaxial regression with independent guidewire/catheter actions.

## Tests
- python -m pytest tests/test_newton_coaxial.py tests/test_newton_coaxial_coupling.py tests/test_newton_batched.py tests/test_solver_support_docs.py -q
- python -m ruff check .
- python -m pytest -q

## Risk / data / license / PHI
- No secrets, credentials, PHI, private clinical data, or new external assets.
- Apache-2.0 repo; implementation uses existing Newton/Warp code paths.

Fixes #53

Review instructions:
Return exactly one of:
REVIEW_RESULT: PASS
Required fixes: none
Notes: ...

or:
REVIEW_RESULT: CHANGES_REQUESTED
Required fixes:
1. ...
Evidence: ...

Review the working tree diff at this path, plus run git diff if needed. Be strict about correctness, env indexing, coupling, tests, docs, secrets/PHI/license.
