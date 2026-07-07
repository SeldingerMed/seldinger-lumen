You are the mandatory Phase 3 Seldinger engineering reviewer.

Required engine: openai-codex/gpt-5.5 / ChatGPT 5.5.
Return exactly one of:
REVIEW_RESULT: PASS
Required fixes: none
Notes: <optional>

or:
REVIEW_RESULT: CHANGES_REQUESTED
Required fixes:
1. <specific actionable fix>
Evidence: <file/line/test/command rationale>

Repo: /Users/colin/Desktop/projects/seldinger/worktrees/seldinger-lumen-task131
Repo boundary: SeldingerMed/seldinger-lumen, open-core Apache-2.0. Do not introduce PHI/private clinical data, secrets, private calibration, or non-Apache-compatible assets.
Task/issue: SeldingerMed/seldinger-lumen#54 — Port stent-retriever clot retrieval to batched envs.

Change summary:
- Fix docs/SOLVER_SUPPORT.md after batched stent-retriever support landed: remove unresolved conflict markers, remove #54 from remaining gap tracker, and document the supported FlowField/clot-coupled batched retrieval path and remaining lumped NewtonFlow limit.
- Update tests/test_solver_support_docs.py so the support matrix no longer expects the obsolete "batched stent-retriever retrieval is not ported" guard, and assert docs contain no conflict markers or stale #54 tracker link.
- Add tests/test_newton_stentriever.py::test_batched_retrieve_keeps_fragmentation_independent to verify two-env batched retrieval keeps fragmentation/retrieval/damage state independent: env 0 retrieves under aspiration while env 1 fragments with no retrieval.
- Type-annotate ClotField.retrieve_batched aspiration as float | np.ndarray because batched retrieval already supports per-env aspiration arrays via np.broadcast_to.

Diff to review:
Run: git -C /Users/colin/Desktop/projects/seldinger/worktrees/seldinger-lumen-task131 diff -- docs/SOLVER_SUPPORT.md lumen/newton/clot.py tests/test_newton_stentriever.py tests/test_solver_support_docs.py

Local checks run:
- python -m pytest tests/test_newton_stentriever.py tests/test_newton_batched.py::test_batched_stentriever_retrieval_diverges_per_env tests/test_solver_support_docs.py -q -> 9 passed in 1.67s
- python -m ruff check . -> All checks passed
- python -m pytest -q -> 355 passed in 334.30s

Intended PR body:
## Summary
- Finalizes the #54 solver-support contract now that batched stent-retriever retrieval runs through the FlowField/clot device-coupling path.
- Removes stale conflict markers / obsolete #54 tracker text from the support matrix and keeps the docs test aligned with the current runtime guards.
- Adds a two-env batched retrieval regression where one env retrieves under aspiration while the other fragments, proving damage/retrieved state stays per-env.

## Tests
- python -m pytest tests/test_newton_stentriever.py tests/test_newton_batched.py::test_batched_stentriever_retrieval_diverges_per_env tests/test_solver_support_docs.py -q
- python -m ruff check .
- python -m pytest -q

## Risk / data / license / PHI
- No secrets, credentials, PHI, private clinical data, or new external assets.
- Apache-2.0 repo; changes are docs/tests plus a type annotation for an existing supported numpy-array code path.

Fixes #54

Review requirements:
- Verify this PR should close #54 given the existing batched implementation in master and this follow-up doc/test hardening.
- Verify no obsolete guard or conflict marker remains in docs/tests.
- Verify the new fragmentation test correctly proves per-env independence and is not brittle.
- Verify local checks are appropriate.
