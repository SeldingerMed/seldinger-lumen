You are the mandatory Seldinger engineering review subagent.
Required engine: openai-codex/gpt-5.5 / ChatGPT 5.5.

Repo: /Users/colin/Desktop/projects/seldinger/worktrees/seldinger-lumen-task131
GitHub issue/task: SeldingerMed/seldinger-lumen#54 / task 137, port stent-retriever clot retrieval to batched envs.
Existing PR being extended due overlap: #76 feat/task131-batched-stentriever.

Changed files/diff: read agentops/reports/review-diff-task137.diff.

Intent / PR body update:
## Summary
- Finalizes the #54 solver-support contract now that batched stent-retriever retrieval runs through the FlowField/clot device-coupling path.
- Keeps batched retrieval state independent across envs, including per-env clot occlusion, damage, mask, retrieved distance, stentriever engagement, and scalar or per-env aspiration commands.
- Adds a two-env batched retrieval regression where one env retrieves under aspiration while the other fragments, plus existing divergent capture/slip coverage.

## Tests
- python -m pytest tests/test_newton_batched.py::test_batched_stentriever_accepts_per_env_aspiration -q (RED failed before fix with ValueError: truth value of array aspiration is ambiguous)
- python -m pytest tests/test_newton_batched.py::test_batched_stentriever_accepts_per_env_aspiration tests/test_newton_batched.py::test_batched_stentriever_retrieval_diverges_per_env -q
- python -m pytest tests/test_newton_stentriever.py tests/test_newton_batched.py::test_batched_stentriever_retrieval_diverges_per_env tests/test_newton_batched.py::test_batched_stentriever_accepts_per_env_aspiration tests/test_solver_support_docs.py -q
- python -m ruff check .
- python -m pytest -q

Risk/data/license/PHI/secrets:
- No secrets, credentials, PHI, private clinical data, or new external assets.
- Apache-2.0 open-core repo; changes are simulator code/tests/docs only, no CathSim/private calibration assets.

Review output contract:
Return exactly one of:
REVIEW_RESULT: PASS
Required fixes: none
Notes: ...

or:
REVIEW_RESULT: CHANGES_REQUESTED
Required fixes:
1. ...
Evidence: ...
