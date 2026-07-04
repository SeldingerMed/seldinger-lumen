import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _not_implemented_messages(source: str) -> set[str]:
    tree = ast.parse(source)
    messages = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        if not isinstance(func, ast.Name) or func.id != "NotImplementedError":
            continue
        if node.exc.args and isinstance(node.exc.args[0], ast.Constant):
            messages.add(str(node.exc.args[0].value))
    return messages


def test_solver_support_matrix_tracks_batched_guardrails():
    support = (ROOT / "docs" / "SOLVER_SUPPORT.md").read_text()
    sim = (ROOT / "lumen" / "newton" / "sim.py").read_text()
    not_implemented_messages = _not_implemented_messages(sim)

    required_guards = [
        ("coaxial assemblies are single-env (batched coaxial is future)", "coaxial assemblies are single-env"),
        (
            "batched flow requires the 1-D FlowField; the lumped NewtonFlow is single-env (analytic fallback)",
            "batched flow requires the 1-D FlowField",
        ),
        (
            "batched stent-retriever retrieval is not ported (per-env host force balance); run retrieval single-env",
            "batched stent-retriever retrieval is not ported",
        ),
        (
            "tree contact takes R0 from each edge's lumen field; a sim-level lumen_field doesn't apply",
            "tree contact takes R0 from each edge's lumen field",
        ),
        (
            "edge-aware tree flow/clot coupling is not wired yet: flow drag and clot grids need per-edge graph fields, not a single route centerline",
            "edge-aware tree flow/clot coupling is not wired yet",
        ),
        (
            "an aneurysm needs the 1-D FlowField (it reads the neck pressure P(s)); pass flow=FlowField(...)",
            "an aneurysm needs the 1-D FlowField",
        ),
        (
            "aneurysm flow diversion is single-env (the sac reads the host pressure field)",
            "aneurysm flow diversion is single-env",
        ),
    ]
    for source_guard, doc_guard in required_guards:
        assert source_guard in not_implemented_messages
        assert doc_guard in support

    assert "| 1-D `FlowField` coupling | ✅ | ✅ | none | — |" in support
    assert "| Vascular-tree contact | ✅ | ✅ | none | — |" in support
    for issue_ref in ("53", "54", "55", "56"):
        assert f"| #{issue_ref} |" not in support
        assert f"[#{issue_ref}](https://github.com/SeldingerMed/seldinger-lumen/issues/{issue_ref})" in support

    assert "## Follow-up implementation tracker" in support
    for closure_evidence in (
        "two-env coaxial construction/step test",
        "two-env retrieval test",
        "two-env tree contact test",
        "two-env aneurysm test",
    ):
        assert closure_evidence in support


def test_readme_and_architecture_link_solver_support_matrix():
    readme = (ROOT / "README.md").read_text()
    architecture = (ROOT / "ARCHITECTURE.md").read_text()

    link = "docs/SOLVER_SUPPORT.md"
    assert link in readme
    assert link in architecture
