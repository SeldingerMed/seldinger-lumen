from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_solver_support_matrix_tracks_batched_guardrails():
    support = (ROOT / "docs" / "SOLVER_SUPPORT.md").read_text()
    sim = (ROOT / "lumen" / "newton" / "sim.py").read_text()

    required_guards = [
        ("coaxial assemblies are single-env", "coaxial assemblies are single-env"),
        ("batched flow requires the 1-D FlowField", "batched flow requires the 1-D FlowField"),
        (
            "batched stent-retriever retrieval is not ported",
            "batched stent-retriever retrieval is not ported",
        ),
        ("tree contact is single-env", "tree contact is single-env"),
        (
            "tree contact takes R0 from each edge's lumen field",
            "tree contact takes R0 from each edge",
        ),
        ("tree + flow/clot is not wired", "tree + flow/clot is not wired"),
        ("an aneurysm needs the 1-D FlowField", "an aneurysm needs the 1-D FlowField"),
        ("aneurysm flow diversion is single-env", "aneurysm flow diversion is single-env"),
    ]
    for doc_guard, source_guard in required_guards:
        assert source_guard in sim
        assert doc_guard in support

    for issue_ref in ("#53", "#54", "#55", "#56"):
        assert issue_ref in support


def test_readme_and_architecture_link_solver_support_matrix():
    readme = (ROOT / "README.md").read_text()
    architecture = (ROOT / "ARCHITECTURE.md").read_text()

    link = "docs/SOLVER_SUPPORT.md"
    assert link in readme
    assert link in architecture
