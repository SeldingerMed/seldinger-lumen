"""Unit tests for the blind-first navigation report + replay certificate.

All tests use a tiny in-memory producer: they never import the Newton
engine, the rollout layer, or read any real case data. The integrated run
against ``lumen.experiments.image_rollout.run_case`` is exercised separately.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from lumen.experiments import navigation_report as nr


# ---------------------------------------------------------------------------
# Tiny in-memory producer (2 frames, 4x5 uint8, full new contract shape)
# ---------------------------------------------------------------------------

def _frame(h, w, fill):
    return {"shape": [h, w], "data": base64.b64encode(bytes([fill] * (h * w))).decode()}


def _result(case_id, tip=(1.0, 2.0, 3.0), reached=True):
    return {
        "case_id": case_id,
        "goal": "advance to the vessel landmark",
        "frames": [_frame(4, 5, 40), _frame(4, 5, 200)],
        "action": "insert 0.5",
        "hidden": {"tip_xyz_mm": [list(tip), [2.0, 2.0, 4.0]],
                   "reason": "split entropy below threshold"},
        "scores": {
            "target_reached": reached,
            "max_penetration_mm": 0.4,
            "peak_wall_load_sim_units": 12.5,
            "abstained": False,
        },
    }


CASE_IDS = ("case-a", "case-b", "case-c")
RESULTS = {cid: _result(cid, tip=(float(i), 1.0, 2.0), reached=(i != 1))
           for i, cid in enumerate(CASE_IDS)}
PRODUCER = RESULTS.__getitem__  # run_case(case_id) -> result


# ---------------------------------------------------------------------------
# Blind-first HTML report
# ---------------------------------------------------------------------------

def test_report_shows_frames_blind_without_hidden_content():
    html = nr.build_html([RESULTS[c] for c in CASE_IDS], case_labels=["Case 1", "Case 2", "Case 3"])
    # both frames of case-a rendered as image data
    assert "data:image/png;base64," in html
    # blind portion must NOT contain the hidden trajectory, reason or case ids
    before, sep, details = html.partition("<details")
    assert sep
    assert "target_reached" not in before and "Executed action" not in before
    assert "tip_xyz_mm" not in before
    hidden_reason = "split entropy below threshold"
    assert hidden_reason in details and hidden_reason not in before
    assert "case-a" not in before


def test_report_shows_all_scores_separately_and_action():
    html = nr.build_html([RESULTS[c] for c in CASE_IDS])
    for key in ("target_reached", "max_penetration_mm",
                "peak_wall_load_sim_units", "abstained"):
        assert key in html
    assert "insert 0.5" in html


def test_report_labels_cases_without_spoiler_ids():
    html = nr.build_html([RESULTS[c] for c in CASE_IDS])
    assert "Case 1" in html and "Case 3" in html


def test_report_renders_hidden_trajectory_visually():
    html = nr.build_html([RESULTS["case-a"]])
    assert "<svg" in html and "polyline" in html


def test_png_encoding_roundtrips_dimensions():
    png = nr.encode_gray_png(bytes(range(20)), 4, 5)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR: width=4, height=5
    assert int.from_bytes(png[16:20], "big") == 4
    assert int.from_bytes(png[20:24], "big") == 5


# ---------------------------------------------------------------------------
# Certificate: build + verify
# ---------------------------------------------------------------------------

def test_certificate_roundtrip_verifies():
    cert = nr.build_certificate(PRODUCER, CASE_IDS)
    assert nr.verify_certificate(cert, PRODUCER, CASE_IDS) is True


def test_certificate_rejects_tampered_result():
    cert = nr.build_certificate(PRODUCER, CASE_IDS)
    tampered = json.loads(json.dumps(cert))
    tampered["cases"]["case-b"]["result"]["scores"]["max_penetration_mm"] = 0.0
    with pytest.raises(nr.CertificateRejected):
        nr.verify_certificate(tampered, PRODUCER, CASE_IDS)


def test_certificate_rejects_tampered_digest():
    cert = nr.build_certificate(PRODUCER, CASE_IDS)
    tampered = json.loads(json.dumps(cert))
    tampered["cases"]["case-a"]["digest"] = "0" * 64
    with pytest.raises(nr.CertificateRejected):
        nr.verify_certificate(tampered, PRODUCER, CASE_IDS)


def test_certificate_rejects_missing_case():
    cert = nr.build_certificate(PRODUCER, CASE_IDS)
    del cert["cases"]["case-c"]
    with pytest.raises(nr.CertificateRejected):
        nr.verify_certificate(cert, PRODUCER, CASE_IDS)


def test_certificate_rejects_non_dict_and_wrong_schema():
    with pytest.raises(nr.CertificateRejected):
        nr.verify_certificate(["not", "a", "dict"], PRODUCER, CASE_IDS)
    with pytest.raises(nr.CertificateRejected):
        nr.verify_certificate({"schema": "nope", "cases": {}}, PRODUCER, CASE_IDS)


def test_certificate_never_reads_files_named_inside_it():
    cert = nr.build_certificate(PRODUCER, CASE_IDS)
    # extra keys OUTSIDE the digested results must not be followed or opened
    cert["claim_paths"] = ["/etc/passwd", "../../../etc/passwd"]
    assert nr.verify_certificate(cert, PRODUCER, CASE_IDS) is True


# ---------------------------------------------------------------------------
# tools/image_navigation.sh — the single-command entry point
# ---------------------------------------------------------------------------

def test_image_navigation_sh_is_valid_bash_with_uv_pipeline():
    import pathlib
    import subprocess
    script = (pathlib.Path(__file__).resolve().parent.parent / "tools"
              / "image_navigation.sh")
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "uv sync" in text and "--locked" in text
    assert "navigation_report" in text
    assert "CUDA_VISIBLE_DEVICES" in text  # force CPU
    assert "set -euo pipefail" in text

def test_make_output_dir_creates_fresh_safe_dir(tmp_path):
    d = nr.make_output_dir(str(tmp_path))
    assert d.startswith(str(tmp_path)) and d != str(tmp_path)
    # second call must not collide or reuse
    d2 = nr.make_output_dir(str(tmp_path))
    assert d2 != d


def test_make_output_dir_rejects_unsafe_paths(tmp_path):
    with pytest.raises(ValueError):
        nr.make_output_dir(str(tmp_path / "sub" / ".." / "escape"))


def test_run_report_writes_report_and_certificate(tmp_path, capsys):
    rc = nr.main(["--out", str(tmp_path)], producer=PRODUCER, case_ids=CASE_IDS)
    assert rc == 0
    out_dir = capsys.readouterr().out.splitlines()[0]
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "report.html"))
    assert os.path.isfile(os.path.join(out_dir, "certificate.json"))


def test_main_verify_passes_on_good_certificate(tmp_path):
    nr.main(["--out", str(tmp_path)], producer=PRODUCER, case_ids=CASE_IDS)
    cert_path = next(tmp_path.glob("*/certificate.json"))
    rc = nr.main(["--verify", str(cert_path)], producer=PRODUCER, case_ids=CASE_IDS)
    assert rc == 0


def test_main_verify_fails_on_tampered_certificate(tmp_path):
    nr.main(["--out", str(tmp_path)], producer=PRODUCER, case_ids=CASE_IDS)
    cert_path = next(tmp_path.glob("*/certificate.json"))
    cert = json.loads(cert_path.read_text())
    cert["cases"]["case-b"]["digest"] = "f" * 64
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(cert))
    assert nr.main(["--verify", str(bad)], producer=PRODUCER, case_ids=CASE_IDS) == 1


def test_public_audit_writer_emits_verdict(tmp_path):
    nr.write_audit(tmp_path, lambda: {"schema": "example", "verdict": "independent"})
    assert json.loads((tmp_path / "replacement-audit.json").read_text())["verdict"] == "independent"
