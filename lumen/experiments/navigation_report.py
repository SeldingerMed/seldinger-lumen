"""Blind-first HTML navigation report with a replay certificate verifier.

Consumer layer for the image-guided navigation rollout. It never imports the
Newton engine or any heavy dependency at module import time — the rollout
producer (``lumen.experiments.image_rollout.run_case``) is imported lazily so
independent unit tests can inject a tiny in-memory producer instead.

Rollout contract consumed here (new final rollout contract)::

    from lumen.experiments.image_rollout import CASES, run_case
    CASES  # tuple of strings: "case-a", "case-b", "case-c"
    run_case(case_id) -> {
        case_id, goal,
        frames: [{"shape": [h, w], "data": <base64 uint8>}, ...],
        action: str,
        hidden: {"tip_xyz_mm": [[x, y, z], ...], "reason": str},
        scores: {"target_reached": bool, "max_penetration_mm": float,
                 "peak_wall_load_sim_units": float, "abstained": bool},
    }

Blind-first policy of the report:

* everything the policy could see (fluoro frames) and every separately
  measured score is rendered immediately;
* the hidden device trajectory (rendered visually as an SVG polyline) and the
  reviewer's disclosure reason sit behind a ``<details>`` disclosure the
  reader must click;
* case IDs are NOT shown in the blind portion — cases are labelled
  "Case 1..N" by index so the reader cannot match a case to a frozen ID
  before disclosing. (In this registry the IDs are already non-spoiling,
  but the report never relies on that.)

Replay certificate:

``build_certificate`` records the SHA-256 digest of every case result.
``verify_certificate`` re-runs ALL cases through the producer and compares
digests, rejecting tampering, missing cases, or a malformed document. The
verifier reads ONLY the certificate document handed to it — it never opens
any path mentioned inside that (untrusted) document.

CLI (single command from a clean checkout via ``tools/image_navigation.sh``)::

    uv run --no-sync python -m lumen.experiments.navigation_report [--out DIR]
    uv run --no-sync python -m lumen.experiments.navigation_report --verify CERT.json
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import sys
import zlib
from datetime import datetime, timezone

CERT_SCHEMA = "lumen-navigation-certificate-v1"
DEFAULT_PRODUCER_MODULE = "lumen.experiments.image_rollout"
__all__ = [
    "CertificateRejected", "build_certificate", "verify_certificate",
    "build_html", "encode_gray_png", "make_output_dir", "main",
]


class CertificateRejected(ValueError):
    """The replay certificate is malformed, tampered with, or incomplete."""


# ---------------------------------------------------------------------------
# Minimal grayscale PNG encoder (stdlib only: zlib)
# ---------------------------------------------------------------------------

def encode_gray_png(pixels: bytes, width: int, height: int) -> bytes:
    """Encode an 8-bit grayscale image as a PNG (no external deps)."""
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    if len(pixels) != width * height:
        raise ValueError("pixel buffer does not match width*height")

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + ctype + data
                + zlib.crc32(ctype + data).to_bytes(4, "big"))

    ihdr = (width.to_bytes(4, "big") + height.to_bytes(4, "big")
            + bytes([8, 0, 0, 0, 0]))            # 8-bit, grayscale
    raw = b"".join(b"\x00" + pixels[y * width:(y + 1) * width]
                   for y in range(height))    # filter 0 per scanline
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def _frame_data_uri(frame: dict) -> str:
    shape = frame.get("shape") or [0, 0]
    h, w = int(shape[0]), int(shape[1])
    data = base64.b64decode(frame["data"], validate=True)
    if len(data) != h * w:
        raise ValueError(f"frame payload {len(data)} != shape {h}x{w}")
    return ("data:image/png;base64,"
            + base64.b64encode(encode_gray_png(data, w, h)).decode())


# ---------------------------------------------------------------------------
# Blind-first HTML report
# ---------------------------------------------------------------------------

_SVG_W, _SVG_H = 240.0, 240.0


def _trajectory_svg(points) -> str:
    """Render the hidden tip trajectory (mm) as an SVG polyline."""
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in points]
    if not pts:
        return "<p>(no trajectory points)</p>"
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    span = [max(hi[i] - lo[i], 1e-9) for i in range(3)]

    def px(p):  # x-y view of the mm trajectory, padded
        return (_SVG_W * 0.05 + (_SVG_W * 0.9) * (p[0] - lo[0]) / span[0],
                _SVG_H * 0.95 - (_SVG_H * 0.9) * (p[1] - lo[1]) / span[1])

    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(px, pts))
    # depth encoded as dot radius
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.5 + 3.0 * (p[2] - lo[2]) / span[2]:.1f}"/>'
                   for p, (x, y) in zip(pts, map(px, pts)))
    return (f'<svg width="{_SVG_W:.0f}" height="{_SVG_H:.0f}" role="img"'
            f' aria-label="hidden tip trajectory">'
            f'<rect width="100%" height="100%" fill="#fafafa"/>'
            f'<polyline points="{coords}" fill="none" stroke="#b30000"'
            f' stroke-width="2"/>{dots}</svg>'
            f'<p class="hidden-note">tip trajectory (mm); dot radius = depth (z)</p>')


def build_html(results, case_labels=None) -> str:
    """Render the blind-first HTML report for a list of case results."""
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Navigation Report (blind-first)</title><style>",
        "body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px}",
        "details{border:1px solid #999;padding:0.5rem;margin:0.5rem 0}",
        "summary{cursor:pointer;font-weight:bold}",
        ".hidden-note{color:#666;font-size:0.85rem}",
        "img.frame{image-rendering:pixelated;width:200px;border:1px solid #ccc}",
        "</style></head><body>",
        "<h1>Choose the contrasted branch</h1><p>Inspect the detector frame before revealing the pulse, trajectory and scores. Scripted CPU simulation. Wall load is in simulator units, not real force or injury.</p>",
    ]
    labels = case_labels or [f"Case {i + 1}" for i in range(len(results))]
    for i, (label, res) in enumerate(zip(labels, results), 1):
        esc = html.escape
        parts.append(f"<h2>{esc(label)}</h2>")
        parts.append(f"<p><b>Goal:</b> {esc(str(res.get('goal', '')))}</p>")


        # --- blind: frames ------------------------------------------------
        parts.append("<h3>Fluoro frames (chronological)</h3><p>")
        for j, frame in enumerate(res.get("frames", []), 1):
            uri = _frame_data_uri(frame)
            shape = frame.get("shape", ["?", "?"])
            parts.append(f"<figure style='display:inline-block;margin:4px'>"
                         f"<img class='frame' src='{uri}' alt='frame {j}'>"
                         f"<figcaption>frame {j} ({shape[0]}x{shape[1]})</figcaption></figure>")
        parts.append("</p>")

        parts.append("<details><summary>Reveal hidden evidence "
                     "(trajectory, reason, case identity)</summary>")
        parts.append(f"<p><b>Executed action:</b> {esc(str(res.get('action', '')))}</p>")
        # --- blind: all scores, shown separately --------------------------
        scores = res.get("scores", {})
        parts.append("<h3>Scores (all measured separately)</h3><table><tr><th>score</th><th>value</th></tr>")
        for key in ("target_reached", "max_penetration_mm",
                    "peak_wall_load_sim_units", "abstained"):
            parts.append(f"<tr><td>{esc(key)}</td><td>{esc(str(scores.get(key)))}</td></tr>")
        parts.append("</table>")

        # --- sealed: hidden trajectory + reason ---------------------------
        hidden = res.get("hidden", {})
        parts.append(f"<p><b>Case ID:</b> {esc(str(res.get('case_id', '?')))}</p>")
        parts.append("<h4>Reason</h4><p>"
                     + esc(str(hidden.get("reason", ""))) + "</p>")
        parts.append("<h4>Hidden tip trajectory</h4>")
        parts.append(_trajectory_svg(hidden.get("tip_xyz_mm", [])))
        parts.append("</details>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Replay certificate
# ---------------------------------------------------------------------------

def _digest(result) -> str:
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_certificate(producer, case_ids) -> dict:
    """Run every case and record its result + digest in a certificate dict."""
    cases = {}
    for cid in case_ids:
        result = producer(cid)
        cases[str(cid)] = {"digest": _digest(result), "result": result}
    return {
        "schema": CERT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": f"{DEFAULT_PRODUCER_MODULE}.run_case",
        "cases": cases,
    }


def verify_certificate(cert, producer, case_ids) -> bool:
    """Re-run ALL cases and compare digests; raise CertificateRejected on any
    tampering, missing case, or malformed document.

    Only the in-memory ``cert`` document is inspected; no path found inside it
    is ever opened or followed.
    """
    if not isinstance(cert, dict):
        raise CertificateRejected("certificate is not a JSON object")
    if cert.get("schema") != CERT_SCHEMA:
        raise CertificateRejected(f"unknown certificate schema {cert.get('schema')!r}")
    stored = cert.get("cases")
    if not isinstance(stored, dict):
        raise CertificateRejected("certificate has no 'cases' object")

    expected = [str(c) for c in case_ids]
    missing = [c for c in expected if c not in stored]
    if missing:
        raise CertificateRejected(f"missing case(s) in certificate: {missing}")

    if set(stored) != set(expected):
        raise CertificateRejected("unexpected case in certificate")
    for cid in expected:
        entry = stored[cid]
        if not isinstance(entry, dict) or "digest" not in entry or "result" not in entry:
            raise CertificateRejected(f"case {cid!r} is malformed")
        if not isinstance(entry["digest"], str) or len(entry["digest"]) != 64:
            raise CertificateRejected(f"case {cid!r} has a malformed digest")
        if _digest(entry["result"]) != entry["digest"]:
            raise CertificateRejected(f"stored result for case {cid!r} does not "
                                      "match its digest (tampered)")
        try:
            rerun = producer(cid)
        except Exception as exc:  # noqa: BLE001 — producer failures reject
            raise CertificateRejected(f"replay of case {cid!r} failed: {exc}") from exc
        if _digest(rerun) != entry["digest"]:
            raise CertificateRejected(f"digest mismatch for case {cid!r} "
                                      "(tampered or non-replayable)")
    return True


# ---------------------------------------------------------------------------
# Safe output directory
# ---------------------------------------------------------------------------

def make_output_dir(parent: str | None = None) -> str:
    """Create a fresh, non-clobbering output directory and return its path.

    The directory never pre-exists (no overwrite of prior reports) and the
    requested parent is validated against traversal tricks.
    """
    parent = os.path.abspath(parent or os.getcwd())
    if not os.path.isdir(parent):
        raise ValueError(f"output parent directory does not exist: {parent}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in range(10000):
        candidate = os.path.join(parent, f"lumen_navigation_report-{stamp}"
                                         + (f"-{suffix}" if suffix else ""))
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            break
    resolved = os.path.realpath(candidate)
    if os.path.dirname(resolved) != parent:
        raise ValueError(f"unsafe output path: {candidate}")
    return resolved


def _load_default_producer():
    """Lazy import of the real rollout layer (never at module import time)."""
    module = __import__(DEFAULT_PRODUCER_MODULE, fromlist=["run_case"])
    producer = module.run_case
    case_ids = module.CASES
    ids = [c if isinstance(c, str) else c.get("case_id") for c in case_ids]
    return producer, [str(c) for c in ids]


def _load_certificate(path: str) -> dict:
    """Load a certificate from a user-supplied path (the only file we read)."""
    full = os.path.abspath(path)
    if not os.path.isfile(full):
        raise CertificateRejected(f"certificate file not found: {path}")
    if os.path.realpath(full) != full:
        raise CertificateRejected("refusing symlinked certificate path")
    with open(full, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_audit(out_dir, produce):
    from pathlib import Path

    (Path(out_dir) / "replacement-audit.json").write_text(
        json.dumps(produce(), allow_nan=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None, producer=None, case_ids=None) -> int:
    """CLI entry point.

    Default: run all cases, write report.html + certificate.json into a fresh
    safe directory, print the path, then re-verify the certificate.
    ``--verify CERT.json``: re-run all cases against an existing certificate.

    ``producer``/``case_ids`` parameters exist for unit tests with an
    in-memory producer; the CLI path uses the real rollout layer lazily.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    verify_path = None
    out_parent = None
    it = iter(argv)
    for arg in it:
        if arg == "--verify":
            verify_path = next(it, None)
        elif arg == "--out":
            out_parent = next(it, None)
        else:
            print(f"usage: python -m lumen.experiments.navigation_report "
                  f"[--out DIR] [--verify CERT.json]; unknown arg {arg!r}",
                  file=sys.stderr)
            return 2

    default_run = producer is None and case_ids is None
    if producer is None or case_ids is None:
        producer, case_ids = _load_default_producer()

    try:
        if verify_path is not None:
            cert = _load_certificate(verify_path)
            verify_certificate(cert, producer, case_ids)
            print(f"VERIFIED: {verify_path} — all cases replay with matching digests")
            return 0
        out_dir = make_output_dir(out_parent)
        cert = build_certificate(producer, case_ids)
        report = build_html([cert["cases"][c]["result"] for c in case_ids])
        report_path = os.path.join(out_dir, "report.html")
        cert_path = os.path.join(out_dir, "certificate.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        with open(cert_path, "w", encoding="utf-8") as fh:
            json.dump(cert, fh, indent=2)
        verify_certificate(cert, producer, case_ids)
        if default_run:
            from .lineage_audit import replacement_audit
            write_audit(out_dir, replacement_audit)
        print(out_dir)
        print(f"report: {report_path}")
        print(f"certificate: {cert_path}")
        return 0
    except CertificateRejected as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
