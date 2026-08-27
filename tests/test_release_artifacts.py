"""Release copies of the preprint must stay synchronized with their source."""

from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "launch" / "preprint"
PUBLIC_DIR = ROOT / "docs" / "assets" / "launch"


def test_public_preprint_copies_match_source_and_each_other():
    source_tex = (SOURCE_DIR / "lumen_preprint.tex").read_bytes()
    source_pdf = (SOURCE_DIR / "lumen_preprint.pdf").read_bytes()
    source_zip = (SOURCE_DIR / "lumen_preprint_latex.zip").read_bytes()

    assert (PUBLIC_DIR / "lumen-preprint.pdf").read_bytes() == source_pdf
    assert (PUBLIC_DIR / "lumen-preprint-latex.zip").read_bytes() == source_zip

    with ZipFile(SOURCE_DIR / "lumen_preprint_latex.zip") as archive:
        assert archive.read("lumen_preprint.tex") == source_tex

    text = source_tex.decode("utf-8")
    assert "centerline penetration" in text
    assert "6.7\\% safe success" not in text
    assert "force threshold used in the comparison" not in text
