from pathlib import Path

from tools.check_docs_site import main, parse_args


def test_markdown_and_generated_site_links_pass(tmp_path: Path) -> None:
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "logo.svg").write_text("<svg />", encoding="utf-8")
    (tmp_path / "docs" / "EPISODE_SCHEMA.md").write_text("# Schema\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text(
        "![logo](assets/logo.svg)\n[Schema](EPISODE_SCHEMA.md)\n[GitHub](https://github.com/SeldingerMed/seldinger-lumen)\n",
        encoding="utf-8",
    )

    site = tmp_path / "_site"
    (site / "assets").mkdir(parents=True)
    (site / "assets" / "logo.svg").write_text("<svg />", encoding="utf-8")
    (site / "EPISODE_SCHEMA.html").write_text("<html></html>", encoding="utf-8")
    (site / "index.html").write_text(
        '<a href="EPISODE_SCHEMA.html">schema</a><img src="assets/logo.svg">',
        encoding="utf-8",
    )

    assert main(["--repo-root", str(tmp_path), "--site-dir", str(site)]) == 0


def test_missing_markdown_link_fails(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text("[missing](missing.md)\n", encoding="utf-8")

    assert main(["--repo-root", str(tmp_path)]) == 1


def test_missing_generated_site_link_fails(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# ok\n", encoding="utf-8")
    site = tmp_path / "_site"
    site.mkdir()
    (site / "index.html").write_text('<a href="missing.html">missing</a>', encoding="utf-8")

    assert main(["--repo-root", str(tmp_path), "--site-dir", str(site)]) == 1


def test_generated_site_links_allow_project_pages_base_path(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# ok\n", encoding="utf-8")
    site = tmp_path / "_site"
    (site / "assets").mkdir(parents=True)
    (site / "assets" / "logo.svg").write_text("<svg />", encoding="utf-8")
    (site / "index.html").write_text(
        '<img src="/seldinger-lumen/assets/logo.svg">', encoding="utf-8"
    )

    assert (
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--site-dir",
                str(site),
                "--base-path",
                "/seldinger-lumen",
            ]
        )
        == 0
    )


def test_custom_markdown_patterns_replace_defaults() -> None:
    assert parse_args([]).markdown == ["*.md", "docs/**/*.md"]
    assert parse_args(["--markdown", "custom/**/*.md"]).markdown == ["custom/**/*.md"]


def test_markdown_links_ignore_decode_errors(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "target.md").write_text("# target\n", encoding="utf-8")
    (docs / "index.md").write_bytes(b"\xff[Target](target.md)\n")

    assert main(["--repo-root", str(tmp_path)]) == 0
