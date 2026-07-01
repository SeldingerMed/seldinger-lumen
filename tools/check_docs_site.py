#!/usr/bin/env python3
"""Validate documentation source links and the generated GitHub Pages site.

This is intentionally dependency-free so CI can run it after the Pages/Jekyll
build without adding another package manager. It checks local links only: remote
URLs are allowed but not fetched, keeping PR CI deterministic.
"""

from __future__ import annotations

import argparse
import html.parser
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_ATTR_RE = re.compile(r"(?:href|src)=['\"]([^'\"]+)['\"]", re.IGNORECASE)
LOCAL_SCHEMES = {"", None}
IGNORED_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    target: str
    reason: str

    def format(self, root: Path) -> str:
        try:
            rel_source = self.source.relative_to(root)
        except ValueError:
            rel_source = self.source
        return f"{rel_source}: {self.target} ({self.reason})"


class LocalLinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name.lower() in {"href", "src"}:
                self.links.append(value)


def normalize_target(raw_url: str) -> str | None:
    url = raw_url.strip()
    if not url or url.startswith("#"):
        return None
    parsed = urlparse(url)
    if parsed.scheme in IGNORED_SCHEMES or parsed.netloc:
        return None
    if parsed.scheme not in LOCAL_SCHEMES:
        return None
    return unquote(parsed.path)


def strip_base_path(target: str, base_path: str) -> str:
    if not target.startswith("/") or not base_path:
        return target
    normalized_base = "/" + base_path.strip("/")
    if target == normalized_base:
        return "/"
    if target.startswith(normalized_base + "/"):
        return target[len(normalized_base) :]
    return target


def existing_path_for_site_link(
    source: Path, site_root: Path, target: str, base_path: str = ""
) -> Path | None:
    target = strip_base_path(target, base_path)
    candidate = (site_root / target.lstrip("/")) if target.startswith("/") else (source.parent / target)
    candidates = [candidate]
    if not candidate.suffix:
        candidates.append(candidate / "index.html")
        candidates.append(candidate.with_suffix(".html"))
    elif candidate.suffix == ".md":
        candidates.append(candidate.with_suffix(".html"))
    for path in candidates:
        if path.exists():
            return path
    return None


def existing_path_for_source_link(source: Path, repo_root: Path, target: str) -> Path | None:
    candidate = (repo_root / target.lstrip("/")) if target.startswith("/") else (source.parent / target)
    candidates = [candidate]
    if not candidate.suffix:
        candidates.append(candidate / "README.md")
        candidates.append(candidate.with_suffix(".md"))
    for path in candidates:
        if path.exists():
            return path
    return None


def extract_markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    links = [match.group(1) for match in MARKDOWN_LINK_RE.finditer(text)]
    links.extend(match.group(1) for match in HTML_ATTR_RE.finditer(text))
    return links


def extract_html_links(path: Path) -> list[str]:
    parser = LocalLinkParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return parser.links


def validate_markdown_sources(repo_root: Path, include_patterns: list[str]) -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    seen: set[Path] = set()
    for pattern in include_patterns:
        for path in repo_root.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            for raw_link in extract_markdown_links(path):
                target = normalize_target(raw_link)
                if target is None:
                    continue
                if existing_path_for_source_link(path, repo_root, target) is None:
                    broken.append(BrokenLink(path, raw_link, "missing source target"))
    return broken


def validate_site(site_root: Path, base_path: str = "") -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    for path in sorted(site_root.rglob("*.html")):
        for raw_link in extract_html_links(path):
            target = normalize_target(raw_link)
            if target is None:
                continue
            if existing_path_for_site_link(path, site_root, target, base_path=base_path) is None:
                broken.append(BrokenLink(path, raw_link, "missing generated site target"))
    return broken


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path, help="Repository root to scan")
    parser.add_argument("--site-dir", type=Path, help="Generated Pages/Jekyll output directory")
    parser.add_argument(
        "--base-path",
        default="",
        help="Published project Pages base path to strip from absolute local links (for example /seldinger-lumen)",
    )
    parser.add_argument(
        "--markdown",
        action="append",
        help="Markdown glob relative to repo root; repeatable",
    )
    args = parser.parse_args(argv)
    if args.markdown is None:
        args.markdown = ["*.md", "docs/**/*.md"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()
    broken = validate_markdown_sources(repo_root, args.markdown)

    if args.site_dir:
        site_root = args.site_dir.resolve()
        if not site_root.is_dir():
            broken.append(BrokenLink(site_root, str(site_root), "site directory does not exist"))
        else:
            broken.extend(validate_site(site_root, base_path=args.base_path))

    if broken:
        print("Documentation link validation failed:", file=sys.stderr)
        for item in broken:
            print(f"- {item.format(repo_root)}", file=sys.stderr)
        return 1

    checked_site = f" and generated site {args.site_dir}" if args.site_dir else ""
    print(f"Documentation links OK for {repo_root}{checked_site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
