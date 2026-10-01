#!/usr/bin/env python3
"""Validate that algorithm documentation files follow the project template.

Checks every .md file under doc/ (excluding README.md, TEMPLATE.md, and backups)
for the required section headings defined in the template, that each one is
listed in its domain index doc/<domain>/README.md, and that relative links and
images in doc/ resolve (including #anchors into other docs).

Exit codes:
    0 — all files pass, or no algorithm documentation files found
    1 — doc directory is missing, or one or more files fail a check
"""

import pathlib
import re
import sys

REQUIRED_SECTIONS = [
    "Overview & Motivation",
    "Mathematical Theory",
    "Complexity Analysis",
    "Step-by-Step Walkthrough",
    "Pitfalls & Edge Cases",
    "Variants & Generalizations",
    "Applications",
    "Connections to Other Algorithms",
    "References & Further Reading",
]

SKIP_NAMES = {"README.md", "TEMPLATE.md"}
SKIP_DIRS = {".backup"}

DOC_ROOT = pathlib.Path(__file__).resolve().parent.parent / "doc"

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^```.*?^```|<!--.*?-->", re.MULTILINE | re.DOTALL)


def extract_h2_headings(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, re.MULTILINE)


def validate_file(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    headings = extract_h2_headings(text)
    missing = [s for s in REQUIRED_SECTIONS if s not in headings]
    return missing


def slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: pathlib.Path) -> set[str]:
    text = FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    return {slug(h) for h in HEADING_LINE_RE.findall(text)}


def index_errors(path: pathlib.Path) -> list[str]:
    readme = path.parent / "README.md"
    if path.parent == DOC_ROOT or not readme.is_file():
        return [f"no domain index {readme.relative_to(DOC_ROOT)}"]
    listed = {
        (readme.parent / t.split("#", 1)[0]).resolve()
        for t in LINK_RE.findall(readme.read_text(encoding="utf-8"))
    }
    if path.resolve() not in listed:
        return [f"not listed in {readme.relative_to(DOC_ROOT)}"]
    return []


def link_errors(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    text = FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        linked = (path.parent / file_part).resolve() if file_part else path
        if not linked.exists():
            errors.append(f"broken link: {target}")
        elif anchor and linked.suffix == ".md" and anchor not in anchors(linked):
            errors.append(f"broken anchor: {target}")
    return errors


def main() -> int:
    if not DOC_ROOT.is_dir():
        print(f"ERROR: doc directory not found at {DOC_ROOT}", file=sys.stderr)
        return 1

    md_files = sorted(
        p
        for p in DOC_ROOT.rglob("*.md")
        if p.name not in SKIP_NAMES
        and not any(part in SKIP_DIRS for part in p.parts)
    )

    if not md_files:
        print("WARNING: no algorithm documentation files found.")
        return 0

    failures: dict[pathlib.Path, list[str]] = {}
    for path in md_files:
        problems = [f"missing: ## {s}" for s in validate_file(path)]
        problems += index_errors(path)
        if problems:
            failures.setdefault(path, []).extend(problems)

    all_md = sorted(p for p in DOC_ROOT.rglob("*.md") if not any(part in SKIP_DIRS for part in p.parts))
    for path in all_md:
        problems = link_errors(path)
        if problems:
            failures.setdefault(path, []).extend(problems)

    if failures:
        print(f"FAIL: {len(failures)} file(s) failed the documentation checks:\n")
        for path, problems in sorted(failures.items()):
            rel = path.relative_to(DOC_ROOT)
            print(f"  {rel}")
            for problem in problems:
                print(f"    - {problem}")
            print()
        return 1

    print(f"OK: {len(md_files)} file(s) validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
