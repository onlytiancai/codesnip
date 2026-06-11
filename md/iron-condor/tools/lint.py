#!/usr/bin/env python3
"""
Lint the LLM Wiki for health issues.

Usage:
    python tools/lint.py
    python tools/lint.py --save          # save lint report to wiki/lint-report.md

Checks:
  - Orphan pages (no inbound wikilinks from other pages)
  - Broken wikilinks (pointing to pages that don't exist)
  - Missing entity pages (entities mentioned in 3+ pages but no page)
  - Contradictions between pages
  - Data gaps and suggested new sources
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date

import os

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
LOG_FILE = WIKI_DIR / "log.md"
SCHEMA_FILE = REPO_ROOT / "CLAUDE.md"


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def call_llm(prompt: str, model_env: str, default_model: str, max_tokens: int = 4096) -> str:
    try:
        from litellm import completion
    except ImportError:
        print("Error: litellm not installed. Run: pip install litellm")
        sys.exit(1)
        
    model = os.getenv(model_env, default_model)
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def all_wiki_pages() -> list[Path]:
    return [p for p in WIKI_DIR.rglob("*.md")
            if p.name not in ("index.md", "log.md", "lint-report.md")]


def extract_wikilinks(content: str) -> list[str]:
    return re.findall(r'\[\[([^\]]+)\]\]', content)


def page_name_to_path(name: str) -> list[Path]:
    """Try to resolve a [[WikiLink]] to a file path."""
    candidates = []
    for p in all_wiki_pages():
        if p.stem.lower() == name.lower() or p.stem == name:
            candidates.append(p)
    return candidates


def find_orphans(pages: list[Path]) -> list[Path]:
    inbound = defaultdict(int)
    for p in pages:
        content = read_file(p)
        for link in extract_wikilinks(content):
            resolved = page_name_to_path(link)
            for r in resolved:
                inbound[r] += 1
    return [p for p in pages if inbound[p] == 0 and p != WIKI_DIR / "overview.md"]


def find_broken_links(pages: list[Path]) -> list[tuple[Path, str]]:
    broken = []
    for p in pages:
        content = read_file(p)
        for link in extract_wikilinks(content):
            if not page_name_to_path(link):
                broken.append((p, link))
    return broken


def find_missing_entities(pages: list[Path]) -> list[str]:
    """Find entity-like names mentioned in 3+ pages but lacking their own page."""
    mention_counts: dict[str, int] = defaultdict(int)
    existing_pages = {p.stem.lower() for p in pages}
    for p in pages:
        content = read_file(p)
        links = extract_wikilinks(content)
        for link in links:
            if link.lower() not in existing_pages:
                mention_counts[link] += 1
    return [name for name, count in mention_counts.items() if count >= 3]


def run_lint():
    pages = all_wiki_pages()
    today = date.today().isoformat()

    if not pages:
        print("Wiki is empty. Nothing to lint.")
        return ""

    print(f"Linting {len(pages)} wiki pages...")

    # Deterministic checks
    orphans = find_orphans(pages)
    broken = find_broken_links(pages)
    missing_entities = find_missing_entities(pages)

    print(f"  orphans: {len(orphans)}")
    print(f"  broken links: {len(broken)}")
    print(f"  missing entity pages: {len(missing_entities)}")

    # Build context for semantic checks (contradictions, gaps)
    # Use a sample of pages to stay within context limits
    sample = pages[:20]
    pages_context = ""
    for p in sample:
        rel = p.relative_to(REPO_ROOT)
        pages_context += f"\n\n### {rel}\n{read_file(p)[:1500]}"  # truncate long pages

    print("  running semantic lint via API...")
    prompt = f"""You are linting an LLM Wiki. Review the pages below and identify:
1. Contradictions between pages (claims that conflict)
2. Stale content (summaries that newer sources have superseded)
3. Data gaps (important questions the wiki can't answer — suggest specific sources to find)
4. Concepts mentioned but lacking depth

Wiki pages (sample of {len(sample)} pages):
{pages_context}

Return a markdown lint report with these sections:
## Contradictions
## Stale Content
## Data Gaps & Suggested Sources
## Concepts Needing More Depth

Be specific — name the exact pages and claims involved.
"""
    semantic_report = call_llm(prompt, "LLM_MODEL", "claude-3-5-sonnet-latest", max_tokens=3000)

    # Compose full report
    report_lines = [
        f"# Wiki Lint Report — {today}",
        "",
        f"Scanned {len(pages)} pages.",
        "",
        "## Structural Issues",
        "",
    ]

    if orphans:
        report_lines.append("### Orphan Pages (no inbound links)")
        for p in orphans:
            report_lines.append(f"- `{p.relative_to(REPO_ROOT)}`")
        report_lines.append("")

    if broken:
        report_lines.append("### Broken Wikilinks")
        for page, link in broken:
            report_lines.append(f"- `{page.relative_to(REPO_ROOT)}` links to `[[{link}]]` — not found")
        report_lines.append("")

    if missing_entities:
        report_lines.append("### Missing Entity Pages (mentioned 3+ times but no page)")
        report_lines.append("> [!warning] Action Required\n> Run `python3 generate_missing_entities.py` to automatically materialize these missing hubs.")
        for name in missing_entities:
            report_lines.append(f"- `[[{name}]]`")
        report_lines.append("")

    if not orphans and not broken and not missing_entities:
        report_lines.append("No structural issues found.")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")
    report_lines.append(semantic_report)

    report = "\n".join(report_lines)
    print("\n" + report)
    return report


def append_log(entry: str):
    existing = read_file(LOG_FILE)
    LOG_FILE.write_text(entry.strip() + "\n\n" + existing, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lint the LLM Wiki")
    parser.add_argument("--save", action="store_true", help="Save lint report to wiki/lint-report.md")
    args = parser.parse_args()

    report = run_lint()

    if args.save and report:
        report_path = WIKI_DIR / "lint-report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\nSaved: {report_path.relative_to(REPO_ROOT)}")

    today = date.today().isoformat()
    append_log(f"## [{today}] lint | Wiki health check\n\nRan lint. See lint-report.md for details.")
