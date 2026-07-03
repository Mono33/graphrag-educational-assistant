#!/usr/bin/env python3
"""
Phase 3C — deterministic import rewrite script.

Walks the repo and rewrites all imports of the 7 root modules and the 3
root packages (config, graph_retriever, context_builder, text2cypher,
multilingual_text2cypher, query_metrics, llm_chain, agent, api, domains)
into their new homes under the `aix.*` namespace (src/aix/...).

The mapping is:

    config                     -> aix.core.config
    graph_retriever            -> aix.retrieval.graph_retriever
    context_builder            -> aix.retrieval.context_builder
    text2cypher                -> aix.retrieval.text2cypher
    multilingual_text2cypher   -> aix.retrieval.multilingual_text2cypher
    query_metrics              -> aix.retrieval.query_metrics
    llm_chain                  -> aix.generation.llm_chain
    agent                      -> aix.agent
    api                        -> aix.api
    domains                    -> aix.domains

Every occurrence of `from <old> ...` and `import <old> ...` (top-level
or indented inside functions/try blocks) is rewritten. Already-rewritten
imports (e.g. `from aix.core.config ...`) are NOT touched (regex uses a
strict `from\\s+<old>` boundary that does not match `from aix.<old>`).

Usage:
    # Preview every change without writing anything:
    python scripts/_phase3c_rewrite_imports.py --dry-run

    # Apply changes in-place (after `git mv` of the modules):
    python scripts/_phase3c_rewrite_imports.py

Designed to be IDEMPOTENT: running it twice in a row makes the same
files match zero patterns the second time. Safe to re-run.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

# -----------------------------------------------------------------------------
# Mapping table — order does not matter (each pattern is an exact module name)
# -----------------------------------------------------------------------------
ROOT_MAP: dict[str, str] = {
    "config":                   "aix.core.config",
    "graph_retriever":          "aix.retrieval.graph_retriever",
    "context_builder":          "aix.retrieval.context_builder",
    "text2cypher":              "aix.retrieval.text2cypher",
    "multilingual_text2cypher": "aix.retrieval.multilingual_text2cypher",
    "query_metrics":            "aix.retrieval.query_metrics",
    "llm_chain":                "aix.generation.llm_chain",
    "agent":                    "aix.agent",
    "api":                      "aix.api",
    "domains":                  "aix.domains",
}

# -----------------------------------------------------------------------------
# Directories to scan (relative to repo root)
# -----------------------------------------------------------------------------
SCAN_DIRS: tuple[str, ...] = (
    "src",       # the moved package tree (after R3)
    "apps",      # streamlit + cli entry points
    "scripts",   # ops / ingest / audit / data_prep / ml utilities
    "tests",     # pytest tests (incl. integration)
)

# Skip cache / venv / build artefacts
SKIP_DIR_NAMES: frozenset[str] = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", ".env",
    "build", "dist", "node_modules", ".git",
    "artifacts", "data", "logs", "neo4j_migrations", "archive",
    ".devcontainer", ".github",
    "_generate_report_template.py",
})


def _build_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Compile two regex patterns per old module:

       - `from <old>(\\s|.|$)`        — anywhere on a line (any indentation)
       - `^<indent>import <old>(\\s|.|$)`  — ONLY at start-of-line (after
         optional indentation); never inside `from X import <name>` because
         in that case `<name>` is an imported value, not a module name.
    """
    patterns: list[tuple[re.Pattern[str], str]] = []
    for old, new in ROOT_MAP.items():
        # `from <old>` followed by whitespace, dot, or end-of-line.
        # Negative lookbehind `(?<!\.)` ensures we don't match `from foo.<old>`.
        from_re = re.compile(
            rf"(?<!\.)\bfrom(\s+){re.escape(old)}(?=[\s.\r\n]|$)",
            re.MULTILINE,
        )
        patterns.append((from_re, rf"from\1{new}"))

        # `import <old>` only at the START of a (possibly indented) logical
        # line. This anchors with `^[ \t]*` and is therefore never matched
        # in the middle of a `from X import Y` statement.
        import_re = re.compile(
            rf"(?m)^([ \t]*)import(\s+){re.escape(old)}(?=[\s.\r\n,]|$)",
        )
        patterns.append((import_re, rf"\1import\2{new}"))
    return patterns


def _iter_py_files(root: Path) -> Iterable[Path]:
    """Yield every .py file under SCAN_DIRS, skipping cache/venv/build."""
    for top in SCAN_DIRS:
        base = root / top
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if any(part in SKIP_DIR_NAMES for part in p.parts):
                continue
            # Skip the rewrite script itself
            if p.name == "_phase3c_rewrite_imports.py":
                continue
            yield p


def _rewrite_text(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> tuple[str, int]:
    """Return (new_text, total_substitutions)."""
    total = 0
    for pat, repl in patterns:
        text, n = pat.subn(repl, text)
        total += n
    return text, total


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3C — rewrite imports for the src/aix/ layout."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing any files.",
    )
    ap.add_argument(
        "--root",
        default=".",
        help="Repo root (defaults to current working directory).",
    )
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-file diff hunks (requires --dry-run for safety).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").exists():
        print(f"ERROR: no pyproject.toml found at {root}", file=sys.stderr)
        return 2

    patterns = _build_patterns()

    files_changed: list[tuple[Path, int]] = []
    files_scanned = 0

    for path in _iter_py_files(root):
        files_scanned += 1
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"  ! skip (non-utf8): {path.relative_to(root)}")
            continue

        rewritten, n = _rewrite_text(original, patterns)
        if n == 0:
            continue

        files_changed.append((path, n))
        rel = path.relative_to(root)
        print(f"  {'(dry) ' if args.dry_run else ''}rewrote {n:>3d} import(s):  {rel}")

        if args.verbose and args.dry_run:
            # Print a tiny diff summary (lines that changed)
            orig_lines = original.splitlines()
            new_lines = rewritten.splitlines()
            for i, (o, n_) in enumerate(zip(orig_lines, new_lines, strict=False), start=1):
                if o != n_:
                    print(f"    L{i:<5d}-: {o}")
                    print(f"    L{i:<5d}+: {n_}")

        if not args.dry_run:
            path.write_text(rewritten, encoding="utf-8", newline="\n")

    print()
    print("=" * 60)
    print(f"Scanned: {files_scanned:>4d} python file(s)")
    print(f"Changed: {len(files_changed):>4d} file(s)")
    print(f"Total imports rewritten: {sum(n for _, n in files_changed)}")
    if args.dry_run:
        print("(DRY RUN — no files were modified.)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
