#!/usr/bin/env python3
"""
Concatenate all .py files under a root directory into a single text file.

- Uses PEP 263-aware encoding detection (tokenize.open) to read each file.
- Writes UTF-8 output, preserving the original code text (no reformatting).
- Skips common “noise” folders by default (.git, __pycache__, venvs, etc.).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import tokenize

DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".venv", "venv", "env", "build", "dist", "node_modules"
}

HEADER_LINE = "=" * 80

def iter_py_files(root: Path, exclude_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    # Walk with pruning
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories in-place for efficiency
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(Path(dirpath) / fn)
    # Deterministic order
    files.sort(key=lambda p: p.as_posix().lower())
    return files

def write_header(out, rel_path: Path, size_bytes: int | None):
    out.write(f"{HEADER_LINE}\n")
    out.write(f"FILE: {rel_path.as_posix()}")
    if size_bytes is not None:
        out.write(f"  (size: {size_bytes} bytes)")
    out.write("\n")
    out.write(f"{HEADER_LINE}\n")

def concat_python(
    root: Path,
    out_path: Path,
    exclude_dirs: set[str],
    include_headers: bool = True,
) -> int:
    root = root.resolve()
    out_path = out_path.resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: root '{root}' is not a directory.", file=sys.stderr)
        return 2

    py_files = iter_py_files(root, exclude_dirs)
    if not py_files:
        print("No Python files found.", file=sys.stderr)
        return 1

    # Stream writing; do not load everything into memory.
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for i, f in enumerate(py_files, 1):
            rel = f.relative_to(root)
            try:
                # PEP 263-aware opener; preserves code text correctly
                with tokenize.open(f) as src:
                    content = src.read()
            except Exception as e:
                print(f"[WARN] Skipping {rel}: {e}", file=sys.stderr)
                continue

            if include_headers:
                try:
                    size_bytes = f.stat().st_size
                except Exception:
                    size_bytes = None
                write_header(out, rel, size_bytes)

            out.write(content)
            # Ensure a trailing newline between files (if the file didn't end with one)
            if not content.endswith("\n"):
                out.write("\n")
            # Add a blank line after each file for readability
            if include_headers:
                out.write("\n")

    print(f"Done. Wrote {len(py_files)} files into '{out_path}'.")
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate all .py files under ROOT into OUTPUT (UTF-8)."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan")
    parser.add_argument("output", type=Path, help="Output .txt file path")
    parser.add_argument(
        "--no-headers",
        action="store_true",
        help="Do not add file path headers between files"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional directory names to exclude (space-separated)"
    )
    args = parser.parse_args()

    exclude_dirs = set(DEFAULT_EXCLUDES) | set(args.exclude or [])
    rc = concat_python(args.root, args.output, exclude_dirs, include_headers=not args.no_headers)
    sys.exit(rc)

if __name__ == "__main__":
    main()
