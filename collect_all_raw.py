#!/usr/bin/env python3
"""
Collect EVERY file under ROOT into a single output file (raw bytes, no exceptions).

- Preserves bytes exactly (reads/writes in binary).
- Adds clear ASCII headers with file path and byte size before each file's bytes.
- No directory/extension filters. Hidden files included.
- Skips only the output file itself to avoid self-inclusion.
- By default does NOT follow symlinked dirs (use --followlinks to enable).
- Deterministic order (case-insensitive, path-based sort).

NOTE: The output is a binary concatenation with headers. It is not guaranteed to
be valid UTF-8 text because it contains raw bytes from arbitrary files.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

HEADER_LINE = b"=" * 80 + b"\n"

def iter_all_files(root: Path, followlinks: bool) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=followlinks):
        for fn in filenames:
            files.append(Path(dirpath) / fn)
    files.sort(key=lambda p: p.as_posix().lower())
    return files

def write_header(outb, rel_path: Path, size_bytes: int | None):
    outb.write(HEADER_LINE)
    # encode path with surrogateescape so any odd bytes round-trip in UTF-8
    rel_enc = rel_path.as_posix().encode("utf-8", "surrogateescape")
    outb.write(b"FILE: " + rel_enc)
    if size_bytes is not None:
        outb.write(b"  (size: " + str(size_bytes).encode("ascii") + b" bytes)")
    outb.write(b"\n")
    outb.write(HEADER_LINE)

def collect_all(root: Path, out_path: Path, followlinks: bool, include_headers: bool) -> int:
    root = root.resolve()
    out_path = out_path.resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: root '{root}' is not a directory.", file=sys.stderr)
        return 2

    files = iter_all_files(root, followlinks)
    if not files:
        print("No files found.", file=sys.stderr)
        return 1

    # Ensure output dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as outb:
        for f in files:
            # don't include the output file itself
            try:
                if f.resolve() == out_path:
                    continue
            except Exception:
                pass

            rel = f.relative_to(root)
            try:
                size_bytes = f.stat().st_size
            except Exception:
                size_bytes = None

            try:
                data = f.read_bytes()
            except Exception as e:
                # still write a header to record the error (no exceptions policy = we log, not skip silently)
                if include_headers:
                    write_header(outb, rel, size_bytes)
                    msg = f"[READ_ERROR] {e}".encode("utf-8", "surrogateescape")
                    outb.write(msg + b"\n\n")
                # and continue
                print(f"[WARN] Skipping (read error) {rel}: {e}", file=sys.stderr)
                continue

            if include_headers:
                write_header(outb, rel, size_bytes)

            outb.write(data)

            # Add a newline separator AFTER the bytes (does not alter file bytes above)
            # This helps readers visually separate content blocks.
            outb.write(b"\n\n")

    print(f"Done. Packed {len(files)} files into '{out_path}'.")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Concatenate ALL files under ROOT into OUTPUT (raw bytes).")
    ap.add_argument("root", type=Path, help="Root directory to scan")
    ap.add_argument("output", type=Path, help="Output file path (will be binary)")
    ap.add_argument("--no-headers", action="store_true", help="Do not write headers (pure raw concat)")
    ap.add_argument("--followlinks", action="store_true", help="Follow symlinked directories")
    args = ap.parse_args()

    rc = collect_all(args.root, args.output, followlinks=args.followlinks, include_headers=not args.no_headers)
    sys.exit(rc)

if __name__ == "__main__":
    main()
