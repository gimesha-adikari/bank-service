import argparse, re, sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_IGNORES = [r"\.git", r"node_modules", r"build", r"dist", r"out", r"\.gradle", r"\.idea", r"\.cache", r"\.next"]

def should_ignore(path: Path, ignore_res: List[re.Pattern]) -> bool:
    sp = str(path)
    return any(pat.search(sp) for pat in ignore_res)

def tree_lines(root: Path, ignores: List[str], dirs_only: bool, max_depth: int) -> Iterable[str]:
    ignore_res = [re.compile(p) for p in ignores]
    root = root.resolve()
    yield f"{root.name}/"
    def walk(d: Path, prefix: str, depth: int):
        if 0 <= max_depth < depth:
            return
        try:
            entries = sorted(
                [e for e in d.iterdir() if not should_ignore(e, ignore_res)],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            return
        for i, e in enumerate(entries):
            last = (i == len(entries) - 1)
            conn = "└── " if last else "├── "
            if e.is_dir():
                yield f"{prefix}{conn}{e.name}/"
                ext = "    " if last else "│   "
                yield from walk(e, prefix + ext, depth + 1)
            elif not dirs_only:
                yield f"{prefix}{conn}{e.name}"
    yield from walk(root, "", 1)

def main():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Export directory structure as text starting from this script's folder.")
    ap.add_argument("--output", default=str(script_dir / "tree.txt"),
                    help="Output path (default: ./tree.txt). Use '-' for stdout.")
    ap.add_argument("--ignore", nargs="*", default=DEFAULT_IGNORES,
                    help="Regex patterns to ignore (space-separated)")
    ap.add_argument("--dirs-only", action="store_true", help="List only directories")
    ap.add_argument("--max-depth", type=int, default=-1, help="Limit depth; -1 = unlimited")
    args = ap.parse_args()

    lines = list(tree_lines(script_dir, args.ignore, args.dirs_only, args.max_depth))
    data = "\n".join(lines)

    if args.output == "-" or args.output.lower() == "stdout":
        print(data)
        return

    out = Path(args.output).expanduser()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(data, encoding="utf-8")
        print(f"[OK] Root: {script_dir}")
        print(f"[OK] Wrote {len(lines)} lines to {out.resolve()}")
    except PermissionError:
        fallback = Path.home() / "tree.txt"
        fallback.write_text(data, encoding="utf-8")
        print(f"[WARN] Permission denied writing {out}.")
        print(f"[OK] Wrote to fallback: {fallback} ({len(lines)} lines)")

if __name__ == "__main__":
    main()
