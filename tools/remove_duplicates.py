"""Find duplicate images by SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report duplicate image files.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--report", default="duplicates.txt")
    return parser.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    groups = defaultdict(list)
    for path in Path(args.images).rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            groups[digest(path)].append(path)
    lines = []
    for files in groups.values():
        if len(files) > 1:
            lines.append(" | ".join(str(p) for p in files))
    Path(args.report).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"duplicate_groups: {len(lines)}")


if __name__ == "__main__":
    main()
