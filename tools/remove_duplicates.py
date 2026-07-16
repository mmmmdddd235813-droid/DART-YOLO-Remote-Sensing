"""Find duplicate images by SHA-256 digest and optionally move duplicates."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report duplicate images and optionally move redundant copies.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--report", default="duplicates.txt")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move duplicate copies to --removed-dir. Without this flag the script only writes a report.",
    )
    parser.add_argument(
        "--removed-dir",
        default="duplicates_removed",
        help="Directory used when --apply is set. Files are moved, not permanently deleted.",
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_destination(root: Path, source: Path) -> Path:
    candidate = root / source.name
    if not candidate.exists():
        return candidate
    stem, suffix = source.stem, source.suffix
    index = 1
    while True:
        candidate = root / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    args = parse_args()
    image_root = Path(args.images)
    groups = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            groups[digest(path)].append(path)
    lines = []
    moved = []
    for files in groups.values():
        if len(files) > 1:
            files = sorted(files)
            keep = files[0]
            duplicates = files[1:]
            lines.append("KEEP " + str(keep) + " | MOVE " + " | ".join(str(p) for p in duplicates))
            if args.apply:
                removed_root = Path(args.removed_dir)
                removed_root.mkdir(parents=True, exist_ok=True)
                for duplicate in duplicates:
                    destination = unique_destination(removed_root, duplicate)
                    shutil.move(str(duplicate), str(destination))
                    moved.append(f"{duplicate} -> {destination}")
    Path(args.report).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"duplicate_groups: {len(lines)}")
    if args.apply:
        print(f"moved_files: {len(moved)}")


if __name__ == "__main__":
    main()
