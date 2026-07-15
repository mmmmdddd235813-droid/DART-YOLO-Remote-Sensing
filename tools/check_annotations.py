"""Check YOLO annotation files for format errors and split overlap."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO labels and train/val split files.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-list")
    parser.add_argument("--val-list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bad = []
    for label in Path(args.labels).rglob("*.txt"):
        for i, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                bad.append(f"{label}:{i}: expected 5 fields")
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                bad.append(f"{label}:{i}: non-numeric field")
                continue
            if cls < 0 or any(v < 0 or v > 1 for v in vals):
                bad.append(f"{label}:{i}: value outside expected range")

    print(f"annotation_errors: {len(bad)}")
    for item in bad[:50]:
        print(item)

    if args.train_list and args.val_list:
        train = set(Path(args.train_list).read_text(encoding="utf-8").splitlines())
        val = set(Path(args.val_list).read_text(encoding="utf-8").splitlines())
        overlap = train & val
        print(f"train_count: {len(train)}")
        print(f"val_count: {len(val)}")
        print(f"train_val_overlap: {len(overlap)}")
        for item in sorted(overlap)[:20]:
            print(item)

    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
