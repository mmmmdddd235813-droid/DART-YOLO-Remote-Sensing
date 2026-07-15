"""Keep tank-related classes from YOLO labels."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter YOLO labels to selected class ids.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--keep", nargs="+", required=True, help="Class ids to keep.")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep = set(args.keep)
    src = Path(args.labels)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    for label in src.glob("*.txt"):
        lines = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] in keep:
                lines.append(line)
        (dst / label.name).write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


if __name__ == "__main__":
    main()
