"""Map all remaining tank labels to class id 0."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map YOLO label class ids to tank=0.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.labels)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    for label in src.glob("*.txt"):
        lines = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                parts[0] = "0"
                lines.append(" ".join(parts[:5]))
        (dst / label.name).write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


if __name__ == "__main__":
    main()
