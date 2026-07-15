"""Convert box annotations to YOLO txt files.

Input CSV columns:
image,class,xmin,ymin,xmax,ymax,width,height
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV box annotations to YOLO format.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--class-map", required=True, help="CSV with columns name,id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_map = {}
    with open(args.class_map, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_map[row["name"]] = int(row["id"])

    labels = defaultdict(list)
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = class_map[row["class"]]
            xmin, ymin, xmax, ymax = map(float, (row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
            width, height = float(row["width"]), float(row["height"])
            xc = ((xmin + xmax) / 2.0) / width
            yc = ((ymin + ymax) / 2.0) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            labels[Path(row["image"]).stem].append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    for stem, lines in labels.items():
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
