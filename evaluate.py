"""Evaluate a trained DART-YOLO checkpoint.

The script reports Precision, Recall, mAP@0.5, and mAP@0.5:0.95 using the
dataset split defined by the supplied data YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from register_modules import register_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DART-YOLO.")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--data", default="data/uav_tank.yaml", help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--project", default="runs/evaluate")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_modules()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
    )
    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "save_dir": str(metrics.save_dir),
    }
    print(json.dumps(result, indent=2))
    Path(metrics.save_dir, "metrics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
