"""Train DART-YOLO with the Ultralytics training interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from register_modules import register_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DART-YOLO.")
    parser.add_argument("--model", default="DART-YOLO.yaml", help="Model YAML or checkpoint path.")
    parser.add_argument("--data", default="data/uav_tank.yaml", help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="dart_yolo", help="Run name.")
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_modules()
    model = YOLO(args.resume or args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        optimizer=args.optimizer,
        seed=args.seed,
        pretrained=args.pretrained,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        close_mosaic=args.close_mosaic,
        cache=False,
        single_cls=False,
        amp=True,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
