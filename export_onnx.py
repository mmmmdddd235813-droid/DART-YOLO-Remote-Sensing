"""Export DART-YOLO checkpoints to ONNX FP32, batch size 1, 640x640."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from register_modules import register_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DART-YOLO to ONNX FP32.")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_modules()
    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        half=False,
        int8=False,
        dynamic=False,
        simplify=args.simplify,
        opset=args.opset,
        nms=False,
    )


if __name__ == "__main__":
    main()
