"""Benchmark an ONNX model with ONNX Runtime on Jetson AGX Xavier 32 GB.

Protocol used in the paper:
- ONNX Runtime 1.12.1
- CUDAExecutionProvider
- FP32 ONNX model
- batch size 1
- 640x640 input

Timing includes ONNX Runtime inference only by default. Image decoding,
preprocessing, and postprocessing are excluded unless you extend this script.
FPS is computed as 1000 / average_latency_ms.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DART-YOLO ONNX with ONNX Runtime CUDAExecutionProvider.")
    parser.add_argument("--onnx", required=True, help="Path to FP32 ONNX model.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=300, help="Timed iterations.")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.onnx, providers=providers)
    active_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)

    for _ in range(args.warmup):
        session.run(None, {input_name: dummy})

    start = time.perf_counter()
    for _ in range(args.runs):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    avg_ms = elapsed * 1000.0 / args.runs
    fps = 1000.0 / avg_ms
    print(f"provider: {active_provider}")
    print(f"warmup_iterations: {args.warmup}")
    print(f"timed_iterations: {args.runs}")
    print(f"average_latency_ms: {avg_ms:.3f}")
    print(f"fps: {fps:.2f}")
    print("timing_scope: ONNX Runtime inference only; preprocessing and postprocessing are excluded.")


if __name__ == "__main__":
    main()
