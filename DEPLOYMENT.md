# Deployment

## ONNX Export

The paper deployment setting uses FP32 ONNX with 640 x 640 input and batch size 1. FP16, INT8, and TensorRT are not used by `export_onnx.py`.

```bash
python export_onnx.py --weights runs/train/dart_yolo_uav_tank/weights/best.pt --imgsz 640
```

## Jetson AGX Xavier Benchmark

The benchmark script targets Jetson AGX Xavier 32 GB with ONNX Runtime 1.12.1 and CUDAExecutionProvider.

```bash
python benchmark_jetson_ort.py --onnx runs/train/dart_yolo_uav_tank/weights/best.onnx --warmup 50 --runs 300
```

The default timing scope is ONNX Runtime inference only. Image decoding, resizing, normalization, NMS, and other postprocessing steps are excluded unless the script is extended.

FPS is computed as:

```text
FPS = 1000 / average_latency_ms
```

The JetPack and CUDA versions should be reported according to the actual Jetson device used for benchmarking.
