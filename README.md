# DART-YOLO: Rotation-Adaptive Multi-Scale Attention for Efficient Small-Object Detection in Remote-Sensing Imagery

This repository contains the reference implementation for DART-YOLO, a parameter-efficient and accuracy-oriented YOLO-based detector for remote sensing object detection. The model is designed for small and densely distributed targets in aerial and remote sensing imagery.

The implementation follows the manuscript submitted to *The Visual Computer*. The public repository focuses on transparent model components, reproducible training commands, dataset split documentation, evaluation, ONNX export, and Jetson ONNX Runtime benchmarking.

## Model Components

DART-YOLO is built from four named modules:

- `CSD.py`: Channel-Split Downsampling module.
- `DRFM.py`: Dynamic Rotational Feature Module.
- `LMSA.py`: Lightweight Multi-Scale Attention module.
- `DGAM.py`: Dynamic Global Attention Module.

The full model configuration is provided in:

- `DART-YOLO.yaml`

`SPD.py` is kept only as a backward-compatible alias for `CSD.py`.

## Repository Structure

```text
DART-YOLO-Remote-Sensing/
  DART-YOLO.yaml
  CSD.py
  DRFM.py
  LMSA.py
  DGAM.py
  SPD.py
  train.py
  evaluate.py
  export_onnx.py
  benchmark_jetson_ort.py
  requirements.txt
  data/
    uav_tank.yaml
    visdrone2019.yaml
    dior.yaml
  splits/
    uav_tank/train.txt
    uav_tank/val.txt
    dior/train.txt
    dior/val.txt
  tools/
    convert_to_yolo.py
    filter_tank_classes.py
    unify_tank_labels.py
    remove_duplicates.py
    check_annotations.py
  docs/
    DATASETS.md
  TRAINING.md
  EVALUATION.md
  DEPLOYMENT.md
  CITATION.cff
```

## Installation

Python 3.12.7, PyTorch 2.3.1 with CUDA 12.7, and Ultralytics 8.3.159 were used for the main experiments.

```bash
git clone https://github.com/mmmmdddd235813-droid/DART-YOLO-Remote-Sensing.git
cd DART-YOLO-Remote-Sensing
pip install -r requirements.txt
```

The custom modules are written as regular PyTorch modules. To use `DART-YOLO.yaml` directly with Ultralytics, make sure these modules are registered or copied into the Ultralytics custom module import path used by your training environment.

## Datasets

Three datasets are used:

- UAV-Tank: 3,763 images, single class `tank`.
- VisDrone2019: official train/val splits, with 6,471 train images and 548 val images.
- DIOR: fixed 80/20 split used in the manuscript, with 18,770 train images and 4,693 val images. This is not an official leaderboard protocol.

UAV-Tank images are not redistributed in full because some source datasets have license restrictions. The repository provides source information, reconstruction steps, and fixed split files. See [docs/DATASETS.md](docs/DATASETS.md).

## Training

Default training command:

```bash
python train.py --model DART-YOLO.yaml --data data/uav_tank.yaml --imgsz 640 --epochs 200 --batch 8 --optimizer SGD --seed 0 --no-pretrained
```

All principal detector comparisons reported in the manuscript were trained without COCO-pretrained weights.

Examples for the three datasets are listed in [TRAINING.md](TRAINING.md).

## Evaluation

```bash
python evaluate.py --weights runs/train/dart_yolo/weights/best.pt --data data/uav_tank.yaml --imgsz 640 --batch 8
```

The evaluation script reports Precision, Recall, mAP@0.5, and mAP@0.5:0.95. See [EVALUATION.md](EVALUATION.md).

## ONNX Export and Jetson Benchmarking

Export FP32 ONNX at 640 x 640 with batch size 1:

```bash
python export_onnx.py --weights runs/train/dart_yolo/weights/best.pt --imgsz 640
```

Benchmark on Jetson AGX Xavier 32 GB with ONNX Runtime 1.12.1 and CUDAExecutionProvider:

```bash
python benchmark_jetson_ort.py --onnx best.onnx --warmup 50 --runs 300
```

See [DEPLOYMENT.md](DEPLOYMENT.md).

## Citation

If this repository is useful for your work, please cite the manuscript once the final bibliographic information is available. A provisional citation file is provided in [CITATION.cff](CITATION.cff).
