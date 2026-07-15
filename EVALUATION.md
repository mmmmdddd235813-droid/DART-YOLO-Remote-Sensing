# Evaluation

The evaluation protocol uses the validation split specified in each dataset YAML file. The reported metrics are Precision, Recall, mAP@0.5, and mAP@0.5:0.95.

## Command

```bash
python evaluate.py --weights runs/train/dart_yolo_uav_tank/weights/best.pt --data data/uav_tank.yaml --imgsz 640 --batch 8 --split val
```

## Dataset-Specific Examples

```bash
python evaluate.py --weights runs/train/dart_yolo_uav_tank/weights/best.pt --data data/uav_tank.yaml
python evaluate.py --weights runs/train/dart_yolo_visdrone2019/weights/best.pt --data data/visdrone2019.yaml
python evaluate.py --weights runs/train/dart_yolo_dior/weights/best.pt --data data/dior.yaml
```

The script also writes `metrics_summary.json` to the evaluation run directory.
