# Training

All training runs use 640 x 640 input resolution, SGD optimizer, fixed random seed, and the dataset split defined by the corresponding YAML file.

## UAV-Tank

```bash
python train.py --model DART-YOLO.yaml --data data/uav_tank.yaml --imgsz 640 --epochs 200 --batch 8 --optimizer SGD --seed 0 --pretrained --project runs/train --name dart_yolo_uav_tank
```

## VisDrone2019

```bash
python train.py --model DART-YOLO.yaml --data data/visdrone2019.yaml --imgsz 640 --epochs 200 --batch 8 --optimizer SGD --seed 0 --pretrained --project runs/train --name dart_yolo_visdrone2019
```

## DIOR

```bash
python train.py --model DART-YOLO.yaml --data data/dior.yaml --imgsz 640 --epochs 200 --batch 8 --optimizer SGD --seed 0 --pretrained --project runs/train --name dart_yolo_dior
```

## Notes

- `--pretrained` enables Ultralytics pretrained initialization when available.
- `--no-pretrained` trains from scratch.
- `--resume path/to/last.pt` resumes an interrupted run.
- Dataset paths in `data/*.yaml` should be adjusted to match the local dataset location.
