# Dataset Notes

## UAV-Tank

UAV-Tank contains 3,763 final images and one class:

```text
tank
```

Fixed split:

- `splits/uav_tank/train.txt`: 3,331 images
- `splits/uav_tank/val.txt`: 432 images

The dataset is reconstructed from public source projects. The full original images are not redistributed here because license terms differ across sources. The repository provides the source list, reconstruction process, and fixed split files.

### Source Records

UAV-Tank is reconstructed from five public object-detection source projects. The
original image archives are not redistributed in this repository. The source
records below follow the manuscript source table.

| Source project | Task type | License | Source project images | Role in UAV-Tank construction |
| --- | --- | --- | ---: | --- |
| military asset | Object detection | CC BY 4.0 | 633 | Tank labels retained. |
| FPV drone | Object detection | CC BY 4.0 | 296 | Tank samples retained and unified. |
| tank_500_0828 | Object detection | CC BY 4.0 | 500 | Tank-specific detection source. |
| tank test | Object detection | CC BY 4.0 | 582 | Tank detection annotations retained and unified. |
| tank | Object detection | CC BY 4.0 | 6,667 | Tank labels retained and unified. |

### Reconstruction Process

1. Manually select tank-related images from the source projects.
2. Inspect and standardize bounding boxes.
3. Remove non-tank classes.
4. Map heterogeneous tank-related labels to the single class `tank`.
5. Remove duplicate images.
6. Convert annotations to YOLO format.
7. Apply the fixed train/val lists in `splits/uav_tank/`.
8. Use the final 3,763-image, single-class dataset for training and evaluation.

Only object-detection bounding boxes are used in the final UAV-Tank protocol.

## VisDrone2019

Official VisDrone2019 train/val splits are used:

- train: 6,471 images
- val: 548 images

The test-dev and test-challenge splits are not used for training, validation, or hyperparameter tuning.

## DIOR

The DIOR experiments use a fixed 80/20 split:

- `splits/dior/train.txt`: 18,770 images
- `splits/dior/val.txt`: 4,693 images

This split is used for the manuscript experiments and is not an official leaderboard protocol.
