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
original image archives are not redistributed in this repository. The table below
lists the public source records used for reconstruction.

| Source project | URL | Task type | License | Original image count | Role |
| --- | --- | --- | --- | ---: | --- |
| Tank Dataset, Onyx | https://universe.roboflow.com/onyx/tank-qgurh | Object detection | CC BY 4.0 | 1,279 | Tank-only source images and bounding boxes for seed samples. |
| RussiantTankDroneImagesLowQuality Dataset, tank | https://universe.roboflow.com/tank-s4xwz/russianttankdroneimageslowquality | Object detection | CC BY 4.0 | 993 | Low-quality drone images of armored vehicles; tank-related labels were retained and standardized. |
| military tank detection Dataset, Keshav Memorial Institute of Technology | https://universe.roboflow.com/keshav-memorial-institute-of-technology-lbgah/military-tank-detection-rwlhy | Object detection | CC BY 4.0 | 800 | Additional tank detection samples for appearance diversity. |
| Aerial Tanks Dataset, Yamen | https://universe.roboflow.com/yamen-gm7rm/aerial-tanks | Object detection | CC BY 4.0 | 618 | Aerial tank samples for scale and viewpoint diversity. |
| Tanks detection Dataset, tanks | https://universe.roboflow.com/tanks-r1anu/tanks-detection-d0ayl-eympz | Object detection | MIT | 2,586 | Multi-type tank source images used after class filtering and tank-label unification. |

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
