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
full source manifest is maintained with the dataset reconstruction records rather
than embedded in this code repository, because the original image archives are
not redistributed here. Each source record contains the project name, URL, task
type, license, original image count, and its use in the final UAV-Tank subset.
All five source records are treated as `Object detection` sources in the
manuscript protocol.

### Reconstruction Process

1. Manually select tank-related images from the source projects.
2. Inspect and standardize bounding boxes.
3. Remove non-tank classes.
4. Map heterogeneous tank-related labels to the single class `tank`.
5. Remove duplicate images.
6. Convert annotations to YOLO format.
7. Apply the fixed train/val lists in `splits/uav_tank/`.
8. Use the final 3,763-image, single-class dataset for training and evaluation.

No segmentation labels, masks, or mask-to-box conversion are used in the final UAV-Tank protocol.

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
