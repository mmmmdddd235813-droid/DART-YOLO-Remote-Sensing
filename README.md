# DART-YOLO: Dynamic Rotational Attention for Enhanced Small Target Detection in Remote Sensing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
This repository contains the official implementation of the paper **"Dynamic Rotational Attention for Enhanced Small Target Detection in Remote Sensing"**, submitted to *The Visual Computer*.

## 📖 Abstract
Small target detection in remote sensing images remains a significant challenge due to variable target scales, complex backgrounds, and dense distributions. To address these issues, we introduce a lightweight and efficient detection model, **DART (Dynamic Rotational Attention)**, based on the YOLO framework. Our model integrates:
* **DRFM (Dynamic Rotational Feature Module):** For rotation-adaptive capabilities.
* **LMSA (Lightweight Multi-Scale Attention):** For contextual information enhancement.
* **DGAM (Dynamic Global Attention Module):** For incorporating global context and positional awareness.

Extensive experiments on UAV-tank, VisDrone2019, and DIOR datasets demonstrate significant improvements in mAP50 alongside a reduction in model size.

## 📂 Repository Structure
The code is organized as follows:
* `train_0928.py`: The main training script to reproduce the experiments.
* `DART-YOLO.yaml`: The model configuration file defining the network architecture.
* `DRFM.py`: Implementation of the Dynamic Rotational Feature Module.
* `DGAM.py`: Implementation of the Dynamic Global Attention Module.
* `LMSA.py`: Implementation of the Lightweight Multi-Scale Attention module.
* `SPD.py`: Space-to-Depth module implementation.
* `requirements.txt`: List of Python dependencies.

## 🚀 Getting Started

### 1. Environment Setup
The code requires Python 3.8+ and PyTorch. We recommend using a Conda environment.

bash
# Clone the repository
git clone [https://github.com/mmmmdddd235813-droid/DART-YOLO-Remote-Sensing.git](https://github.com/mmmmdddd235813-droid/DART-YOLO-Remote-Sensing.git)
cd DART-YOLO-Remote-Sensing

# Install dependencies
pip install -r requirements.txt

## Dataset Preparation
The datasets used in this paper are publicly available. Please download them from their official repositories:

VisDrone2019 Dataset:

Official GitHub: https://github.com/VisDrone/VisDrone-Dataset

Description: A large-scale benchmark for object detection in drone-captured images.

DIOR Dataset:

Official GitHub: https://github.com/gcheng-nwpu/DIOR-Dataset

Description: A large-scale benchmark dataset for object detection in optical remote sensing images.

UAV-Tank Dataset:

Note: Please ensure the dataset follows the standard YOLO directory structure.

Directory Structure: Please/datasets
/datasets
    /VisDrone
        /images
        /labels
    /DIOR
        /images
        /labels
        
## Training

To train the DART-YOLO model on your dataset, run the main training script:
python train.py

## 🔗 Citation
If you use this code or model in your research, please refer to our manuscript submitted to The Visual Computer:

@article{DART_YOLO_2025,
  title={Dynamic Rotational Attention for Enhanced Small Target Detection in Remote Sensing},
  author={Dezhi Sun and Qiang Shen and others},
  journal={The Visual Computer},
  year={2025},
  note={Under Review}
}
