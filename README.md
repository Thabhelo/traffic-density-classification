# Deep Learning - Traffic Density Classification

This notebook leverages a custom dataset collected from Singapore's Land Transport Authority (LTA) API to build and evaluate deep learning models for traffic density classification. The images are labeled across five traffic density levels and are captured under varying conditions, including different lighting and camera angles.

---

## Table of Contents

- Overview
- Dataset Description
- Data Splitting
- Usage Instructions
- Dependencies (PyTorch)
- Method Overview (ConvNeXt‑Tiny)
- Quick Start (Local and Colab)
- Results
- Acknowledgements
- License

---

## Overview

This project is centered around developing a robust deep learning solution to classify traffic density based on images captured from multiple traffic cameras. The dataset includes images taken during both daytime and nighttime, ensuring that the models are well‑equipped to handle diverse lighting and environmental conditions.

---

## Dataset Description

- Source: Singapore Land Transport Authority (LTA) API  
- Image Count: 4,054 images  
- Traffic Density Categories:  
  - Empty  
  - Low  
  - Medium  
  - High  
  - Traffic Jam  
- Camera Selection:  
  Out of 87 available traffic cameras, 20 were carefully selected based on:  
  - Variability in traffic density  
  - Different lighting conditions  
  - Clear visibility without obstructions  
  - Suitable camera angles  

The dataset provides a well‑rounded collection of images, making it ideal for training, validating, and testing deep learning models focused on traffic analysis.

---

## Data Splitting

The dataset has been divided into three subsets to ensure robust model evaluation:

- Training Set: 80% of the dataset  
- Validation Set: 10% of the dataset  
- Testing Set: 10% of the dataset  

This split enables effective model training while ensuring that performance metrics are evaluated on unseen data.

---

## Usage Instructions

1) Open the notebook in Colab using the badge at the top, or directly:
   - `https://colab.research.google.com/github/Thabhelo/traffic-density-classification/blob/main/Traffic_Density_Classification_with_EfficientNet.ipynb`
2) Prepare the dataset:
   - Upload or download the dataset zip (not included in this repo) and unzip to:
     - `/content/traffic_density/Final Dataset/{training,validation,testing}/<ClassName>`
   - If running locally, update the dataset paths in the configuration cell accordingly.
3) Run the notebook cells sequentially to preprocess data, train the model (two stages), and evaluate performance.

Note on runtime: On a local CPU, training can be slow. Using Colab with a GPU significantly reduces runtime. Apple Silicon (M‑series) is supported via PyTorch MPS and works well for experimentation.

---

## Dependencies (PyTorch)

Core libraries used by the notebook:
- PyTorch (`torch`, `torchvision`)
- PyTorch Image Models (`timm`)
- Albumentations (data augmentation)
- Grad‑CAM (`grad-cam`) for model interpretability
- NumPy, Pandas, Matplotlib, Seaborn, scikit‑learn, OpenCV (headless)

Example pinned setup (local macOS, Apple Silicon):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install \
  torch==2.8.0 torchvision==0.23.0 timm==1.0.9 \
  albumentations==1.4.20 grad-cam==1.5.5 \
  numpy==2.2.6 pandas==2.2.3 seaborn==0.13.2 matplotlib==3.9.2 \
  scikit-learn==1.5.2 opencv-python-headless==4.12.0.88
```

---

## Method Overview (ConvNeXt‑Tiny)

- Backbone: ConvNeXt‑Tiny (via `timm`), initialized with pretrained weights  
- Input size: 384×384  
- Transforms: RandAugment and Random Erasing for training, ImageNet normalization  
- Training: AdamW optimizer, OneCycleLR scheduler, MixUp/CutMix regularization with Soft‑Target Cross‑Entropy, AMP where available (CUDA/MPS)  
- Two stages:  
  - Feature extraction: freeze backbone, train classifier head  
  - Fine‑tuning: unfreeze all layers, train with a lower learning rate  
- Outputs:  
  - `/content/convnext_feature_extractor.pth`  
  - `/content/convnext_finetuned.pth`  
  - Classification report and confusion matrix  

---

## Quick Start

### Local (macOS, Apple Silicon)
Use the pinned setup above, open the notebook, and point the dataset paths to your local folders. The notebook auto‑detects MPS (Apple Silicon), CUDA, or CPU.

### Colab
```bash
!pip install albumentations torch torchvision timm grad-cam
```
Upload or mount the dataset, unzip to `/content/traffic_density/`, then run all cells.

---

## Results

On the provided split, test accuracy is typically around 0.91 with balanced per‑class precision and recall.

---

## Acknowledgements

- Singapore Land Transport Authority (LTA)  
- Sudhanshu Rastogi  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.