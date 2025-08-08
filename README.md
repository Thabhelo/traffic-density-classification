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

### Adaptation and originality
This work was initially inspired by and adapted from Sudhanshu Rastogi’s traffic density classification project. It has been substantially redesigned (approximately 70% of the pipeline and code paths) to improve capability, reproducibility, and usability:
- Replaced Google’s EfficientNet‑B0 with Meta’s ConvNeXt‑Tiny (via `timm`), chosen for stronger performance at similar compute and a modern training recipe.
- Introduced a new training stack: AdamW optimizer, OneCycleLR scheduler, mixed‑precision (AMP), MixUp/CutMix with Soft‑Target Cross‑Entropy, and deterministic evaluation.
- Switched to high‑quality `timm` transforms at 384×384 with RandAugment and Random Erasing.
- Added Apple Silicon (MPS) support, better device handling, and pin‑memory hygiene.
- Implemented optional explainability (Grad‑CAM) and automated dataset creation: API fetching and YOLO‑based auto‑labeling into density buckets, with end‑to‑end scripts and a turnkey notebook setup cell.
These changes make the project easy to fork and run while providing a clearly original, extended feature set beyond the source inspiration.

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

> NOTE:
I had to subscribe to Google Colab’s premium tier to get faster GPU access for the convolutional neural network to handle high‑resolution image classification. On my 2024 MacBook Air with the M3 chip — powerful as it is — the process could take around 20 hours. By switching to a cloud‑based GPU (in this case, the NVIDIA P100), runtime drops to about 30–45 minutes. Colab’s architecture supports both Python and R, and grants up to 89.6 GB of RAM, making it a crucial resource for memory‑intensive tasks. If you need to run something that pushes beyond what a local CPU can handle, I highly recommend it! If you think we are friends, let me know and I can grant you access to my paid subscription.

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

#### Build your own dataset programmatically (optional)
1) Fetch raw images from Singapore's data.gov.sg Traffic Images API and create a ZIP expected by the notebook:
```bash
!python fetch_lta_images_to_zip.py --snapshots 3 --interval 60
!unzip /content/traffic-density-singapore.zip -d /content/traffic_density
```
2) Auto‑label images into `Empty/Low/Medium/High/Traffic Jam` using a YOLO vehicle counter and create the final dataset ZIP:
```bash
!pip install ultralytics tqdm
!python auto_label_density.py \
  --raw_dir /content/traffic_density/raw \
  --out_root /content/traffic_density \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --model yolov8n.pt --conf 0.25 --ymin 0.0 --ymax 1.0 \
  --make_zip --zip_path /content/traffic-density-singapore.zip
!unzip /content/traffic-density-singapore.zip -d /content/traffic_density
```
Notes:
- Thresholds for density mapping can be tuned: `--thr_low`, `--thr_med`, `--thr_high`.
- Keep weight files (`.pth`) outside dataset folders.

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
