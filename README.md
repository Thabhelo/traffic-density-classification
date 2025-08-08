# Traffic Density Classification (PyTorch · ConvNeXt-Tiny)

Classify traffic density from camera images into five categories: `Empty`, `Low`, `Medium`, `High`, `Traffic Jam`. The notebook uses PyTorch with a ConvNeXt‑Tiny backbone (via `timm`) and a two‑stage transfer learning recipe (feature extraction then fine‑tuning). On the provided dataset (4,054 images from Singapore LTA cameras), the model reaches around 91% test accuracy.

### Open in Colab
- Use the badge at the top of the notebook, or open:
  - `https://colab.research.google.com/github/Thabhelo/traffic-density-classification/blob/main/Traffic_Density_Classification_with_EfficientNet.ipynb`

### Dataset
- Source: Singapore Land Transport Authority (LTA) API (images captured by selected cameras under varied lighting and viewpoints)
- Classes: `Empty`, `Low`, `Medium`, `High`, `Traffic Jam`
- Split: 80% train, 10% val, 10% test
- The dataset zip is not included in this repository. In Colab, upload or download it and unzip to:
  - `/content/traffic_density/Final Dataset/{training,validation,testing}/<ClassName>`
- If running locally, adjust the paths in the configuration cell to point to your dataset.

### Method (brief)
- Backbone: ConvNeXt‑Tiny pretrained weights from `timm`
- Input size: 384×384; transforms include RandAugment and Random Erasing
- Training: AdamW, OneCycleLR, MixUp/CutMix with Soft‑Target Cross‑Entropy, AMP where available
- Two stages: feature extraction (freeze backbone), then fine‑tuning (unfreeze all)
- Outputs: model weights saved to `/content/convnext_feature_extractor.pth` and `/content/convnext_finetuned.pth`, plus a classification report and confusion matrix

### Quick start (local macOS, Apple Silicon)
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
- Open the notebook and set dataset paths if not using Colab. The notebook auto‑detects MPS (Apple Silicon), CUDA, or CPU.

### Quick start (Colab)
```bash
!pip install albumentations torch torchvision timm grad-cam
```
- Upload or mount the dataset zip; unzip to `/content/traffic_density/`.
- Run all cells sequentially.

### Results
- On the provided split, test accuracy is typically ~0.91 with balanced per‑class precision/recall.

### Acknowledgements
- Singapore Land Transport Authority (LTA)
- Sudhanshu Rastogi

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.