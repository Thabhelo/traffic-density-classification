#!/usr/bin/env python3
"""
Programmatically auto-label traffic images by counting vehicles with YOLO and
organize them into traffic density classes:
  - Empty, Low, Medium, High, Traffic Jam

It copies images into:
  <out_root>/Final Dataset/{training,validation,testing}/<ClassName>/

and optionally creates the notebook-expected ZIP:
  /content/traffic-density-singapore.zip (override with --zip_path)

Example (Colab):
  !pip install ultralytics tqdm
  !python auto_label_density.py \
      --raw_dir /content/traffic_density/raw \
      --out_root /content/traffic_density \
      --make_zip --zip_path /content/traffic-density-singapore.zip

Example (local):
  python3 auto_label_density.py \
      --raw_dir ./traffic_density/raw \
      --out_root ./traffic_density \
      --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
      --model yolov8n.pt --conf 0.25 --make_zip --zip_path ./traffic-density-singapore.zip
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# Delay import so script can show helpful error if missing
try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASSES_DEFAULT = ["car", "bus", "truck", "motorbike"]


@dataclass
class Thresholds:
    low: int = 5
    medium: int = 15
    high: int = 30

    def to_label(self, count: int) -> str:
        if count <= 0:
            return "Empty"
        if count <= self.low:
            return "Low"
        if count <= self.medium:
            return "Medium"
        if count <= self.high:
            return "High"
        return "Traffic Jam"


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def within_vertical_roi(box_xyxy: np.ndarray, ymin: float, ymax: float, img_h: int) -> bool:
    x1, y1, x2, y2 = box_xyxy
    cy = (y1 + y2) / 2.0
    y_norm = float(cy) / float(max(1, img_h))
    return ymin <= y_norm <= ymax


def count_vehicles(model: "YOLO", image_path: Path, classes: List[str], conf: float, ymin: float, ymax: float) -> int:
    res = model(str(image_path), conf=conf, verbose=False)[0]
    if res.boxes is None or res.boxes.shape[0] == 0:
        return 0
    count = 0
    img_h, img_w = int(res.orig_shape[0]), int(res.orig_shape[1])
    id_to_name = res.names  # dict id->name
    for i in range(len(res.boxes)):
        cls_id = int(res.boxes.cls[i].item())
        cls_name = id_to_name.get(cls_id, str(cls_id))
        if cls_name not in classes:
            continue
        xyxy = res.boxes.xyxy[i].cpu().numpy()
        if within_vertical_roi(xyxy, ymin=ymin, ymax=ymax, img_h=img_h):
            count += 1
    return count


def random_split(seed: int, train_r: float, val_r: float, test_r: float) -> str:
    rnd = random.Random(seed).random()
    if rnd < train_r:
        return "training"
    if rnd < train_r + val_r:
        return "validation"
    return "testing"


def zip_for_notebook(out_root: Path, zip_path: Path) -> None:
    # Zip the parent so resulting archive contains traffic_density/... at root
    parent = out_root.parent
    if parent == out_root:
        parent = out_root
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(parent):
            for fn in files:
                full = Path(root) / fn
                rel = full.relative_to(parent)
                zf.write(full, arcname=str(rel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label traffic density by counting vehicles with YOLO.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Root directory of raw images (per camera or flat)")
    parser.add_argument("--out_root", type=str, required=True, help="Output root that will contain 'Final Dataset' structure")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--classes", type=str, default=",".join(CLASSES_DEFAULT), help="Comma-separated class names to count")
    parser.add_argument("--ymin", type=float, default=0.0, help="Vertical ROI min (0..1)")
    parser.add_argument("--ymax", type=float, default=1.0, help="Vertical ROI max (0..1)")

    parser.add_argument("--thr_low", type=int, default=5)
    parser.add_argument("--thr_med", type=int, default=15)
    parser.add_argument("--thr_high", type=int, default=30)

    parser.add_argument("--make_zip", action="store_true", help="Create notebook-expected zip after labeling")
    parser.add_argument("--zip_path", type=str, default="/content/traffic-density-singapore.zip")

    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    thresholds = Thresholds(low=args.thr_low, medium=args.thr_med, high=args.thr_high)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    raw_dir = Path(args.raw_dir).resolve()
    out_root = Path(args.out_root).resolve()
    final_root = out_root / "Final Dataset"
    for split in ("training", "validation", "testing"):
        for cls in ("Empty", "Low", "Medium", "High", "Traffic Jam"):
            (final_root / split / cls).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    summary_rows: List[Tuple[str, int, str, str]] = []
    image_paths = list(iter_images(raw_dir))
    if not image_paths:
        print(f"No images found in {raw_dir}")
        return

    random.seed(args.seed)
    for img_path in image_paths:
        count = count_vehicles(model, img_path, classes=classes, conf=args.conf, ymin=args.ymin, ymax=args.ymax)
        label = thresholds.to_label(count)
        # Split by random draw per image (seeded)
        split = random_split(seed=args.seed + hash(img_path) % (10**6),
                             train_r=args.train_ratio, val_r=args.val_ratio, test_r=args.test_ratio)

        dest = final_root / split / label / img_path.name
        shutil.copy2(img_path, dest)
        summary_rows.append((str(img_path), count, label, split))

    # Save a CSV summary
    csv_path = out_root / "auto_labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "vehicle_count", "label", "split"])
        w.writerows(summary_rows)

    # Optionally create the notebook-expected zip
    if args.make_zip:
        zip_path = Path(args.zip_path).resolve()
        # Zip parent of out_root so archive contains traffic_density/... at root
        parent = out_root.parent
        if parent == out_root:
            parent = out_root
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(parent):
                for fn in files:
                    full = Path(root) / fn
                    rel = full.relative_to(parent)
                    zf.write(full, arcname=str(rel))

        print(f"ZIP created at: {zip_path}")

    # Summary to stdout
    totals = {}
    for _, _, label, split in summary_rows:
        totals.setdefault((split, label), 0)
        totals[(split, label)] += 1
    print("Label counts by split:")
    for (split, label), n in sorted(totals.items()):
        print(f"  {split:10s} {label:12s}: {n}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()


