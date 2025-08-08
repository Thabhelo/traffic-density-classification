#!/usr/bin/env python3
"""
Fetch Singapore traffic images (data.gov.sg Traffic Images API) and create a ZIP
containing the downloaded images, placed at the path expected by the notebook.

This script does NOT label images. It stores them under:
  <out_dir>/raw/<camera_id>/<timestamp>.jpg

Default paths are set for Google Colab usage:
  - out_dir: /content/traffic_density
  - zip_path: /content/traffic-density-singapore.zip

Usage examples:
  - Colab (one snapshot):
      !python fetch_lta_images_to_zip.py --snapshots 1 --interval 1

  - Local (macOS):
      python3 fetch_lta_images_to_zip.py \
        --out_dir ./traffic_density \
        --zip_path ./traffic-density-singapore.zip \
        --snapshots 3 --interval 60
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests


API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"


def fetch_latest_snapshot(out_root: Path, timeout: int = 30) -> int:
    """Fetch the latest traffic images snapshot and save per camera.

    Returns the number of images written.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    response = requests.get(API_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    items = data.get("items") or []
    if not items:
        return 0

    cameras = items[0].get("cameras") or []
    written = 0
    for cam in cameras:
        img_url: Optional[str] = cam.get("image")
        cam_id: Optional[str] = cam.get("camera_id")
        ts: Optional[str] = cam.get("timestamp")
        if not img_url or not cam_id or not ts:
            continue

        # Clean timestamp for filename
        safe_ts = ts.replace(":", "-")
        ext = Path(img_url).suffix or ".jpg"

        cam_dir = out_root / "raw" / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        dest = cam_dir / f"{safe_ts}{ext}"
        if dest.exists():
            # Skip if already downloaded
            continue

        try:
            img_resp = requests.get(img_url, timeout=timeout)
            img_resp.raise_for_status()
        except Exception:
            continue

        with open(dest, "wb") as f:
            f.write(img_resp.content)
        written += 1

    return written


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    """Zip the contents of source_dir into zip_path, preserving relative paths.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for fn in files:
                full = Path(root) / fn
                rel = full.relative_to(source_dir.parent)
                zf.write(full, arcname=str(rel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Singapore traffic images and zip them for the notebook.")
    parser.add_argument("--out_dir", type=str, default="/content/traffic_density", help="Root directory to store images before zipping")
    parser.add_argument("--zip_path", type=str, default="/content/traffic-density-singapore.zip", help="Destination ZIP path expected by the notebook")
    parser.add_argument("--snapshots", type=int, default=1, help="Number of snapshots to fetch (one call per snapshot)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds to wait between snapshots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    zip_out = Path(args.zip_path).resolve()

    total_written = 0
    for i in range(max(1, args.snapshots)):
        written = fetch_latest_snapshot(out_dir)
        total_written += written
        if i + 1 < args.snapshots:
            time.sleep(max(1, args.interval))

    # Create the zip in the expected location
    # We zip the parent of `traffic_density` so that paths inside zip start with `traffic_density/...`
    # out_dir = /content/traffic_density → parent = /content
    parent = out_dir.parent
    if parent == out_dir:
        # Safety: if user set out_dir to a root-like path, zip the directory itself
        parent = out_dir

    zip_dir(source_dir=parent, zip_path=zip_out)

    print(f"Images written this run: {total_written}")
    print(f"ZIP created at: {zip_out}")
    print(f"Unzip in Colab with: !unzip {zip_out} -d /content/traffic_density")


if __name__ == "__main__":
    main()


