"""
Download the Kather 2016 colorectal histology dataset from Zenodo.

Usage:
    python -m src.download_data          # from project root
    python src/download_data.py          # also works

Downloads ~1.2 GB zip → extracts to data/Kather_texture_2016_image_tiles_5000/
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Allow running as both module and script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, ZENODO_URL, ZIP_FILENAME, DATASET_DIR


def download_file(url: str, dest_path: str) -> None:
    """Stream-download a file with a progress bar."""
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, ZIP_FILENAME)

    # Skip download if dataset already extracted
    if os.path.isdir(DATASET_DIR):
        print(f"Dataset already exists at {DATASET_DIR} — skipping download.")
        return

    # Download
    if not os.path.isfile(zip_path):
        print(f"Downloading from {ZENODO_URL} ...")
        download_file(ZENODO_URL, zip_path)
        print("Download complete.")
    else:
        print(f"Zip already exists at {zip_path} — skipping download.")

    # Extract
    print(f"Extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")

    # Clean up zip to save disk space
    os.remove(zip_path)
    print(f"Removed {zip_path}.")

    # Verify
    class_dirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
    print(f"\nFound {len(class_dirs)} class folders: {sorted(class_dirs)}")
    total_images = sum(
        len(os.listdir(os.path.join(DATASET_DIR, d))) for d in class_dirs
    )
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()
