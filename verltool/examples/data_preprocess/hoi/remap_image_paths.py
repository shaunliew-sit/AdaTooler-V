"""
Remap hardcoded local image paths in HOI parquet files to cluster paths.

The parquet stores absolute image paths (set at data-prep time on local machine).
This script patches those paths without regenerating the full dataset, preserving
the already-embedded YOLO proposals in the prompt text.

Usage (run on cluster from verltool/):
    python examples/data_preprocess/hoi/remap_image_paths.py \
        --hico_cluster_dir /workspace/data/hico_20160224_det \
        --swig_cluster_dir /workspace/data/swig_hoi \
        --parquet_dir hoi_rl
"""
import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Path remapping
# ---------------------------------------------------------------------------

def remap_path(old_path: str, hico_dir: str, swig_dir: str) -> str:
    """Replace local path prefix with cluster path prefix."""
    p = Path(old_path)
    parts = p.parts

    # Find anchor: hico_20160224_det or swig_hoi in the path
    for i, part in enumerate(parts):
        if part == "hico_20160224_det":
            suffix = Path(*parts[i + 1:])
            return str(Path(hico_dir) / suffix)
        if part == "swig_hoi":
            suffix = Path(*parts[i + 1:])
            return str(Path(swig_dir) / suffix)

    # Unknown prefix — return unchanged and warn
    print(f"  WARNING: could not remap path: {old_path}")
    return old_path


def remap_images_field(images: list, hico_dir: str, swig_dir: str) -> list:
    """Remap the 'images' column: list of dicts with 'image' key."""
    remapped = []
    for item in images:
        new_item = dict(item)
        if "image" in new_item and isinstance(new_item["image"], str):
            new_item["image"] = remap_path(new_item["image"], hico_dir, swig_dir)
        remapped.append(new_item)
    return remapped


def remap_extra_info(extra_info: dict, hico_dir: str, swig_dir: str) -> dict:
    """Remap 'images' list inside extra_info dict."""
    new_extra = dict(extra_info)
    if "images" in new_extra and isinstance(new_extra["images"], list):
        new_extra["images"] = [
            remap_path(p, hico_dir, swig_dir) if isinstance(p, str) else p
            for p in new_extra["images"]
        ]
    return new_extra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hico_cluster_dir", required=True,
                        help="Cluster path to hico_20160224_det root")
    parser.add_argument("--swig_cluster_dir", required=True,
                        help="Cluster path to swig_hoi root")
    parser.add_argument("--parquet_dir", default="data/hoi",
                        help="Directory containing train.parquet and val.parquet")
    args = parser.parse_args()

    hico_dir = args.hico_cluster_dir.rstrip("/")
    swig_dir = args.swig_cluster_dir.rstrip("/")
    parquet_dir = Path(args.parquet_dir)

    for split in ["train", "val"]:
        parquet_path = parquet_dir / f"{split}.parquet"
        if not parquet_path.exists():
            print(f"Skipping {parquet_path} (not found)")
            continue

        print(f"\nProcessing {parquet_path} ...")
        df = pd.read_parquet(parquet_path)
        print(f"  Rows: {len(df)}")

        # Show a before-sample
        sample_before = df.iloc[0]["images"]
        print(f"  Before: {sample_before[0]['image'] if sample_before else 'N/A'}")

        # Remap 'images' column
        df["images"] = df["images"].apply(
            lambda imgs: remap_images_field(imgs, hico_dir, swig_dir)
        )

        # Remap paths inside 'extra_info'
        df["extra_info"] = df["extra_info"].apply(
            lambda ei: remap_extra_info(ei, hico_dir, swig_dir)
        )

        # Show after-sample
        sample_after = df.iloc[0]["images"]
        print(f"  After:  {sample_after[0]['image'] if sample_after else 'N/A'}")

        # Save back
        df.to_parquet(str(parquet_path), index=False)
        print(f"  Saved {parquet_path}")

    print("\nDone. Parquet image paths updated for cluster.")


if __name__ == "__main__":
    main()
