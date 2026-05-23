#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download CTVid-Bench images and videos from HuggingFace.

Usage:
  python tools/download_data.py --output_dir ./data
  python tools/download_data.py --variants GT blur downsample_x4 --output_dir ./data
  python tools/download_data.py --type spatial_qa --variants GT
"""

import os
import argparse


REPO_ID = "CTVid/CTVid-Bench"

SPATIAL_SIZES = {"GT": "9.0 GB", "blur": "8.7 GB", "downsample_x4": "1.1 GB"}
TEMPORAL_IMG_SIZES = {"GT": "31 GB", "blur": "21 GB", "downsample_x4": "2.8 GB"}
TEMPORAL_VID_SIZES = {"GT": "332 MB", "blur": "278 MB", "downsample_x4": "53 MB"}


def check_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        return True
    except ImportError:
        print("[error] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False


def download_spatial_images(variants: list, output_dir: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    for variant in variants:
        size = SPATIAL_SIZES.get(variant, "?")
        print(f"\n[spatial_qa] Downloading '{variant}' images ({size}) …")
        local_dir = os.path.join(output_dir, "spatial_qa", "images", variant)
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"spatial_qa/images/{variant}/*"],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"  Saved to: {local_dir}")


def download_temporal_images(variants: list, output_dir: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    for variant in variants:
        size = TEMPORAL_IMG_SIZES.get(variant, "?")
        print(f"\n[temporal_qa] Downloading '{variant}' images ({size}) …")
        local_dir = os.path.join(output_dir, "temporal_qa", "images", variant)
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"temporal_qa/images/{variant}/*"],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"  Saved to: {local_dir}")


def download_temporal_videos(variants: list, output_dir: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    for variant in variants:
        size = TEMPORAL_VID_SIZES.get(variant, "?")
        print(f"\n[temporal_qa] Downloading '{variant}' videos ({size}) …")
        local_dir = os.path.join(output_dir, "temporal_qa", "videos", variant)
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"temporal_qa/videos/{variant}/*"],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"  Saved to: {local_dir}")


def print_size_summary(variants: list, data_type: str) -> None:
    print(f"\nEstimated download sizes for variants: {variants}")
    if data_type in ("spatial_qa", "all"):
        total_spatial = sum(
            float(SPATIAL_SIZES.get(v, "0").split()[0])
            for v in variants if v in SPATIAL_SIZES
        )
        print(f"  Spatial QA images:  ~{total_spatial:.1f} GB")
    if data_type in ("temporal_qa", "all"):
        total_timg = sum(
            float(TEMPORAL_IMG_SIZES.get(v, "0").split()[0])
            for v in variants if v in TEMPORAL_IMG_SIZES
        )
        total_tvid = sum(
            float(TEMPORAL_VID_SIZES.get(v, "0").replace(" MB", "e-3").replace(" GB", ""))
            for v in variants if v in TEMPORAL_VID_SIZES
        )
        print(f"  Temporal QA images: ~{total_timg:.1f} GB")
        print(f"  Temporal QA videos: ~{total_tvid*1000:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download CTVid-Bench data from HuggingFace.")
    parser.add_argument(
        "--variants", nargs="+", default=["GT", "blur", "downsample_x4"],
        choices=["GT", "blur", "downsample_x4"],
        help="Which quality variants to download (default: all three)."
    )
    parser.add_argument(
        "--type", default="all",
        choices=["all", "spatial_qa", "temporal_qa"],
        help="Which data type to download (default: all)."
    )
    parser.add_argument(
        "--output_dir", default="./data",
        help="Local output directory (default: ./data)."
    )
    parser.add_argument(
        "--no_videos", action="store_true",
        help="Skip temporal QA video download (saves 663 MB)."
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace access token (only needed if dataset is gated)."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print size summary without downloading."
    )
    args = parser.parse_args()

    if not check_huggingface_hub():
        return

    print_size_summary(args.variants, args.type)

    if args.dry_run:
        print("\n[dry_run] No files downloaded.")
        return

    if args.type in ("all", "spatial_qa"):
        download_spatial_images(args.variants, args.output_dir, args.token)

    if args.type in ("all", "temporal_qa"):
        download_temporal_images(args.variants, args.output_dir, args.token)
        if not args.no_videos:
            download_temporal_videos(args.variants, args.output_dir, args.token)

    print(f"\nDownload complete. Update configs/default.yaml with:")
    print(f"  spatial_qa.image_root: {os.path.abspath(args.output_dir)}/spatial_qa/images")
    print(f"  temporal_qa.image_root: {os.path.abspath(args.output_dir)}/temporal_qa/images")
    print(f"  temporal_qa.video_root: {os.path.abspath(args.output_dir)}/temporal_qa/videos")


if __name__ == "__main__":
    main()
