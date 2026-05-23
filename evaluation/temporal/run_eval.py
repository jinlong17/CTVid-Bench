#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Temporal QA evaluation — vision-language model baseline.

Loads each video clip, presents each question to a VLM, and records answers.
The output JSON format matches the input annotations with model answers added.

Usage:
  python evaluation/temporal/run_eval.py --dataset GT --config configs/default.yaml
  python evaluation/temporal/run_eval.py --dataset blur --model Qwen/Qwen2-VL-7B-Instruct
"""

import os
import sys
import json
import glob
import argparse

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def is_fully_annotated(data: dict, model_name: str) -> bool:
    key = f"model_answer_{model_name}"
    for q in data.get("questions", []):
        if key not in q:
            return False
    return True


def find_video_path(cfg: dict, dataset: str, clip_filename: str) -> str | None:
    """Resolve video path from config video_root."""
    video_root = cfg["temporal_qa"].get("video_root", "")
    if not video_root:
        return None
    lang = "CHN" if "CHN" in clip_filename else "EN"
    clip_id = os.path.splitext(clip_filename.replace(f"{lang}_", ""))[0]
    candidate = os.path.join(video_root, dataset, "videos", f"{lang}_{clip_id}.mp4")
    return candidate if os.path.exists(candidate) else None


# ---------------------------------------------------------------------------
# Model inference (placeholder — swap in your VLM)
# ---------------------------------------------------------------------------

def load_model(model_name_or_path: str, device_id: int):
    """Load VLM. Returns (model, processor) tuple.

    Replace this function with your own model loading logic.
    Currently returns (None, None) as a placeholder for the evaluation harness.
    """
    print(f"[info] Model loading stub for '{model_name_or_path}' on cuda:{device_id}.")
    print("[info] Replace load_model() and answer_question() with your VLM implementation.")
    return None, None


def answer_question(model, processor, video_path: str | None, question_text: str, choices: list) -> str:
    """Query the VLM with a question about the video.

    Parameters
    ----------
    model, processor : VLM model/processor (from load_model)
    video_path       : Path to the video file (may be None if not found)
    question_text    : The question string
    choices          : List of answer choices (empty for open-ended questions)

    Returns the model's answer string.
    """
    # Placeholder: return empty string — replace with actual VLM inference
    return ""


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(model, processor, dataset: str, cfg: dict, output_dir: str, model_name: str, incremental: bool) -> None:
    temporal_cfg = cfg["temporal_qa"]
    folder = temporal_cfg["dataset_to_annotation_folder"].get(dataset)
    if not folder:
        print(f"[error] No annotation folder for '{dataset}'.")
        return

    ann_dir = os.path.join(temporal_cfg["annotation_dir"], folder)
    if not os.path.isdir(ann_dir):
        print(f"[warn] Annotation dir not found: {ann_dir}")
        return

    ds_out_dir = os.path.join(output_dir, dataset)
    os.makedirs(ds_out_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    print(f"\n[{dataset}] {len(json_files)} annotation files.")

    answer_key = f"model_answer_{model_name}"

    for json_path in json_files:
        filename = os.path.basename(json_path)
        out_path = os.path.join(ds_out_dir, filename)

        if incremental and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if is_fully_annotated(existing, model_name):
                    print(f"  [skip] {filename}")
                    continue
            except Exception:
                pass

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        clip_name = os.path.splitext(filename)[0].replace("_video_questions", "")
        video_path = find_video_path(cfg, dataset, clip_name + ".mp4")

        for q in data.get("questions", []):
            question_text = q.get("q", "")
            # Extract choices from question text (lines starting with valid options)
            choices = [line.strip() for line in question_text.splitlines()
                       if line.strip() and not line.strip().startswith('"') and line.strip()[0].isalpha() and len(line.strip()) < 60][1:]

            model_answer = answer_question(model, processor, video_path, question_text, choices)
            q[answer_key] = model_answer

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"  [saved] {filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Temporal QA VLM evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="GT / blur / downsample_x4 (default: all)")
    parser.add_argument("--output_dir", default="outputs/temporal/qa_results")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct", help="VLM model name or path")
    parser.add_argument("--model_name", default="qwen2vl_7b", help="Tag written into output JSON keys")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--no_incremental", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, processor = load_model(args.model, args.device_id)

    datasets = [args.dataset] if args.dataset else cfg["temporal_qa"]["datasets"]
    for ds in datasets:
        run_dataset(model, processor, ds, cfg, args.output_dir, args.model_name, not args.no_incremental)

    print("\nTemporal QA evaluation complete.")
    print(f"Results in: {args.output_dir}")
    print(f"Run metrics: python evaluation/temporal/metrics.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
