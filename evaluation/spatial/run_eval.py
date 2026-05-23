#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spatial QA evaluation — text-only baseline using Qwen2.5-7B-Instruct.

For each clip in the dataset:
  1. Load the VQA annotation JSON.
  2. For each frame, feed OCR-detected text to the LLM and record its answers.
  3. Write annotated JSONs to output_dir.

Usage:
  python evaluation/spatial/run_eval.py --dataset GT --config configs/default.yaml
  python evaluation/spatial/run_eval.py --dataset blur --device_id 1
  python evaluation/spatial/run_eval.py  # all 3 datasets
"""

import os
import sys
import json
import glob
import argparse

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make evaluation utilities importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_utils import build_ocr_context, build_system_prompt, build_user_prompt, extract_json_answer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QwenTextOnlyEvaluator:
    def __init__(self, model_path: str, device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Loading model from {model_path} on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model ready.")

    def _generate(self, messages: list, max_new_tokens: int = 256) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True)

    def evaluate_frame(self, ocr_results: list, qa_pairs: list) -> list:
        ocr_context = build_ocr_context(ocr_results)
        results = []
        for qa_item in qa_pairs:
            q = qa_item.get("question", {})
            qtype = q.get("question_type", "")
            qtext = q.get("question", "")

            messages = [
                {"role": "system", "content": build_system_prompt(qtype)},
                {"role": "user",   "content": build_user_prompt(qtype, qtext, ocr_context)},
            ]
            raw = self._generate(messages)
            parsed = extract_json_answer(raw)
            model_answer = parsed.get("answer", "__FAILED")

            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Are you sure you accurately answered the question?\n\nAnswer format: Yes or No"},
            ]
            raw_cert = self._generate(messages, max_new_tokens=16)
            cert_line = raw_cert.strip().splitlines()[0].strip()
            model_certainty = cert_line.capitalize() if cert_line.lower() in ("yes", "no") else cert_line

            results.append({"question_type": qtype, "model_answer": model_answer, "model_certainty": model_certainty})
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def load_ocr_results(ocr_output_dir: str, dataset: str, category: str, clip_id: str) -> dict:
    """Load OCR JSON produced by a separate OCR stage (optional)."""
    path = os.path.join(ocr_output_dir, dataset, category, f"{clip_id}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def parse_clip_filename(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[0], parts[-1]
    return "unknown", stem


def is_fully_annotated(clip_data: list, model_name: str) -> bool:
    key = f"acc_uacc_model_answer_{model_name}"
    for clip in clip_data:
        for qframe in clip.get("questions", []):
            for qa in qframe.get("qa_pairs", []):
                if key not in qa.get("question", {}):
                    return False
    return True


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(evaluator, dataset: str, cfg: dict, output_dir: str, model_name: str, incremental: bool) -> None:
    spatial_cfg = cfg["spatial_qa"]
    folder = spatial_cfg["dataset_to_annotation_folder"].get(dataset)
    if not folder:
        print(f"[error] No annotation folder mapping for '{dataset}'.")
        return

    ann_dir = os.path.join(spatial_cfg["annotation_dir"], folder)
    if not os.path.isdir(ann_dir):
        print(f"[warn] Annotation dir not found: {ann_dir}")
        return

    ds_out_dir = os.path.join(output_dir, dataset)
    os.makedirs(ds_out_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    print(f"\n[{dataset}] {len(json_files)} annotation files.")

    answer_key = f"acc_uacc_model_answer_{model_name}"
    certainty_key = f"model_certainty_{model_name}"

    for json_path in json_files:
        clip_filename = os.path.basename(json_path)
        out_path = os.path.join(ds_out_dir, clip_filename)

        if incremental and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if is_fully_annotated(existing, model_name):
                    print(f"  [skip] {clip_filename}")
                    continue
            except Exception:
                pass

        with open(json_path, "r", encoding="utf-8") as fh:
            clip_data = json.load(fh)

        # Annotate each frame
        for clip_entry in clip_data:
            for qframe in clip_entry.get("questions", []):
                frame_name = qframe.get("frame", "")
                # OCR context: use pre-computed OCR if available
                ocr_results = []  # extend: load from ocr_output_dir if using Stage 1

                qa_pairs = qframe.get("qa_pairs", [])
                if not qa_pairs:
                    continue
                frame_results = evaluator.evaluate_frame(ocr_results, qa_pairs)

                for qa_item, res in zip(qa_pairs, frame_results):
                    q = qa_item.setdefault("question", {})
                    q[answer_key] = res["model_answer"]
                    q[certainty_key] = res["model_certainty"]

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(clip_data, fh, ensure_ascii=False, indent=2)
        print(f"  [saved] {clip_filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spatial QA text-only evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="GT / blur / downsample_x4 (default: all)")
    parser.add_argument("--output_dir", default="outputs/spatial/qa_results")
    parser.add_argument("--model_name", default="qwen_textonly")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--no_incremental", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = cfg["model"].get("local_path") or cfg["model"]["name"]
    evaluator = QwenTextOnlyEvaluator(model_path=model_path, device_id=args.device_id)

    datasets = [args.dataset] if args.dataset else cfg["spatial_qa"]["datasets"]
    for ds in datasets:
        run_dataset(evaluator, ds, cfg, args.output_dir, args.model_name, not args.no_incremental)

    print("\nSpatial QA evaluation complete.")
    print(f"Results in: {args.output_dir}")
    print(f"Run metrics: python evaluation/spatial/metrics.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
