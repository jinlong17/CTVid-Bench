#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spatial QA metrics: Accuracy, UAcc (Uncertainty-aware Accuracy), OC (Overconfidence Ratio).

Usage:
  python evaluation/spatial/metrics.py --results_dir outputs/spatial/ --dataset GT
  python evaluation/spatial/metrics.py --results_dir outputs/spatial/  # all datasets
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path


QUESTION_TYPES = ["fill_in", "true_false", "multiple_choice"]


def normalize_text(text: str) -> str:
    return re.sub(r'[\s"""''.!?():（）【】\'`]+', "", text.strip().lower())


def map_question_type(raw_type: str) -> str:
    return "fill_in" if raw_type == "fill_in_blank" else raw_type


def is_correct(gt, pred, qtype: str) -> bool:
    if isinstance(gt, list):
        gt = gt[0] if gt else ""
    if isinstance(pred, list):
        pred = pred[0] if pred else ""
    if not gt or not pred:
        return False
    if qtype == "fill_in":
        return normalize_text(str(gt)) == normalize_text(str(pred))
    elif qtype == "multiple_choice":
        return normalize_text(str(gt)).strip(".") == normalize_text(str(pred)).strip(".")
    elif qtype == "true_false":
        return str(gt).strip().lower() == str(pred).strip().lower()
    return False


def compute_metrics(all_data: list, model_name: str) -> dict:
    suffix_pred = f"acc_uacc_model_answer_{model_name}"
    suffix_cert = f"model_certainty_{model_name}"

    stats = {qt: {"total": 0, "correct": 0, "uacc": 0, "oc": 0} for qt in QUESTION_TYPES}
    total = {"total": 0, "correct": 0, "uacc": 0, "oc": 0}

    for clip in all_data:
        for qframe in clip.get("questions", []):
            for qa in qframe.get("qa_pairs", []):
                q = qa.get("question", {})
                qtype = map_question_type(q.get("question_type", ""))
                if qtype not in QUESTION_TYPES or suffix_pred not in q:
                    continue

                gt = q.get("answer", "")
                pred = q.get(suffix_pred, "")
                cert_raw = q.get(suffix_cert, "")
                if isinstance(cert_raw, list):
                    cert_raw = cert_raw[0] if cert_raw else ""
                cert = str(cert_raw).strip().lower()

                correct = is_correct(gt, pred, qtype)
                is_confident = cert == "yes"
                is_not_confident = cert == "no"

                stats[qtype]["total"] += 1
                total["total"] += 1
                if correct:
                    stats[qtype]["correct"] += 1
                    total["correct"] += 1
                if (correct and is_confident) or (not correct and is_not_confident):
                    stats[qtype]["uacc"] += 1
                    total["uacc"] += 1
                if not correct and is_confident:
                    stats[qtype]["oc"] += 1
                    total["oc"] += 1

    result = {}
    for qt in QUESTION_TYPES:
        s = stats[qt]
        n = s["total"]
        result[qt] = {
            "total_samples": n,
            "correct_predictions": s["correct"],
            "accuracy": round(s["correct"] / n, 4) if n > 0 else 0.0,
            "uacc_score": round(s["uacc"] / n, 4) if n > 0 else 0.0,
            "overconfidence_ratio": round(s["oc"] / n, 4) if n > 0 else 0.0,
        }
    n = total["total"]
    result["total"] = {
        "total_samples": n,
        "correct_predictions": total["correct"],
        "accuracy": round(total["correct"] / n, 4) if n > 0 else 0.0,
        "uacc_score": round(total["uacc"] / n, 4) if n > 0 else 0.0,
        "overconfidence_ratio": round(total["oc"] / n, 4) if n > 0 else 0.0,
    }
    return {"model": model_name, "metrics": result}


def process_dataset(results_dir: str, dataset: str, model_name: str) -> dict | None:
    qa_dir = os.path.join(results_dir, dataset)
    if not os.path.isdir(qa_dir):
        print(f"[warn] Results dir not found: {qa_dir}")
        return None

    all_data = []
    for jf in sorted(Path(qa_dir).glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            all_data.extend(data if isinstance(data, list) else [data])
        except Exception as exc:
            print(f"  [error] {jf.name}: {exc}")

    if not all_data:
        return None

    metrics = compute_metrics(all_data, model_name)
    acc = metrics["metrics"]["total"]["accuracy"]
    print(f"[{dataset}] samples={metrics['metrics']['total']['total_samples']}  accuracy={acc:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Spatial QA metrics (Acc / UAcc / OC).")
    parser.add_argument("--results_dir", required=True, help="Directory with per-dataset QA result JSONs.")
    parser.add_argument("--dataset", default=None, help="Single dataset (GT/blur/downsample_x4). Default: all.")
    parser.add_argument("--model_name", default="qwen_textonly", help="Model name tag in JSON keys.")
    parser.add_argument("--output_dir", default=None, help="Where to save metric JSONs (default: results_dir/../metrics/).")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, "..", "metrics")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["GT", "blur", "downsample_x4"]
    summary = {}

    for ds in datasets:
        result = process_dataset(args.results_dir, ds, args.model_name)
        if result is None:
            continue
        out_path = os.path.join(out_dir, f"{ds}_spatial_metrics.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        summary[ds] = result["metrics"]["total"]

    summary_path = os.path.join(out_dir, "spatial_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model_name, "datasets": summary}, fh, indent=2)

    if summary:
        print(f"\n{'Dataset':<20} {'Total':>8} {'Accuracy':>10} {'UAcc':>10} {'OC':>8}")
        print("-" * 60)
        for ds, m in summary.items():
            print(f"{ds:<20} {m['total_samples']:>8} {m['accuracy']:>10.4f} {m['uacc_score']:>10.4f} {m['overconfidence_ratio']:>8.4f}")
    print(f"\nMetrics saved to {out_dir}/")


if __name__ == "__main__":
    main()
