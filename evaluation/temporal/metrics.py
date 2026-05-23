#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Temporal QA metrics.

Question-type-aware accuracy:
  - Multiple-choice / Yes-No / categorical  →  exact match
  - Numeric (frame indices, counts)          →  exact match OR within ±FRAME_TOLERANCE
  - Multi-select (motion_q5 trajectory)      →  Jaccard similarity ≥ threshold

Usage:
  python evaluation/temporal/metrics.py --results_dir outputs/temporal/ --dataset GT
"""

import os
import json
import argparse
import re
from pathlib import Path

FRAME_TOLERANCE = 2        # ±2 frames for numeric frame-index answers
JACCARD_THRESHOLD = 0.5    # For multi-select questions (motion_q5)

# Question types that are numeric (frame index or count)
NUMERIC_QTYPES = {
    "presence_q2",   # frame count (0–120)
    "presence_q5",   # first appearance frame
    "presence_q6",   # last appearance frame
    "motion_q2",     # frame of max speed
    "motion_q3",     # frame of max acceleration
    "motion_q8",     # frame of trajectory deviation
    "size_q1",       # frame with largest area
    "size_q2",       # frame with smallest area
    "localization_q6",  # number of 3×3 regions (1–9)
}

# Question types that allow multiple correct answers (multi-select)
MULTISELECT_QTYPES = {"motion_q5"}

# All other types use exact string match


def normalize_answer(ans) -> str:
    if isinstance(ans, list):
        return str(ans[0]).strip().lower() if ans else ""
    return str(ans).strip().lower()


def parse_numeric(text) -> float | None:
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    return float(m.group()) if m else None


def parse_multiselect(ans) -> set:
    """Parse a list or comma-separated string into a set of normalised options."""
    if isinstance(ans, list):
        return {str(x).strip().lower() for x in ans}
    if isinstance(ans, str):
        return {x.strip().lower() for x in re.split(r"[,;/]", ans) if x.strip()}
    return {str(ans).strip().lower()}


def is_correct_by_type(gt, pred, qtype: str) -> bool:
    if qtype in NUMERIC_QTYPES:
        gt_num = parse_numeric(gt)
        pred_num = parse_numeric(pred)
        if gt_num is None or pred_num is None:
            return False
        return abs(gt_num - pred_num) <= FRAME_TOLERANCE

    if qtype in MULTISELECT_QTYPES:
        gt_set = parse_multiselect(gt)
        pred_set = parse_multiselect(pred)
        if not gt_set and not pred_set:
            return True
        intersection = len(gt_set & pred_set)
        union = len(gt_set | pred_set)
        return (intersection / union) >= JACCARD_THRESHOLD if union > 0 else False

    # Default: exact normalised string match
    return normalize_answer(gt) == normalize_answer(pred)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

CATEGORIES = ["presence", "localization", "motion", "size", "boundary"]


def question_category(qtype: str) -> str:
    for cat in CATEGORIES:
        if qtype.startswith(cat):
            return cat
    return "other"


def compute_metrics(all_results: list, model_name: str) -> dict:
    answer_key = f"model_answer_{model_name}"

    cat_stats = {c: {"total": 0, "correct": 0} for c in CATEGORIES + ["other"]}
    total = {"total": 0, "correct": 0}

    for item in all_results:
        for q in item.get("questions", []):
            qtype = q.get("question_type", "")
            gt = q.get("a")
            pred = q.get(answer_key)
            if pred is None:
                continue

            correct = is_correct_by_type(gt, pred, qtype)
            cat = question_category(qtype)
            cat_stats[cat]["total"] += 1
            total["total"] += 1
            if correct:
                cat_stats[cat]["correct"] += 1
                total["correct"] += 1

    result = {}
    for cat in CATEGORIES + ["other"]:
        s = cat_stats[cat]
        n = s["total"]
        result[cat] = {
            "total_samples": n,
            "correct_predictions": s["correct"],
            "accuracy": round(s["correct"] / n, 4) if n > 0 else 0.0,
        }

    n = total["total"]
    result["total"] = {
        "total_samples": n,
        "correct_predictions": total["correct"],
        "accuracy": round(total["correct"] / n, 4) if n > 0 else 0.0,
    }
    return {"model": model_name, "metrics": result}


def process_dataset(results_dir: str, dataset: str, model_name: str) -> dict | None:
    ds_dir = os.path.join(results_dir, dataset)
    if not os.path.isdir(ds_dir):
        print(f"[warn] Not found: {ds_dir}")
        return None

    all_data = []
    for jf in sorted(Path(ds_dir).glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            all_data.append(d) if isinstance(d, dict) else all_data.extend(d)
        except Exception as exc:
            print(f"  [error] {jf.name}: {exc}")

    if not all_data:
        return None

    metrics = compute_metrics(all_data, model_name)
    acc = metrics["metrics"]["total"]["accuracy"]
    print(f"[{dataset}] samples={metrics['metrics']['total']['total_samples']}  accuracy={acc:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Temporal QA metrics.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model_name", default="mllm_baseline")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, "..", "metrics")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["GT", "blur", "downsample_x4"]
    summary = {}

    for ds in datasets:
        result = process_dataset(args.results_dir, ds, args.model_name)
        if result is None:
            continue
        out_path = os.path.join(out_dir, f"{ds}_temporal_metrics.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        summary[ds] = result["metrics"]["total"]

    summary_path = os.path.join(out_dir, "temporal_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model_name, "datasets": summary}, fh, indent=2)

    if summary:
        print(f"\n{'Dataset':<20} {'Total':>8} {'Accuracy':>10}")
        print("-" * 42)
        for ds, m in summary.items():
            print(f"{ds:<20} {m['total_samples']:>8} {m['accuracy']:>10.4f}")
    print(f"\nMetrics saved to {out_dir}/")


if __name__ == "__main__":
    main()
