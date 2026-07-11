# Spatial QA Test Set Review Report

**Dataset**: Text-Centric VideoQA for Spatial Understanding  
**Current release location**: `data/spatial/VQA_json/` (GT / blur / downsample_x4)
**Review Date**: 2026-05-22  
**Reviewed By**: Automated audit + manual inspection  

> Status note, 2026-07-07: this report was written before the QA-v2 release
> freeze. The current public release has 389 official clip JSON files per method;
> `all_true_false.json` is no longer in the public per-clip annotation folders.

---

## 1. Dataset Summary

| Metric | Value |
|--------|-------|
| Methods (open-source) | GT · blur · downsample_x4 |
| Video clips (per method) | 389 official clip JSON files |
| Total QA pairs (per method) | **14,770** |
| Question types | fill_in_blank · fill_in · multiple_choice · true_false |
| Languages | Chinese (CHN) · English (EN) · Hardcase |
| Image data (NFS, not in repo) | GT: 9.0 GB · blur: 8.7 GB · downsample_x4: 1.1 GB |

**QA type breakdown (GT, 14,770 total):**

| Type | Count | % |
|------|-------|---|
| fill_in_blank | 3,656 | 24.7% |
| fill_in | 2,191 | 14.8% |
| multiple_choice | 5,210 | 35.3% |
| true_false | 3,713 | 25.1% |

---

## 2. Issue Summary

### 🔴 HIGH — fill_in answers are multi-word lists (evaluation policy unclear)

**Finding**: `fill_in` type questions ask "What word(s) are located within [bbox]?" and their
answers are Python lists with 1–N words, e.g.:

```json
{
  "question_type": "fill_in",
  "question": "What is the word(s) located within [[911,617],[1490,617],[1490,932],[911,932]]? It is ________.",
  "answer": ["此处", "温馨提示", "禁止停", "capybara", "注意安"]
}
```

The current evaluation code (and `metrics.py`) resolves multi-valued GT answers by taking only the
**first element** (`gt[0]`), meaning accuracy is measured against a single canonical answer per
question regardless of how many correct answers exist.

**Impact**: Underestimates model accuracy when a model correctly identifies any (but not the first)
word in the region. Conversely, a model that always guesses the most visible word may score higher
than deserved.

**Recommendation**: Before release, decide on evaluation policy:
- **Option A (current)**: `gt[0]` only — simple, but biased toward the first OCR-detected word.
- **Option B (any-match)**: Correct if model answer matches any element in the GT list — more lenient.
- **Option C (set-match)**: Jaccard similarity between predicted word set and GT list — most rigorous.

Document the chosen policy explicitly in the README and `evaluation/spatial/metrics.py`.

---

### 🔴 HIGH — fill_in vs fill_in_blank are structurally different question types

**Finding**: Two question types are conflated under similar names but have fundamentally different
answer formats:

| Type | Question Pattern | Answer Format | Example |
|------|-----------------|---------------|---------|
| `fill_in_blank` | "Text is 'X____', fill in the blank" | Single character/word string | `"程"`, `"at"`, `"口"` |
| `fill_in` | "What word(s) are in this region?" | List of 1–N strings | `["此处", "温馨提示", ...]` |

These require different evaluation logic. The current `metrics.py` maps both to `"fill_in"` and
uses the same comparator, which is incorrect for multi-word list answers.

**Recommendation**: Either (a) keep them split (`fill_in_blank` vs `fill_in`) with separate metrics,
or (b) merge into one type with a unified list-based evaluator. Update metric labels in the paper.

---

### 🟡 MEDIUM — GT has 390 annotation files vs 389 for blur/downsample_x4

**Historical finding**: `GT_VQA_testing/` contained one extra file:
`all_true_false.json` (3,713 entries, flat list aggregating all GT true/false QA
pairs).

**Impact**: This file is a utility aggregation artifact, NOT an additional test clip. However,
if evaluation scripts glob `*.json` from the annotation directory, they will accidentally load
this file and double-count all true/false questions for GT.

**Current status**: Resolved in the 2026-07-07 release freeze and public repo
copy. `all_true_false.json` is preserved only under the internal
`legacy_aggregates/spatial_true_false/` archive and is not part of
`data/spatial/VQA_json/*_VQA_testing/`.

---

### 🟡 MEDIUM — Some video clips have low fill-in-blank accuracy in existing evaluations (~5–12%)

**Finding**: From existing pipeline results:
- fill_in_blank accuracy: 4.7–12.2% across all methods
- much lower than multiple_choice (~25%) and true_false (~50%)

**Impact**: Could indicate that some fill-in-blank blanks are too ambiguous (partial word with
only 1–2 characters visible) or that the question difficulty is appropriate but creates very
sparse correct signals.

**Recommendation**: Spot-check 10–20 hardest fill_in_blank questions (lowest accuracy) to confirm
the blanks are humanly solvable. Consider providing a `difficulty` tag (easy/medium/hard) in the
release metadata.

---

### 🟡 MEDIUM — Multiple choice distractors quality not verified programmatically

**Finding**: Multiple-choice questions have 4 options (A/B/C/D). Distractors were generated
using "visual/semantic/OCR-level similarity" (per paper). Not verified that all 389–390 files
have exactly 4 options and that the correct answer key is always among A–D.

**Recommendation**: Run a validation script before release:

```python
import json, glob
for f in glob.glob("data/spatial/VQA_json/GT_VQA_testing/*.json"):
    data = json.load(open(f))
    for clip in data:
        for qf in clip.get("questions", []):
            for qa in qf.get("qa_pairs", []):
                q = qa.get("question", {})
                if q.get("question_type") == "multiple_choice":
                    assert q.get("answer") in ("A","B","C","D"), f"Bad answer in {f}"
```

---

### 🟢 LOW — Bounding box coordinates in question text use integer precision

**Finding**: Sampled bounding boxes embedded in question strings (e.g., `[[911,617],[1490,617],
[1490,932],[911,932]]`) use integer pixel coordinates. No sub-pixel decimal issue observed in the
GT/blur/downsample_x4 annotation copies.

**Note**: Sub-pixel decimals may exist in the full 6-method source data on NFS but are not present
in the open-source annotation copies.

---

### 🟢 LOW — No empty answers found

All 14,770 GT QA pairs have non-empty answer fields. Same confirmed for blur and downsample_x4.

---

## 3. Evaluation Correctness Verification

Before public release, verify that `evaluation/spatial/metrics.py` produces results matching
the paper's reported numbers on the GT set:

| Metric | Expected (from paper) | Check |
|--------|----------------------|-------|
| Total Accuracy (GT, text-only baseline) | ~23.3% | `python evaluation/spatial/metrics.py --results_dir <qa_output> --dataset GT` |
| fill_in accuracy | ~5–12% | |
| multiple_choice accuracy | ~23–28% | |
| true_false accuracy | ~48–52% | |

---

## 4. Release Checklist

- [ ] Decide and document `fill_in` evaluation policy (any-match vs first-match vs set-match)
- [x] Move `all_true_false.json` out of `GT_VQA_testing/` annotation directory
- [ ] Add validation script output to README (confirm 0 errors)
- [ ] Add `data/spatial_qa/README.md` with: question type definitions, answer format spec, evaluation policy
- [ ] Upload images to HuggingFace: GT (9.0 GB) · blur (8.7 GB) · downsample_x4 (1.1 GB)
- [ ] Verify metrics match paper Table results after uploading
