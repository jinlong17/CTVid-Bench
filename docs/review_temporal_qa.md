# Temporal QA Test Set Review Report

**Dataset**: Text-Centric VideoQA for Temporal Understanding  
**Current release location**: `data/temporal/videos_vqa/<method>/vqa/` (GT / blur / downsample_x4)
**Review Date**: 2026-05-22  
**Reviewed By**: Automated audit + manual inspection

> Status note, 2026-07-07: this report describes the legacy 28-question
> temporal QA set. The current public release uses QA-v2 with 10 questions per
> clip, 97 clip JSONs per method, and `q_natural` populated for compatibility.
> Keep this file as historical review context; do not use its category counts as
> the current release schema.

---

## 1. Dataset Summary

| Metric | Value |
|--------|-------|
| Methods (open-source) | GT · blur · downsample_x4 |
| Video clips (per method) | **97** |
| Questions per clip | **10** in the current QA-v2 release |
| Total QA pairs (per method) | **970** in the current QA-v2 release |
| Video length | 120 frames per clip |
| Language split | CHN: 47 clips · EN: 50 clips |
| Video data (NFS) | GT: 332 MB · blur: 278 MB · downsample_x4: 53 MB |
| Image data (NFS) | GT: 31 GB · blur: 21 GB · downsample_x4: 2.8 GB |

**Legacy question category breakdown (per method, 2,716 total):**

| Category | Q-types | Count | % |
|----------|---------|-------|---|
| presence | q1–q6 | 582 | 21.4% |
| localization | q1–q6 | 582 | 21.4% |
| motion | q1–q9 | 873 | 32.1% |
| size | q1–q6 | 582 | 21.4% |
| boundary | q1 | 97 | 3.6% |

---

## 2. Issue Summary

### 🔴 HIGH — motion_q5 is a multi-select question requiring special evaluation

**Finding**: `motion_q5` asks about trajectory type and expects a **list of answers**, e.g.:

```json
{
  "question_type": "motion_q5",
  "q": "What type of trajectory does X follow? (Select all that apply)\nOptions:\nstraight line\ncurve\ncircular\nzigzag",
  "a": ["straight line", "zigzag"]
}
```

Confirmed across all 97 clips. This is the only multi-select question in the dataset.

**Impact**: Standard accuracy (exact string match) will always score 0 for motion_q5 because the
model output is unlikely to match the list representation exactly. The metric script must handle
this case specially.

**Current fix in `evaluation/temporal/metrics.py`**: Uses Jaccard similarity ≥ 0.5:

```python
MULTISELECT_QTYPES = {"motion_q5"}
# Jaccard(pred_set, gt_set) ≥ 0.5 → correct
```

**Recommendation**: Confirm Jaccard ≥ 0.5 is the intended policy and document it in the paper's
evaluation section. Consider adding an ablation with stricter thresholds (0.67, 1.0).

---

### 🔴 HIGH — Numeric frame-index answers require ±2 tolerance (not yet standard)

**Finding**: The following question types return frame indices (0–119) or counts as integer answers:

| Q-type | Semantics | Example Answer |
|--------|-----------|----------------|
| presence_q2 | Number of visible frames | `120` |
| presence_q5 | First appearance frame | `0` |
| presence_q6 | Last appearance frame | `119` |
| motion_q2 | Frame of max speed | `118` |
| motion_q3 | Frame of max acceleration | `119` |
| motion_q8 | Frame of trajectory deviation | `118` |
| size_q1 | Frame with largest area | varies |
| size_q2 | Frame with smallest area | varies |

The paper states ±2 frame tolerance for numeric answers. This is now implemented in
`evaluation/temporal/metrics.py` via `FRAME_TOLERANCE = 2`.

**Impact without tolerance**: Any off-by-one or off-by-two prediction is counted as wrong, which
severely underestimates model capability for inherently imprecise temporal tasks.

**Recommendation**: Verify `FRAME_TOLERANCE = 2` is correct per paper specification. Consider
also reporting strict accuracy (tolerance=0) alongside tolerant accuracy for completeness.

---

### 🔴 HIGH — Small test set (97 clips) limits statistical reliability

**Finding**: 97 unique video clips per method. With 2,716 QA pairs total:
- Expected ±95% CI for accuracy of ~40%: approximately ±3.0 percentage points (per-question level)
- But inter-clip correlation means effective N is closer to 97, giving CI ≈ ±5 pp

**Impact**: Small differences (< 2–3 pp) between models may not be statistically significant.
Performance differences between CHN (47) and EN (50) subsets have even wider CI.

**Recommendation**:
1. Report 95% confidence intervals in the paper (bootstrap recommended)
2. Note the dataset size limitation in the paper and discuss plans for expansion
3. For camera-ready: consider if additional test clips can be added

---

### 🟡 MEDIUM — boundary_q1 is the only boundary question (1.67% of total)

**Finding**: There is exactly **1** boundary question type (`boundary_q1`: "Does X touch the
screen edge?") × 97 clips = 97 QA pairs. This is a Yes/No question.

**Impact**: The "boundary" category accuracy is based on only 97 binary questions, with 50%
random baseline. A model that always says "No" achieves ~50–60% depending on the true class
distribution.

**Recommendation**: Either (a) expand boundary questions (e.g., which edge? top/bottom/left/right?
distance to edge?), or (b) merge boundary into localization category for reporting. If kept as-is,
note the caveat in the paper.

---

### 🟡 MEDIUM — localization_q4 mixes categorical and numeric answer formats

**Finding**: `localization_q4` asks "How many frames does X cross the horizontal center line?"
but the answer is stored as a string like `"0 frames"`, `"4 frames"` — not a pure integer.

Meanwhile, `localization_q6` (how many 3×3 grid regions) stores an integer answer like `2`.

**Impact**: Inconsistent answer format for similar count questions. The evaluator must handle
both `"0 frames"` (string) and `2` (int) for semantically similar questions.

**Recommendation**: Standardize: either store all count answers as integers (drop "frames" suffix)
or all as strings. Update the evaluation metric accordingly.

---

### 🟡 MEDIUM — CHN/EN language imbalance (47 vs 50)

**Finding**: 47 Chinese clips vs 50 English clips. Minor imbalance but worth noting.

**Impact**: Cross-language analysis has unequal group sizes. Chinese text typically has more
complex characters that are harder to detect/track under degradation.

**Recommendation**: Report per-language accuracy breakdown in the paper. Note imbalance as a
dataset limitation; plan 50/50 split for future dataset versions.

---

### 🟢 LOW — presence_q2 is open-ended numeric (0–120)

**Finding**: `presence_q2` asks for the exact number of frames the text is visible (free-form
integer, not multiple choice). With 120 possible values, random baseline is essentially 0%.

**Impact**: Even with ±2 tolerance, this is a hard question. Models that predict "all 120" or
"always 0" will score poorly on varied clips.

**Recommendation**: Consider converting to multiple-choice buckets (e.g., 0–30 / 31–60 / 61–90 /
91–120) for cleaner evaluation. If kept as-is, document the evaluation granularity.

---

### 🟢 LOW — Video paths in JSON point to internal NFS paths

**Finding**: The `video_path` field in each annotation JSON points to an internal NFS path:

```json
"video_path": "/nfs.auto/flash_VideoAlg/project/SR/VSR_text/video_test_set/GT/videos/CHN_C0011.mp4"
```

**Impact**: Public users will not have access to `/nfs.auto/...`. The evaluation script must
override `video_path` using the configured `video_root` from `configs/default.yaml`.

**Status**: Resolved for the current public QA-v2 release. Released JSONs use
relative paths such as `temporal/videos_vqa/GT/videos/CHN_C0011.mp4`, and
`evaluation/temporal/run_eval.py` still resolves media via `configs/default.yaml`.

**Recommendation**: Before release, strip `video_path` from the released JSONs or replace with
a relative placeholder like `"videos/CHN_C0011.mp4"`.

---

## 3. Question Type Reference Card

| Q-type | Answer Format | Eval Method | Notes |
|--------|--------------|-------------|-------|
| presence_q1 | Yes/No | exact match | |
| presence_q2 | int (0–120) | ±2 tolerance | open-ended count |
| presence_q3 | MC string | exact match | "0 times" / "1 time" / … |
| presence_q4 | categorical | exact match | "always present" / … |
| presence_q5 | int (frame) | ±2 tolerance | first appearance |
| presence_q6 | int (frame) | ±2 tolerance | last appearance |
| localization_q1 | categorical | exact match | 5 regions |
| localization_q2 | categorical | exact match | 5 regions |
| localization_q3 | categorical | exact match | 6 options incl. "no dominant" |
| localization_q4 | string ("N frames") | exact match | ⚠ inconsistent format |
| localization_q5 | Yes/No | exact match | |
| localization_q6 | int (1–9) | exact match | 3×3 grid count |
| motion_q1 | categorical | exact match | 8-way direction |
| motion_q2 | int (frame) | ±2 tolerance | max speed frame |
| motion_q3 | int (frame) | ±2 tolerance | max accel frame |
| motion_q4 | categorical | exact match | start/middle/end |
| motion_q5 | list | Jaccard ≥ 0.5 | ⚠ multi-select |
| motion_q6 | categorical | exact match | sub-sequence direction |
| motion_q7 | categorical | exact match | sub-sequence pattern |
| motion_q8 | int (frame) | ±2 tolerance | deviation frame |
| motion_q9 | MC string | exact match | speed bucket |
| size_q1 | int (frame) | ±2 tolerance | largest area frame |
| size_q2 | int (frame) | ±2 tolerance | smallest area frame |
| size_q3 | categorical | exact match | approach/recede |
| size_q4 | categorical | exact match | trend |
| size_q5 | Yes/No | exact match | width threshold |
| size_q6 | Yes/No | exact match | width threshold |
| boundary_q1 | Yes/No | exact match | ⚠ only 97 samples |

---

## 4. Release Checklist

- [ ] Confirm Jaccard ≥ 0.5 policy for motion_q5 with paper authors
- [ ] Strip or relativize `video_path` in released JSON files
- [ ] Standardize localization_q4 answer format (string vs int)
- [ ] Add per-language (CHN/EN) accuracy breakdown to evaluation output
- [ ] Document ±2 frame tolerance in paper evaluation section and README
- [ ] Report 95% confidence intervals for key metrics in the paper
- [ ] Consider expanding boundary questions or merging into localization
- [ ] Upload videos to HuggingFace: GT (332 MB) · blur (278 MB) · downsample_x4 (53 MB)
- [ ] Upload images to HuggingFace: GT (31 GB) · blur (21 GB) · downsample_x4 (2.8 GB)
