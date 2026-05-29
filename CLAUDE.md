# CLAUDE.md

Behavioral guidelines for Claude when working in **CTVid-Bench** — the public open-source
benchmark release.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## Part A — Generic Guardrails

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs. If uncertain, ask. If multiple interpretations exist, present them. If a simpler approach exists, say so.

### 2. Simplicity First
Minimum code that solves the problem. No features beyond what was asked. No abstractions for single-use code. No error handling for impossible scenarios. If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes
Touch only what you must. Don't "improve" adjacent code. Match existing style. Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Transform tasks into verifiable goals with explicit success criteria. Loop until verified.

---

## Part B — Project Orientation

**Purpose:** Public open-source release of the ClearText-Video (CTVid) benchmark — CVPR 2026.
Evaluation scripts + annotation JSONs + project page + HuggingFace data link.

**Sister repo:** [CTVid-Research](https://github.com/jinlong17/CTVid-Research) — private full-pipeline codebase with all 6 methods and training code.

**This repo is for external users.** Every design decision should serve someone who clones from GitHub on a fresh machine and has no NFS access.

---

## Part C — Open-Source Scope (Strict)

Only **3 methods** are released here:

| Method | Why included |
|---|---|
| `GT` | High-quality ground truth |
| `blur` | Degraded-quality (motion blur) baseline |
| `downsample_x4` | Degraded-quality (4× downsampling) baseline |

**Do NOT add** `DOVE`, `MIMO-UNetPlus`, `VSR_S3Diff` to this repo — those belong to CTVid-Research only.

If a user asks "why only 3 methods?" the answer is: licensing / paper scope. Other methods are gated to the research repo.

---

## Part D — Data Layout

### What ships in the repo
- `data/spatial_qa/annotations/` — JSON QA pairs for 3 methods (~33 MB, committed)
- `data/temporal_qa/annotations/` — JSON QA pairs for 3 methods (~2.4 MB, committed)
- `data/spatial_qa/images/` and `data/temporal_qa/{images,videos_vqa}/` — **symlinks** into NFS (~74 GB of actual data, not in repo)

### Where images/videos live
On NFS machines: symlinks resolve to `/nfs/.../CTVid-Bench-Research/open_source/`.
On other machines: external users run `python tools/download_data.py` to pull from HuggingFace into `data/`.

### HuggingFace dataset
`<your-hf-org>/CTVid-Bench` (upload deferred — see `DATA.md`). Total open-source payload: ~74.3 GB.
- Spatial images: 18.8 GB (GT 9.0 + blur 8.7 + downsample_x4 1.1)
- Temporal images: 54.8 GB (GT 31 + blur 21 + downsample_x4 2.8)
- Temporal videos: 663 MB (GT 332 + blur 278 + downsample_x4 53)

Anything larger than annotations stays on HuggingFace — do not commit binaries.

---

## Part E — Don't-Touch / Don't-Ship List

| Path or item | Why |
|---|---|
| Model weights (`*.bin`, `*.safetensors`, `*.pt`, `*.pth`) | Never commit. Users download via `huggingface_hub` at runtime. |
| `outputs/`, `results/`, `logs/` | Gitignored. Never commit. |
| `data/spatial_qa/images/`, `data/temporal_qa/{images,videos_vqa}/` contents | Symlinks only — never replace with copies or commit the resolved bytes. |
| Internal NFS paths in JSONs (`video_path: /nfs.auto/...`) | Strip or override at eval time. See `evaluation/temporal/run_eval.py:find_video_path()`. |
| Hardcoded `/nfs/...` paths anywhere in Python | Use `configs/default.yaml` instead. The whole point of this repo is reproducibility off-NFS. |

---

## Part F — Evaluation Conventions

### Two tasks, paired metric module each

```
evaluation/spatial/run_eval.py    →  metrics.py   # Acc · UAcc · OC, list-answer policy (gt[0])
evaluation/temporal/run_eval.py   →  metrics.py   # ±2 frame tolerance, Jaccard ≥ 0.5 for motion_q5
```

### Naming gotchas

- Spatial annotation folders: **UPPERCASE** — `GT_VQA_testing/`, `blur_VQA_testing/`, `downsample_x4_VQA_testing/`
- Temporal annotation folders: **lowercase** — `GT_vqa_testing/`, `blur_vqa_testing/`, `downsample_x4_vqa_testing/`
- The dataset → folder mapping lives in `configs/default.yaml`. Always read it; never assume.

### Multi-select and tolerance (already implemented — don't break)

- `motion_q5` (trajectory type) is the **only** multi-select question. Jaccard ≥ 0.5.
- Numeric frame-index answers (presence_q2/q5/q6, motion_q2/q3/q8, size_q1/q2, localization_q6) use **±2 frame tolerance**.
- See `docs/review_temporal_qa.md` and `docs/review_spatial_qa.md` for known QA dataset caveats.

---

## Part G — Common Commands

```bash
# 1. Download data from HuggingFace (external users)
python tools/download_data.py --variants GT blur downsample_x4

# 2. Verify data access (NFS users — should list 3 methods)
ls data/spatial_qa/images/
ls data/temporal_qa/images/

# 3. Run spatial QA evaluation
python evaluation/spatial/run_eval.py --dataset GT --config configs/default.yaml
python evaluation/spatial/metrics.py --results_dir outputs/spatial/ --dataset GT

# 4. Run temporal QA evaluation
python evaluation/temporal/run_eval.py --dataset GT --config configs/default.yaml
python evaluation/temporal/metrics.py --results_dir outputs/temporal/ --dataset GT
```

---

**These guidelines are working if:** the repo stays clone-and-run on a fresh non-NFS machine, no binaries leak into git, and the 3-method scope holds firm.
