# CTVid-Bench — Data Download Instructions

## What's in This Repo

Annotation JSON files are included directly:

| Path | Size | Contents |
|------|------|----------|
| `data/spatial_qa/annotations/GT_VQA_testing/` | 14 MB | 390 JSON files |
| `data/spatial_qa/annotations/blur_VQA_testing/` | 9.3 MB | 389 JSON files |
| `data/spatial_qa/annotations/downsample_x4_VQA_testing/` | 9.3 MB | 389 JSON files |
| `data/temporal_qa/annotations/GT_vqa_testing/` | ~0.8 MB | 97 JSON files |
| `data/temporal_qa/annotations/blur_vqa_testing/` | ~0.8 MB | 97 JSON files |
| `data/temporal_qa/annotations/downsample_x4_vqa_testing/` | ~0.8 MB | 97 JSON files |

## Images & Videos (HuggingFace)

Large media files are hosted on HuggingFace. Total download sizes:

| Variant | Spatial Images | Temporal Images | Temporal Videos |
|---------|---------------|-----------------|-----------------|
| GT | 9.0 GB | 31 GB | 332 MB |
| blur | 8.7 GB | 21 GB | 278 MB |
| downsample_x4 | 1.1 GB | 2.8 GB | 53 MB |
| **Total** | **18.8 GB** | **54.8 GB** | **663 MB** |

### Option A — Automatic download

```bash
python tools/download_data.py \
    --split test \
    --variants GT blur downsample_x4 \
    --output_dir ./data
```

### Option B — Manual download with huggingface-cli

```bash
pip install huggingface_hub

# Spatial QA images
huggingface-cli download CTVid/CTVid-Bench \
    --repo-type dataset \
    --include "spatial_qa/images/*" \
    --local-dir ./data/spatial_qa/

# Temporal QA images
huggingface-cli download CTVid/CTVid-Bench \
    --repo-type dataset \
    --include "temporal_qa/images/*" \
    --local-dir ./data/temporal_qa/

# Temporal QA videos (smaller)
huggingface-cli download CTVid/CTVid-Bench \
    --repo-type dataset \
    --include "temporal_qa/videos/*" \
    --local-dir ./data/temporal_qa/
```

### Option C — Python datasets library

```python
from datasets import load_dataset

# Spatial QA
ds = load_dataset("CTVid/CTVid-Bench", name="spatial_qa", split="test")

# Temporal QA
ds = load_dataset("CTVid/CTVid-Bench", name="temporal_qa", split="test")
```

## After Downloading

Update `configs/default.yaml` to point to your downloaded data:

```yaml
spatial_qa:
  image_root: "./data/spatial_qa/images"   # change to local path
temporal_qa:
  video_root: "./data/temporal_qa/videos"
  image_root: "./data/temporal_qa/images"
```
