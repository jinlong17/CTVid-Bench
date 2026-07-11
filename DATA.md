# CTVid-Bench — Data Download Instructions

Current Hugging Face dataset: `jinlong17/CTVid-Bench`.

The public release is organized as one dataset repository with top-level
`spatial/`, `temporal/`, and `training/` folders. Testing QA is QA-v2 for the
public variants. Training source frames and OCR JSON are available; refined
training QA is prepared internally but has not been uploaded to the public
dataset yet.

## What's in This Repo

Annotation JSON files are included directly:

| Path | Size | Contents |
|------|------|----------|
| `data/spatial/VQA_json/GT_VQA_testing/` | 9.0 MB | 389 QA-v2 JSON files |
| `data/spatial/VQA_json/blur_VQA_testing/` | 9.0 MB | 389 QA-v2 JSON files |
| `data/spatial/VQA_json/downsample_x4_VQA_testing/` | 9.0 MB | 389 QA-v2 JSON files |
| `data/temporal/videos_vqa/GT/vqa/` | 1.7 MB | 97 QA-v2 JSON files |
| `data/temporal/videos_vqa/blur/vqa/` | 1.7 MB | 97 QA-v2 JSON files |
| `data/temporal/videos_vqa/downsample_x4/vqa/` | 1.7 MB | 97 QA-v2 JSON files |

The older `data/spatial_qa/annotations/` and `data/temporal_qa/annotations/`
template-QA layout is no longer part of this repo. Current JSON path fields are
relative to the public download root, e.g. `spatial/VQA_img/...` and
`temporal/videos_vqa/...`.

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
huggingface-cli download jinlong17/CTVid-Bench \
    --repo-type dataset \
    --include "spatial/VQA_img/*" "spatial/VQA_json/*" \
    --local-dir ./data/

# Temporal QA images
huggingface-cli download jinlong17/CTVid-Bench \
    --repo-type dataset \
    --include "temporal/images/*" \
    --local-dir ./data/

# Temporal QA videos (smaller)
huggingface-cli download jinlong17/CTVid-Bench \
    --repo-type dataset \
    --include "temporal/videos_vqa/*" \
    --local-dir ./data/
```

### Option C — Python datasets library

```python
from datasets import load_dataset

# Spatial QA
ds = load_dataset("jinlong17/CTVid-Bench", name="spatial_qa", split="test")

# Temporal QA
ds = load_dataset("jinlong17/CTVid-Bench", name="temporal_qa", split="test")
```

## After Downloading

Update `configs/default.yaml` to point to your downloaded data:

```yaml
spatial_qa:
  image_root: "./data/spatial/VQA_img"   # change to local path
temporal_qa:
  video_root: "./data/temporal/videos_vqa"
  image_root: "./data/temporal/images"
```
