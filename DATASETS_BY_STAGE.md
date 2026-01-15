# Molmo2 Datasets by Training Stage

Based on the **Molmo2 Technical Report**

---

## Overview

| Stage | Datasets | Storage | Description |
|-------|----------|---------|-------------|
| **Stage 1** | **5** | **~80GB** | Pre-training with fixed ratios |
| **Stage 2** | **100+** | **~500GB (auto) + 2-5TB (manual)** | SFT with sqrt-proportional sampling |
| **Stage 3** | **Same as Stage 2** | **No additional download** | Same data, longer sequences |

---

## Stage 1: Pre-training (5 datasets)

**Goal**: Learn general visual-language alignment

**Configuration**:
- Steps: ~100K
- Sequence length: 4,096 tokens
- Frame limit: 128 frames

**Data Mixture (Fixed Ratios)**:
- **60%** Dense captioning
- **30%** Image pointing
- **10%** NLP data

### Datasets:

| # | Dataset | Description | Examples | Ratio | Size |
|---|---------|-------------|----------|-------|------|
| 1 | `pixmo-cap` | Dense image captioning (~200 words avg) | 710k | 60% | ~30GB |
| 2 | `pixmo-points` | Image pointing with referring expressions | 800k | 15% | ~20GB |
| 3 | `pixmo-count` | Object counting QA | 800k | 10% | ~15GB |
| 4 | `cosyn-point` | Synthetic pointing data | 500k | 5% | ~10GB |
| 5 | `tulu-v2-sft-mixture` | Text-only instruction data | 980k | 10% | ~5GB |

**Total: 5 datasets, ~80GB**

### Download:

```bash
./scripts/download_by_stage.sh stage1
```

Or:

```bash
python3 scripts/download_stage1_pretraining.py
```

---

## Stage 2: Supervised Fine-Tuning (100+ datasets)

**Goal**: Instruction following and multimodal task learning

**Configuration**:
- Steps: ~50K
- Sequence length: 4,096 tokens
- Frame limit: 128 frames

**Data Mixture**: Sqrt-proportional sampling
- Formula: `Weight_i = sqrt(Size_i) / sum(sqrt(Size_j))`

### Dataset Categories:

#### 1. Molmo2 Original Datasets (9 new releases)

| # | Dataset | Description | Examples | Size |
|---|---------|-------------|----------|------|
| 1 | `Molmo2-Cap` | Video dense captioning | 104k video + 431k clip | ~50GB |
| 2 | `Molmo2-AskModelAnything` | Human-authored video QA | 43k | ~20GB |
| 3 | `Molmo2-VideoCapQA` | Synthetic QA from video captions | 1M | ~40GB |
| 4 | `Molmo2-VideoSubtitleQA` | Video QA with subtitles | 300k | ~30GB |
| 5 | `Molmo2-VideoPoint` | **Video temporal pointing (NOVEL)** | 330k | ~25GB |
| 6 | `Molmo2-VideoTrack` | **Video object tracking (NOVEL)** | 220k | ~25GB |
| 7 | `Molmo2-MultiImageQA` | Multi-image QA | 45k | ~15GB |
| 8 | `Molmo2-SynMultiImageQA` | Synthetic multi-image QA | 188k | ~20GB |
| 9 | `Molmo2-MultiImagePoint` | Multi-image pointing | 470k | ~25GB |

#### 2. PixMo Datasets (6 datasets)

| # | Dataset | Description | Examples | Size |
|---|---------|-------------|----------|------|
| 1 | `pixmo-cap` | Dense image captioning | 710k | ~30GB |
| 2 | `pixmo-ask-model-anything` | Human-authored image QA | 71k | ~10GB |
| 3 | `pixmo-cap-qa` | Synthetic QA from captions | 190k | ~15GB |
| 4 | `pixmo-clocks` | Clock reading | 800k | ~10GB |
| 5 | `pixmo-points` | Image pointing | 800k | ~20GB |
| 6 | `pixmo-count` | Object counting | 800k | ~15GB |

#### 3. Academic Image Datasets (12+ datasets)

| # | Dataset | Description | Examples | Size |
|---|---------|-------------|----------|------|
| 1 | `ai2d` | Diagram understanding | 15k | ~5GB |
| 2 | `chartqa` | Chart understanding | 28k | ~10GB |
| 3 | `docvqa` | Document VQA | 39k | ~15GB |
| 4 | `textvqa` | Text-based VQA | 35k | ~10GB |
| 5 | `scienceqa` | Science QA | 6.2k | ~3GB |
| 6 | `aokvqa` | Knowledge-based VQA | 34k | ~8GB |
| 7 | `okvqa` | Outside knowledge VQA | 9k | ~3GB |
| 8 | `infographicvqa` | Infographic VQA | 24k | ~8GB |
| 9 | `stvqa` | Scene text VQA | 25k | ~8GB |
| 10 | `tallyqa` | Counting QA | 250k | ~12GB |
| 11 | `gqa` | Visual reasoning | 1M+ | ~25GB |
| 12 | `nlvr2` | Visual reasoning | 86k | ~8GB |

#### 4. Video Datasets (4+ auto, 50+ manual)

**Auto-downloadable:**

| # | Dataset | Description | Examples | Size |
|---|---------|-------------|----------|------|
| 1 | `nextqa` | Video QA with reasoning | 34k | ~50GB |
| 2 | `perception-test` | Video perception | 12k | ~40GB |
| 3 | `activitynet-qa` | Activity video QA | 58k | ~100GB |
| 4 | `videochatgpt` | Video instruction following | 100k | ~80GB |

**Manual download (50+ datasets):**
- `vqa-v2`, `llava-665k-multi`, `tgif`, `tvqa`, `sports-qa`, `funqa`
- `mevis`, `refvos`, `vicas`, `trackingnet`
- And 40+ more...

#### 5. Text/NLP Data (1 dataset)

| # | Dataset | Description | Examples | Size |
|---|---------|-------------|----------|------|
| 1 | `tulu-v2-sft-mixture` | Text-only instruction data | 980k | ~5GB |

### Download:

```bash
./scripts/download_by_stage.sh stage2
```

Or:

```bash
python3 scripts/download_stage2_sft.py
```

**Total (Auto): ~32 datasets, ~500GB**
**Total (Manual): ~60+ datasets, ~2-5TB**

---

## Stage 3: Long-Context SFT

**Goal**: Handle extended video sequences

**Configuration**:
- Steps: 2K (short fine-tuning)
- Sequence length: **36,864 tokens** (9x longer than Stage 2)
- Frame limit: **384 frames** (3x more than Stage 2)

### **Key Point: SAME DATASETS AS STAGE 2**

Stage 3 uses the **exact same datasets** as Stage 2, but with:
- Longer sequence length
- More video frames
- Context Parallelism (CP) for distributed training

**No additional download needed!**

### Training:

```bash
# Use Stage 2 data with longer sequences
python examples/train_stage3.py --seq-length 36864 --max-frames 384
```

---

## Summary

| Stage | # Datasets | Storage | Download Command |
|-------|------------|---------|------------------|
| **Stage 1** | 5 | ~80GB | `./scripts/download_by_stage.sh stage1` |
| **Stage 2** | 32 (auto) | ~500GB | `./scripts/download_by_stage.sh stage2` |
| **Stage 2** | 60+ (manual) | ~2-5TB | Manual from original sources |
| **Stage 3** | Same as Stage 2 | 0 (already downloaded) | No download needed |

---

## Recommended Download Strategy

### Limited Storage (<500GB):
```bash
# Just Stage 1 - enough to learn VLM basics
./scripts/download_by_stage.sh stage1
```

### Medium Storage (500GB - 1TB):
```bash
# Stage 1 + Stage 2 (auto-downloadable only)
./scripts/download_by_stage.sh all
```

### Full Training (5TB+):
```bash
# All auto-downloadable + manual datasets
./scripts/download_by_stage.sh all
# Then manually download from original sources
```

---

## Quick Reference

**Stage 1 datasets (5 total):**
1. pixmo-cap
2. pixmo-points
3. pixmo-count
4. cosyn-point
5. tulu-v2-sft-mixture

**Stage 2 adds (100+ total):**
- 9 Molmo2 original datasets
- 6 PixMo datasets (overlap with Stage 1)
- 12+ Academic image datasets
- 4+ Video datasets (auto)
- 50+ Video/tracking datasets (manual)

**Stage 3:**
- Same as Stage 2, no new data
