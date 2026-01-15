# How to Download Molmo2 Datasets

**Simple guide to download all datasets used in the Molmo2 technical report.**

---

## TL;DR - Just Download Everything

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download all datasets (one command)
bash scripts/download_all.sh
```

**That's it!** Wait 4-8 hours and you'll have all ~30 datasets (~150GB) in `./data/molmo2_datasets/`

---

## Step-by-Step Guide

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `datasets` - HuggingFace datasets library
- `transformers` - For tokenizers and models
- `torch` - PyTorch
- `opencv-python`, `Pillow` - Image/video processing
- And other utilities

### Step 2: Choose What to Download

#### Option A: Download Everything (Recommended)

```bash
python scripts/download_datasets.py --all
```

Downloads all ~30 datasets:
- 9 new Molmo2 datasets
- 7 PixMo datasets  
- 13 academic datasets
- 1 text dataset

**Size**: ~150GB  
**Time**: 4-8 hours  
**Storage**: Make sure you have 200GB+ free space

#### Option B: Download by Training Stage

**For Pre-training only** (~60GB):
```bash
python scripts/download_datasets.py --stage pretraining
```

Downloads:
- pixmo-cap
- pixmo-points
- pixmo-count
- tulu-v2-sft

**For SFT stage** (~150GB):
```bash
python scripts/download_datasets.py --stage sft
```

Downloads all datasets (same as `--all`)

**For Evaluation** (~5GB):
```bash
python scripts/download_datasets.py --stage eval
```

Downloads only evaluation benchmarks

#### Option C: Download Specific Dataset Groups

**Only the 9 new Molmo2 datasets** (~30GB):
```bash
python scripts/download_datasets.py --molmo2-only
```

**Only PixMo datasets** (~50GB):
```bash
python scripts/download_datasets.py --pixmo-only
```

**Only academic datasets** (~70GB):
```bash
python scripts/download_datasets.py --academic-only
```

#### Option D: Download One Specific Dataset

```bash
# Download just one dataset
python scripts/download_datasets.py --dataset molmo2-cap

# Examples of specific datasets:
python scripts/download_datasets.py --dataset pixmo-cap
python scripts/download_datasets.py --dataset vqa-v2
python scripts/download_datasets.py --dataset molmo2-videopoint
```

### Step 3: Verify Downloads

List downloaded datasets:
```bash
ls -lh data/molmo2_datasets/
```

Check sizes:
```bash
du -sh data/molmo2_datasets/*
```

Inspect a dataset:
```bash
python scripts/inspect_molmo2_data.py molmo2-cap --num-samples 5
```

---

## What Gets Downloaded?

### Molmo2 Datasets (9 datasets - THE NEW STUFF!)

| Dataset | Size | Description |
|---------|------|-------------|
| molmo2-cap | ~10GB | Dense video captioning |
| molmo2-askmodelanything | ~2GB | Human-authored video QA |
| molmo2-capqa | ~15GB | Synthetic video QA |
| molmo2-subtitleqa | ~8GB | Video QA with subtitles |
| molmo2-videopoint | ~12GB | Video temporal pointing ⭐ |
| molmo2-videotrack | ~10GB | Video object tracking ⭐ |
| molmo2-multiimageqa | ~3GB | Multi-image QA |
| molmo2-synmultiimageqa | ~7GB | Synthetic multi-image QA |
| molmo2-multiimagepoint | ~15GB | Multi-image pointing |

⭐ = Novel contributions from Molmo2 paper

### PixMo Datasets (7 datasets)

| Dataset | Size | Description |
|---------|------|-------------|
| pixmo-cap | ~25GB | Dense image captioning |
| pixmo-points | ~10GB | Image pointing |
| pixmo-count | ~5GB | Object counting |
| pixmo-docs | ~8GB | Document understanding |
| pixmo-ask-model-anything | ~3GB | Human-authored image QA |
| pixmo-cap-qa | ~7GB | Synthetic QA |
| pixmo-clocks | ~12GB | Clock reading |

### Academic Datasets (13 datasets on HuggingFace)

Popular VQA and vision datasets:
- VQA v2, TallyQA, ChartQA, DocVQA, TextVQA, etc.
- Total: ~70GB

---

## File Structure

After download, your directory will look like:

```
data/molmo2_datasets/
├── molmo2-cap/
│   ├── train.parquet          # Training data
│   └── validation.parquet      # Validation data
│
├── molmo2-capqa/
│   └── train.parquet
│
├── pixmo-cap/
│   ├── train.parquet
│   └── validation.parquet
│
├── vqa-v2/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
│
└── ... (30+ datasets total)
```

Each dataset is saved as `.parquet` files (efficient columnar format).

---

## Common Issues & Solutions

### Issue: "No module named 'datasets'"

**Solution**:
```bash
pip install datasets transformers
```

### Issue: "403 Forbidden" or dataset access denied

**Solution**: Some datasets require HuggingFace login

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login (you'll need a HuggingFace account)
huggingface-cli login

# Then retry download
python scripts/download_datasets.py --all
```

### Issue: Not enough disk space

**Solution**: Download only what you need

```bash
# Just pre-training datasets (60GB)
python scripts/download_datasets.py --stage pretraining

# Or just Molmo2 datasets (30GB)
python scripts/download_datasets.py --molmo2-only
```

### Issue: Download interrupted

**Solution**: Just re-run the same command. The script will skip already downloaded datasets.

```bash
python scripts/download_datasets.py --all
```

### Issue: Very slow download

**Solution**: 
1. Check your internet connection
2. Try downloading during off-peak hours
3. Download in stages:
   ```bash
   python scripts/download_datasets.py --molmo2-only
   # Wait for completion, then:
   python scripts/download_datasets.py --pixmo-only
   # etc.
   ```

---

## Advanced Options

### Custom Output Directory

```bash
python scripts/download_datasets.py --all --output-dir /mnt/data/datasets
```

### Custom Cache Directory

```bash
python scripts/download_datasets.py --all --cache-dir /mnt/cache/huggingface
```

### List Available Datasets

```bash
python scripts/download_datasets.py --list
```

### Stop on First Error

```bash
python scripts/download_datasets.py --all --no-skip-errors
```

(By default, the script continues even if some datasets fail)

---

## Download Time Estimates

Based on 50 Mbps internet connection:

| What | Size | Time |
|------|------|------|
| Single dataset | 1-15GB | 5-30 min |
| Molmo2 only | ~30GB | 1-2 hours |
| Pre-training | ~60GB | 2-4 hours |
| Everything | ~150GB | 4-8 hours |

With 100 Mbps: Cut times in half  
With 25 Mbps: Double the times

---

## After Download

### 1. Verify Your Downloads

```bash
# List all datasets
ls data/molmo2_datasets/

# Check a specific dataset
python scripts/inspect_molmo2_data.py molmo2-cap
```

### 2. Test the Dataloaders

```bash
python examples/train_with_stage_dataloaders.py --stage 1 --inspect
```

### 3. Start Training

```bash
# Stage 1: Pre-training
python examples/train_with_stage_dataloaders.py --stage 1 --batch-size 32
```

---

## Quick Reference

```bash
# Download everything
bash scripts/download_all.sh

# OR use Python script for more control
python scripts/download_datasets.py --all                    # All datasets
python scripts/download_datasets.py --stage pretraining      # Pre-training only
python scripts/download_datasets.py --molmo2-only            # Molmo2 datasets only
python scripts/download_datasets.py --dataset molmo2-cap     # One dataset
python scripts/download_datasets.py --list                   # List available

# Verify
ls -lh data/molmo2_datasets/
python scripts/inspect_molmo2_data.py molmo2-cap

# Test
python examples/train_with_stage_dataloaders.py --stage 1 --inspect

# Train
python examples/train_with_stage_dataloaders.py --stage 1
```

---

**Need help?** See `scripts/README.md` for detailed documentation.

**Last Updated**: 2026-01-15
