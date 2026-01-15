# ⚡ START HERE - Download Molmo2 Datasets

**Two commands to download all datasets from the Molmo2 paper.**

---

## ✅ Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `datasets`, `transformers`, `torch`, `opencv-python`, and other utilities.

---

## ✅ Step 2: Download Datasets

### Option 1: One-Command Download (Easiest)

```bash
bash scripts/download_all.sh
```

**That's it!** This will download all ~30 datasets (~150GB) to `./data/molmo2_datasets/`

---

### Option 2: More Control with Python Script

```bash
# Download everything
python scripts/download_datasets.py --all

# Or download by stage
python scripts/download_datasets.py --stage pretraining  # 60GB, 5 datasets
python scripts/download_datasets.py --stage sft          # 150GB, 30+ datasets

# Or download only Molmo2 datasets
python scripts/download_datasets.py --molmo2-only        # 30GB, 9 datasets

# Or download one dataset
python scripts/download_datasets.py --dataset molmo2-cap

# See all options
python scripts/download_datasets.py --help
```

---

## What Gets Downloaded?

### The 9 New Molmo2 Datasets (from the paper)

```
✓ molmo2-cap               Dense video captioning
✓ molmo2-askmodelanything  Human-authored video QA
✓ molmo2-capqa             Synthetic video QA
✓ molmo2-subtitleqa        Video QA with subtitles
✓ molmo2-videopoint        Video temporal pointing (NOVEL)
✓ molmo2-videotrack        Video object tracking (NOVEL)
✓ molmo2-multiimageqa      Multi-image QA
✓ molmo2-synmultiimageqa   Synthetic multi-image QA
✓ molmo2-multiimagepoint   Multi-image pointing
```

### Plus: PixMo, Academic, and Text Datasets

```
✓ 7 PixMo datasets (pixmo-cap, pixmo-points, etc.)
✓ 13 Academic datasets (VQA v2, DocVQA, ChartQA, etc.)
✓ 1 Text dataset (Tulu v2)
```

**Total**: ~30 datasets, ~150GB

---

## Verify Downloads

```bash
# List downloaded datasets
ls data/molmo2_datasets/

# Inspect a dataset
python scripts/inspect_molmo2_data.py molmo2-cap --num-samples 5

# Test dataloader
python examples/train_with_stage_dataloaders.py --stage 1 --inspect
```

---

## Start Training

```bash
# Stage 1: Pre-training
python examples/train_with_stage_dataloaders.py --stage 1 --batch-size 32

# Stage 2: SFT
python examples/train_with_stage_dataloaders.py --stage 2 --batch-size 16

# Stage 3: Long-context
python examples/train_with_stage_dataloaders.py --stage 3 --batch-size 4
```

---

## Need More Help?

- **[DOWNLOAD_GUIDE.md](./DOWNLOAD_GUIDE.md)** - Complete download guide
- **[scripts/README.md](./scripts/README.md)** - Detailed script documentation
- **[QUICKSTART.md](./QUICKSTART.md)** - Full training guide
- **[SUMMARY.md](./SUMMARY.md)** - Complete project summary

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│  DOWNLOAD COMMANDS                                      │
├─────────────────────────────────────────────────────────┤
│  bash scripts/download_all.sh              # Everything │
│  python scripts/download_datasets.py --all # Everything │
│                                                          │
│  python scripts/download_datasets.py --stage pretraining│
│  python scripts/download_datasets.py --stage sft        │
│  python scripts/download_datasets.py --molmo2-only      │
│  python scripts/download_datasets.py --pixmo-only       │
│                                                          │
│  python scripts/download_datasets.py --list # Show all  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  VERIFY & TEST                                          │
├─────────────────────────────────────────────────────────┤
│  ls data/molmo2_datasets/                               │
│  python scripts/inspect_molmo2_data.py molmo2-cap       │
│  python examples/train_with_stage_dataloaders.py \      │
│         --stage 1 --inspect                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  TRAIN                                                  │
├─────────────────────────────────────────────────────────┤
│  python examples/train_with_stage_dataloaders.py \      │
│         --stage 1 --batch-size 32                       │
└─────────────────────────────────────────────────────────┘
```

---

**Time**: 4-8 hours to download  
**Size**: ~150GB total  
**Datasets**: ~30 datasets from Molmo2 paper

**Ready? Run**: `bash scripts/download_all.sh`
