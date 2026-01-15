# Dataset Download Scripts - What Was Created

## ✅ Created Files

### 1. **`requirements.txt`** - Python Dependencies
All required packages to download and process datasets.

```bash
pip install -r requirements.txt
```

### 2. **`scripts/download_datasets.py`** - Main Download Script ⭐
Complete Python script to download all Molmo2 datasets from HuggingFace.

**Features**:
- Downloads ~30 datasets from HuggingFace
- Saves to `./data/molmo2_datasets/`
- Multiple download modes (all, by stage, specific datasets)
- Progress tracking
- Error handling (continues on failures)
- Resume support

**Usage**:
```bash
# Download everything
python scripts/download_datasets.py --all

# Download by stage
python scripts/download_datasets.py --stage pretraining
python scripts/download_datasets.py --stage sft

# Download specific groups
python scripts/download_datasets.py --molmo2-only
python scripts/download_datasets.py --pixmo-only

# Download one dataset
python scripts/download_datasets.py --dataset molmo2-cap

# List all available
python scripts/download_datasets.py --list

# Help
python scripts/download_datasets.py --help
```

### 3. **`scripts/download_all.sh`** - One-Command Bash Script
Simple bash wrapper for easy downloading.

```bash
bash scripts/download_all.sh
```

Does:
1. Checks Python installation
2. Installs dependencies
3. Downloads all datasets
4. Shows summary

### 4. **`scripts/README.md`** - Script Documentation
Detailed documentation for all download scripts including:
- Available datasets list
- Usage examples
- Troubleshooting
- Storage requirements
- Manual download instructions

### 5. **`DOWNLOAD_GUIDE.md`** - User Guide
Complete step-by-step guide for downloading datasets including:
- Installation instructions
- Download options
- File structure explanation
- Common issues & solutions
- After-download steps

### 6. **`START_HERE.md`** - Quick Start Guide
The simplest possible guide - just 2 commands to get started.

---

## What Gets Downloaded?

### Molmo2 Datasets (9 new datasets)
From HuggingFace `allenai/Molmo2-*`:

1. **Molmo2-Cap** - Dense video captioning
2. **Molmo2-AskModelAnything** - Human-authored video QA
3. **Molmo2-VideoCapQA** - Synthetic video QA  
4. **Molmo2-VideoSubtitleQA** - Video QA with subtitles
5. **Molmo2-VideoPoint** - Video temporal pointing (NOVEL)
6. **Molmo2-VideoTrack** - Video object tracking (NOVEL)
7. **Molmo2-MultiImageQA** - Multi-image QA
8. **Molmo2-SynMultiImageQA** - Synthetic multi-image QA
9. **Molmo2-MultiImagePoint** - Multi-image pointing

### PixMo Datasets (7 datasets)
From HuggingFace `allenai/pixmo-*`:

- pixmo-cap
- pixmo-points
- pixmo-count
- pixmo-docs
- pixmo-ask-model-anything
- pixmo-cap-qa
- pixmo-clocks

### Academic Datasets (13 datasets)
From HuggingFace (various orgs):

- VQA v2, TallyQA, ChartQA, DocVQA, TextVQA
- ST-VQA, InfographicVQA, AI2D, NLVR2
- A-OKVQA, OK-VQA, ScienceQA, LLaVA-Instruct

### Text Datasets (1 dataset)
- Tulu v2 SFT mixture

### Evaluation Datasets (4 datasets)
- Molmo2-CapEval
- Molmo2-VideoPointEval
- Molmo2-VideoCountEval
- Molmo2-VideoTrackEval

**Total**: ~30 datasets available on HuggingFace

---

## How It Works

### `download_datasets.py` Flow:

```
1. Parse command-line arguments
   ↓
2. Determine which datasets to download
   ↓
3. For each dataset:
   - Call HuggingFace datasets.load_dataset()
   - Download to cache
   - Save as .parquet files to data/molmo2_datasets/DATASET_NAME/
   ↓
4. Show summary (successful/failed)
```

### File Structure Created:

```
data/molmo2_datasets/
├── molmo2-cap/
│   ├── train.parquet
│   └── validation.parquet
├── molmo2-capqa/
│   └── train.parquet
├── pixmo-cap/
│   ├── train.parquet
│   └── validation.parquet
├── vqa-v2/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
└── ... (30+ datasets)
```

---

## Testing the Scripts

### Test 1: List Available Datasets
```bash
python scripts/download_datasets.py --list
```

Should show all ~30 datasets organized by category.

### Test 2: Download One Small Dataset
```bash
python scripts/download_datasets.py --dataset molmo2-capeval
```

Should download the evaluation dataset (~500MB).

### Test 3: Verify Downloaded Data
```bash
ls -lh data/molmo2_datasets/molmo2-capeval/
```

Should show `.parquet` files.

### Test 4: Inspect Downloaded Data
```bash
python scripts/inspect_molmo2_data.py molmo2-capeval --num-samples 3
```

Should display sample data.

---

## Download Sizes

| Dataset Group | Count | Size | Time (50 Mbps) |
|---------------|-------|------|----------------|
| Molmo2 only | 9 | ~30GB | 1-2 hours |
| PixMo only | 7 | ~50GB | 2-3 hours |
| Academic only | 13 | ~70GB | 3-4 hours |
| Pre-training | 5 | ~60GB | 2-4 hours |
| SFT (all) | 30+ | ~150GB | 4-8 hours |

---

## Comparison with Old Script

### Old: `download_molmo2_data_v2.py`
- ✓ Comprehensive dataset catalog (100+ datasets)
- ✓ Detailed documentation
- ⚠️ Includes many datasets NOT on HuggingFace (manual download)
- ⚠️ More complex configuration

### New: `download_datasets.py` ⭐
- ✓ **Actually downloads datasets** from HuggingFace
- ✓ Simple, focused on what's available NOW
- ✓ ~30 datasets ready to download
- ✓ Easy to use (fewer options)
- ✓ Resume support
- ✓ Works out of the box

**Both scripts are available** - use the new one for actual downloading!

---

## Quick Command Reference

```bash
# Installation
pip install -r requirements.txt

# Download everything (easiest)
bash scripts/download_all.sh

# Download everything (Python)
python scripts/download_datasets.py --all

# Download by stage
python scripts/download_datasets.py --stage pretraining
python scripts/download_datasets.py --stage sft
python scripts/download_datasets.py --stage eval

# Download specific groups
python scripts/download_datasets.py --molmo2-only
python scripts/download_datasets.py --pixmo-only
python scripts/download_datasets.py --academic-only

# Download one dataset
python scripts/download_datasets.py --dataset molmo2-cap

# List available
python scripts/download_datasets.py --list

# Custom location
python scripts/download_datasets.py --all --output-dir /data/datasets

# Help
python scripts/download_datasets.py --help
```

---

## Documentation Files

1. **START_HERE.md** - Quickest guide (2 commands)
2. **DOWNLOAD_GUIDE.md** - Complete download guide
3. **scripts/README.md** - Script documentation
4. **SCRIPTS_CREATED.md** - This file
5. **QUICKSTART.md** - Full training guide
6. **MOLMO2_TECH_REPORT_SUMMARY.md** - Paper summary

---

## What's Next?

After downloading datasets:

1. **Verify**: `ls data/molmo2_datasets/`
2. **Inspect**: `python scripts/inspect_molmo2_data.py molmo2-cap`
3. **Test dataloader**: `python examples/train_with_stage_dataloaders.py --stage 1 --inspect`
4. **Train**: `python examples/train_with_stage_dataloaders.py --stage 1`

---

**Created**: 2026-01-15  
**Status**: ✅ Ready to use  
**Total Datasets**: ~30 from HuggingFace  
**Total Size**: ~150GB
