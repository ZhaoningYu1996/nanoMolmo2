# Dataset Download Scripts

Scripts to download all datasets used in the Molmo2 technical report.

---

## Quick Start

### Method 1: One-Command Download (Recommended)

```bash
# Download ALL datasets (Molmo2 + PixMo + Academic)
bash scripts/download_all.sh
```

This will:
1. Install required Python packages
2. Download all ~30 datasets from HuggingFace
3. Save them to `./data/molmo2_datasets/`

**Expected**: ~150GB total, 4-8 hours download time

---

### Method 2: Python Script (More Control)

Install dependencies first:
```bash
pip install -r requirements.txt
```

Then download datasets:

```bash
# Download ALL datasets
python scripts/download_datasets.py --all

# Download by training stage
python scripts/download_datasets.py --stage pretraining
python scripts/download_datasets.py --stage sft
python scripts/download_datasets.py --stage eval

# Download only Molmo2 datasets (9 new datasets)
python scripts/download_datasets.py --molmo2-only

# Download only PixMo datasets
python scripts/download_datasets.py --pixmo-only

# Download specific dataset
python scripts/download_datasets.py --dataset molmo2-cap

# Custom output directory
python scripts/download_datasets.py --all --output-dir /path/to/datasets

# List all available datasets
python scripts/download_datasets.py --list
```

---

## Available Datasets

### Molmo2 Datasets (9 new datasets from the paper)

1. **molmo2-cap** - Dense video captioning (104k video-level, 431k clip-level)
2. **molmo2-askmodelanything** - Human-authored video QA (43k)
3. **molmo2-capqa** - Synthetic QA from video captions (1M)
4. **molmo2-subtitleqa** - Video QA with subtitle context (300k)
5. **molmo2-videopoint** - Video temporal pointing (330k) ⭐ Novel
6. **molmo2-videotrack** - Video object tracking (220k) ⭐ Novel
7. **molmo2-multiimageqa** - Multi-image QA (45k)
8. **molmo2-synmultiimageqa** - Synthetic multi-image QA (188k)
9. **molmo2-multiimagepoint** - Multi-image pointing (470k)

### Molmo2 Evaluation Datasets (4 datasets)

- **molmo2-capeval** - Caption evaluation benchmark
- **molmo2-videopointeval** - Video pointing evaluation
- **molmo2-videocounteval** - Video counting evaluation
- **molmo2-videotrackeval** - Video tracking evaluation

### PixMo Datasets (7 datasets)

- **pixmo-cap** - Dense image captioning (710k)
- **pixmo-points** - Image pointing (varied)
- **pixmo-count** - Object counting (varied)
- **pixmo-docs** - Document understanding
- **pixmo-ask-model-anything** - Human-authored image QA (71k)
- **pixmo-cap-qa** - Synthetic QA from captions (190k)
- **pixmo-clocks** - Clock reading (800k)

### Academic Datasets (13 datasets on HuggingFace)

- **llava-instruct** - LLaVA instruction-following (2.5M)
- **vqa-v2** - Visual Question Answering v2 (440k)
- **tallyqa** - Counting QA (250k)
- **chartqa** - Chart understanding (28k)
- **docvqa** - Document VQA (39k)
- **textvqa** - Text-based VQA (35k)
- **st-vqa** - Scene text VQA (25k)
- **infographicvqa** - Infographic VQA (24k)
- **ai2d** - Diagram understanding (15k)
- **nlvr2** - Natural language visual reasoning (86k)
- **a-okvqa** - Knowledge-based VQA (34k)
- **ok-vqa** - Outside knowledge VQA (9k)
- **scienceqa** - Science QA (6.2k)

### Text Datasets

- **tulu-v2-sft** - Text-only instruction data (980k)

---

## Training Stage Datasets

### Stage 1: Pre-training (5 datasets)
- pixmo-cap (60% of mixture)
- pixmo-points (25%)
- pixmo-count (5%)
- tulu-v2-sft (10%)

Download:
```bash
python scripts/download_datasets.py --stage pretraining
```

### Stage 2: SFT (30+ datasets)
- All 9 Molmo2 datasets
- All 7 PixMo datasets
- Academic datasets (VQA, DocVQA, ChartQA, etc.)
- Text datasets (Tulu)

Download:
```bash
python scripts/download_datasets.py --stage sft
```

### Stage 3: Long-Context (same as Stage 2)
Uses the same datasets as SFT but with longer sequences.

---

## File Structure After Download

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
│   └── validation.parquet
└── ...
```

Each dataset folder contains:
- `train.parquet` - Training split
- `validation.parquet` - Validation split (if available)
- `test.parquet` - Test split (if available)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "403 Forbidden" or "401 Unauthorized"

Some datasets require HuggingFace login:

1. Create account at https://huggingface.co/
2. Login via CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
3. Accept dataset licenses on HuggingFace website

### Dataset download fails

Some academic datasets may require:
- Accepting terms on HuggingFace
- Manual download from original sources
- Special access permissions

The script will continue with other datasets if `--skip-errors` is used (default).

### Disk space issues

Total download size: ~150GB

Check available space:
```bash
df -h ./data
```

To download only essential datasets:
```bash
# Only Molmo2 datasets (~30GB)
python scripts/download_datasets.py --molmo2-only

# Only pre-training datasets (~50GB)
python scripts/download_datasets.py --stage pretraining
```

### Slow downloads

Downloads are limited by:
1. Your internet speed
2. HuggingFace server speed
3. Geographic location

Expected time: 4-8 hours for all datasets

To resume interrupted downloads, just re-run the same command. Already downloaded datasets will be skipped.

---

## Verifying Downloads

Check what's downloaded:
```bash
ls -lh data/molmo2_datasets/
```

Count datasets:
```bash
ls -1 data/molmo2_datasets/ | wc -l
```

Check dataset sizes:
```bash
du -sh data/molmo2_datasets/*
```

Inspect a dataset:
```bash
python scripts/inspect_molmo2_data.py molmo2-cap --num-samples 5
```

---

## Manual Download (For datasets not on HuggingFace)

Some datasets mentioned in the Molmo2 paper require manual download from original sources:

### Video Datasets
- **TGIF**: https://github.com/raingo/TGIF-Release
- **TVQA**: https://tvqa.cs.unc.edu/
- **Kinetics**: https://www.deepmind.com/open-source/kinetics
- **Ego4D**: https://ego4d-data.org/

### Tracking Datasets
- **TrackingNet**: https://tracking-net.org/
- **GOT-10k**: http://got-10k.aitestunion.com/
- **LaSOT**: https://vision.cs.stonybrook.edu/~lasot/

### Chart/Table Datasets
- **PlotQA**: https://github.com/NiteshMethani/PlotQA
- **DVQA**: https://github.com/kushalkafle/DVQA_dataset
- **FigureQA**: https://www.microsoft.com/en-us/research/project/figureqa-dataset/

After manual download, place them in `data/molmo2_datasets/DATASET_NAME/` following the same structure.

---

## Storage Requirements

| Category | Datasets | Size | Download Time |
|----------|----------|------|---------------|
| Molmo2 Only | 9 | ~30GB | 1-2 hours |
| PixMo Only | 7 | ~50GB | 2-3 hours |
| Pre-training | 5 | ~60GB | 2-4 hours |
| SFT | 30+ | ~150GB | 4-8 hours |
| All | 30+ | ~150GB | 4-8 hours |

*Times assume 50 Mbps internet connection

---

## Next Steps

After downloading datasets:

1. **Verify downloads**:
   ```bash
   python scripts/inspect_molmo2_data.py
   ```

2. **Test dataloaders**:
   ```bash
   python examples/train_with_stage_dataloaders.py --stage 1 --inspect
   ```

3. **Start training**:
   ```bash
   python examples/train_with_stage_dataloaders.py --stage 1
   ```

---

## Additional Resources

- **Full dataset list**: See `scripts/download_datasets.py` for complete list
- **Dataset documentation**: See each dataset's HuggingFace page
- **Molmo2 paper**: https://allenai.org/blog/molmo2
- **HuggingFace collection**: https://huggingface.co/collections/allenai/molmo2-data

---

**Last Updated**: 2026-01-15  
**Version**: 1.0
