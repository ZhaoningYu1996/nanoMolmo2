# nanoMolmo2

**🎓 A Vision-Language Model (VLM) Learning Project**

A minimal implementation of Molmo2 VLM from scratch for educational purposes.

## Overview

**nanoMolmo2** is an educational reimplementation of the [Molmo2](https://molmo.allenai.org/) Vision-Language Model, designed to help developers **learn and understand modern VLM architectures from the ground up**. This hands-on project uses **Qwen3-0.6B** as the base language model while following Molmo2's architecture and training methodology.

🎯 **Primary Goal**: Provide a clear, educational implementation for learning how Vision-Language Models work - from architecture design to multimodal training.

> ⚠️ **Note**: This is a **learning-focused educational project**, not intended for production use.

## Architecture

**nanoMolmo2**: Educational VLM with frozen vision encoder for efficiency

- **Vision Encoder**: Molmo2's CLIP ViT (~300M params) - **🔒 FROZEN during training**
- **Connector**: Linear/MLP projection (~1M params) - **✏️ TRAINABLE**
- **Base LLM**: Qwen3-0.6B (~500M params) - **✏️ TRAINABLE**
- **Training Objective**: Same as Molmo2 (multimodal next-token prediction)

**Why frozen vision encoder?**
- ✅ **50% less memory** (~20GB vs ~30GB per GPU)
- ✅ **30-40% faster training** (skip vision backward pass)
- ✅ **Stable features** (pre-trained CLIP is already excellent)
- ✅ **Focus learning** on language understanding

**Total trainable**: ~501M parameters  
**Hardware**: Runs on 2-4 A100 40GB GPUs

See [MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md) for complete details.

## Quick Start

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download datasets

```bash
# Stage 1 pre-training only (~80GB, 5 datasets)
python scripts/download_datasets.py --stage pretrain

# Stage 2 & 3 SFT datasets (~500GB, 29 datasets)
python scripts/download_datasets.py --stage sft

# All stages
python scripts/download_datasets.py --stage all

# Useful options:
python scripts/download_datasets.py --list      # See all datasets
python scripts/download_datasets.py --check     # Check download status
python scripts/download_datasets.py --dry-run   # Preview without downloading
```

**Storage by Stage** (based on Molmo2 tech report):
- **Stage 1**: ~80GB (5 datasets) - Pre-training with fixed ratios
- **Stage 2 & 3**: ~500GB (29 datasets) - SFT (Stage 3 uses same data)

See [DATASETS_BY_STAGE.md](./DATASETS_BY_STAGE.md) for complete breakdown.

### Step 3: Train

```bash
# Stage 1: Pre-training
python examples/train_with_stage_dataloaders.py --stage 1

# Stage 2: SFT
python examples/train_with_stage_dataloaders.py --stage 2

# Stage 3: Long-context (same data, longer sequences)
python examples/train_with_stage_dataloaders.py --stage 3
```

## Project Structure

```
nanoMolmo2/
├── config/
│   ├── model_config.yaml         # Model architecture config
│   └── train_config.yaml         # Training parameters
├── data/
│   ├── dataloaders/              # Dataset implementations
│   │   ├── base.py               # Base classes
│   │   ├── image_datasets.py     # Image dataset loaders
│   │   ├── video_datasets.py     # Video dataset loaders
│   │   └── utils.py              # Utilities (packing, weighting)
│   └── stage_dataloaders.py      # Stage-specific data modules
├── examples/
│   ├── minimal_pure_pytorch.py   # Minimal VLM implementation
│   └── train_with_stage_dataloaders.py  # Training example
├── scripts/
│   ├── download_datasets.py      # Dataset downloader
│   ├── inspect_molmo2_data.py    # Data inspection tool
│   └── verify_model_setup.py     # Setup verification
├── tests/
│   └── test_dataloaders.py       # Unit tests
├── DATASETS_BY_STAGE.md          # Dataset breakdown by stage
├── MODEL_ARCHITECTURE.md         # Architecture details
├── MOLMO2_TECH_REPORT_SUMMARY.md # Tech report summary
├── PURE_PYTORCH_GUIDE.md         # Pure PyTorch implementation
├── QUICKSTART.md                 # Quick start guide
├── TRAINING_PIPELINE.md          # Training pipeline details
├── YOUR_SETUP.md                 # Your specific setup
├── requirements.txt              # Full dependencies
└── requirements_minimal.txt      # Minimal dependencies
```

## Training Pipeline

Based on Molmo2's 3-stage approach:

```
Stage 1: Pre-training (5 datasets, ~80GB)
├── 60% Dense captioning (PixMo-Cap)
├── 30% Image pointing (PixMo-Points, PixMo-Count, CoSyn-Point)
└── 10% NLP data (Tulu)
    ↓
Stage 2: Supervised Fine-Tuning (100+ datasets)
├── Molmo2 datasets (video cap, QA, pointing, tracking)
├── PixMo datasets (image cap, QA, pointing)
└── Academic datasets (VQA, DocVQA, ChartQA, ...)
    ↓
Stage 3: Long-Context SFT (same datasets as Stage 2)
├── Longer sequences: 36,864 tokens (vs 4,096)
└── More frames: 384 (vs 128)
```

## Documentation

- **[DATASETS_BY_STAGE.md](./DATASETS_BY_STAGE.md)** - Complete dataset breakdown
- **[MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)** - Architecture details
- **[MOLMO2_TECH_REPORT_SUMMARY.md](./MOLMO2_TECH_REPORT_SUMMARY.md)** - Molmo2 paper summary
- **[TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md)** - Training pipeline guide
- **[PURE_PYTORCH_GUIDE.md](./PURE_PYTORCH_GUIDE.md)** - Pure PyTorch implementation
- **[QUICKSTART.md](./QUICKSTART.md)** - Detailed quick start
- **[YOUR_SETUP.md](./YOUR_SETUP.md)** - Your specific configuration

## References

- [Molmo2 Blog Post](https://allenai.org/blog/molmo2)
- [Molmo2 Technical Report](https://molmo.allenai.org/)
- [Qwen3 Model](https://github.com/QwenLM/Qwen3)
- [HuggingFace Molmo2 Collection](https://huggingface.co/collections/allenai/molmo2-data)

## License

MIT License - See LICENSE file for details.

---

**Educational Use Only** | Built with ❤️ for learning VLM architectures
