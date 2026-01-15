# nanoMolmo2

**🎓 A Vision-Language Model (VLM) Learning Project**

A minimal implementation of Molmo2 VLM from scratch for educational purposes.

> **⚡ QUICK START**: See [START_HERE.md](./START_HERE.md) to download all datasets in 2 commands!
> 
> **📖 FULL GUIDE**: See [DOWNLOAD_GUIDE.md](./DOWNLOAD_GUIDE.md) for complete download instructions.

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

## Training Data

Following Molmo2's data mixture:
- Image-text pairs
- Instruction-following datasets
- Visual reasoning tasks
- Detailed captions

Refer to the [Molmo2 technical report](https://molmo.allenai.org/) for complete data composition.

## Training Setup

**Hardware Requirements**:
- 2-4 NVIDIA A100 GPUs (40GB or 80GB)
- Distributed training with DeepSpeed/FSDP

**Training Flow**:
1. Vision encoder pretraining (optional, can use pretrained)
2. Connector alignment phase
3. Joint multimodal training

## Project Structure

```
nanoMolmo2/
├── models/           # Model architectures
├── data/            # Data loading and preprocessing
├── training/        # Training scripts and configs
├── configs/         # Model and training configurations
├── scripts/         # Utility scripts
└── README.md
```

## Getting Started

### Prerequisites

**Pure PyTorch Implementation** (Minimal Dependencies):

```bash
# Core requirements only (5 packages)
pip install -r requirements_minimal.txt

# This installs:
# - torch, torchvision (deep learning)
# - numpy, pillow (data processing)
# - opencv-python (video processing)
```

**OR Full Requirements** (with HuggingFace tools):

```bash
# If you want to use pre-trained models and HF datasets
pip install -r requirements.txt
```

**We implement everything from scratch using pure PyTorch for maximum learning!**  
See [PURE_PYTORCH_GUIDE.md](./PURE_PYTORCH_GUIDE.md) for details.

### Quick Start

**👉 See [START_HERE.md](./START_HERE.md) for the simplest guide!**

**Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2: Download datasets**

```bash
# ONE COMMAND - Download all ~30 datasets from Molmo2 paper
bash scripts/download_all.sh

# OR use Python script for more control
python scripts/download_datasets.py --all                # Everything (~150GB)
python scripts/download_datasets.py --stage pretraining  # Just pre-training (~60GB)
python scripts/download_datasets.py --molmo2-only        # Just Molmo2 datasets (~30GB)
python scripts/download_datasets.py --list               # See all available
```

**Time**: 4-8 hours | **Size**: ~150GB | **Datasets**: ~30 from HuggingFace

See [DOWNLOAD_GUIDE.md](./DOWNLOAD_GUIDE.md) for detailed instructions.

**Step 3: Verify downloads**

```bash
# List downloaded datasets
ls data/molmo2_datasets/

# Inspect specific dataset
python scripts/inspect_molmo2_data.py molmo2-cap --num-samples 5

# Test dataloader
python examples/train_with_stage_dataloaders.py --stage 1 --inspect
```

**Step 4: Train**

```bash
# Stage 1: Pre-training (60% caption, 30% pointing, 10% NLP)
python examples/train_with_stage_dataloaders.py --stage 1 --batch-size 32

# Stage 2: SFT (100+ datasets with sqrt-proportional sampling)
python examples/train_with_stage_dataloaders.py --stage 2 --batch-size 16

# Stage 3: Long-context (36,864 tokens, 384 frames)
python examples/train_with_stage_dataloaders.py --stage 3 --batch-size 4

# Or train all stages sequentially
python examples/train_with_stage_dataloaders.py --all-stages
```

**See [QUICKSTART.md](./QUICKSTART.md) for detailed instructions.**

**Step 0 (Optional): Verify your setup**

```bash
# Check that model components can load and estimate memory
python scripts/verify_model_setup.py
```

This will show you:
- ✓ Vision encoder loads correctly
- ✓ Qwen3-0.6B LLM loads correctly
- ✓ Memory estimates (~20GB per GPU)
- ✓ Trainable parameter count (~500M)

## 📚 Documentation

### ⭐ New Documentation (2026-01-15)

- **[YOUR_SETUP.md](./YOUR_SETUP.md)** - 🎯 **YOUR SPECIFIC SETUP** - Frozen encoder + Qwen3-0.6B configuration
- **[MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)** - Complete architecture guide with code
- **[SUMMARY.md](./SUMMARY.md)** - ✨ Complete implementation summary and next steps
- **[MOLMO2_TECH_REPORT_SUMMARY.md](./MOLMO2_TECH_REPORT_SUMMARY.md)** - Comprehensive Molmo2 paper summary (30+ pages)
- **[TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md)** - Visual guide to 3-stage training pipeline
- **[QUICKSTART.md](./QUICKSTART.md)** - Complete guide from setup to training
- **[START_HERE.md](./START_HERE.md)** - Quickest 2-command download guide
- **[DOWNLOAD_GUIDE.md](./DOWNLOAD_GUIDE.md)** - Complete dataset download guide

### Core Documentation

- **[data/stage_dataloaders.py](./data/stage_dataloaders.py)** - ⭐ NEW: Stage-specific dataloaders implementation
- **[examples/train_with_stage_dataloaders.py](./examples/train_with_stage_dataloaders.py)** - ⭐ NEW: Training examples
- **[data/README.md](./data/README.md)** - Data pipeline overview
- **[data/dataloaders/README.md](./data/dataloaders/README.md)** - Detailed dataloader documentation (70+ pages)

### Training Pipeline

The training follows Molmo2's 3-stage approach:

```
Stage 1: Pre-training (Image-only)
├── 60% Dense captioning (PixMo-Cap)
├── 30% Image pointing (PixMo-Points, PixMo-Count)
└── 10% NLP data (Tulu)
    ↓
    Steps: ~100K | Seq: 4,096 tokens | Hardware: 16-32 A100s
    
Stage 2: Supervised Fine-Tuning
├── Molmo2 datasets (video cap, QA, pointing, tracking)
├── PixMo datasets (image cap, QA, pointing)
└── Academic datasets (VQA, DocVQA, ChartQA, ...)
    ↓
    Steps: ~50K | Seq: 4,096 tokens | Frames: 128 | Hardware: 8-16 A100s
    
Stage 3: Long-Context SFT
├── Same datasets as Stage 2
└── Extended sequences and frames
    ↓
    Steps: 2K | Seq: 36,864 tokens | Frames: 384 | Hardware: 16-32 A100s
```

### Datasets

**9 New Molmo2 Datasets** (released by Ai2):
- `Molmo2-Cap` - Dense video captioning
- `Molmo2-AskModelAnything` - Human-authored video QA
- `Molmo2-VideoCapQA` - Synthetic video QA
- `Molmo2-VideoSubtitleQA` - Video QA with subtitles
- `Molmo2-VideoPoint` - Video temporal pointing (NOVEL)
- `Molmo2-VideoTrack` - Video object tracking (NOVEL)
- `Molmo2-MultiImageQA` - Multi-image QA
- `Molmo2-SynMultiImageQA` - Synthetic multi-image QA
- `Molmo2-MultiImagePoint` - Multi-image pointing

**Plus**: PixMo datasets, 100+ academic datasets (see `scripts/download_molmo2_data_v2.py`)

All available on HuggingFace: https://huggingface.co/collections/allenai/molmo2-data

## References

- [Molmo2 Blog Post](https://allenai.org/blog/molmo2)
- [Molmo2 Technical Report](https://molmo.allenai.org/)
- [Qwen3 Model](https://github.com/QwenLM/Qwen3)
- [HuggingFace Molmo2 Collection](https://huggingface.co/collections/allenai/molmo2-data)

## Learning Resources

**This project is specifically designed for learning Vision-Language Models (VLMs)**, covering:

- **VLM Architecture Fundamentals**: Understanding how vision encoders connect with language models
- **Multimodal Fusion**: How images and text are processed together
- **Vision-Language Pretraining**: Strategies for training models on multimodal data
- **Efficient VLM Training**: Training large multimodal models on limited GPU resources (2-4 A100s)
- **Distributed Training**: Practical experience with PyTorch FSDP/DeepSpeed for VLMs

💡 **Perfect for**: ML engineers wanting to understand VLM internals, researchers exploring multimodal architectures, students learning modern AI systems.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Allen Institute for AI for the Molmo2 architecture and methodology
- Alibaba Cloud for the Qwen3 base model
- The open-source ML community

---

**Educational Use Only** | Built with ❤️ for learning VLM architectures
