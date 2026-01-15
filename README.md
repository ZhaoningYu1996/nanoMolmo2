# nanoMolmo2

**🎓 A Vision-Language Model (VLM) Learning Project**

A minimal implementation of Molmo2 VLM from scratch for educational purposes.

## Overview

**nanoMolmo2** is an educational reimplementation of the [Molmo2](https://molmo.allenai.org/) Vision-Language Model, designed to help developers **learn and understand modern VLM architectures from the ground up**. This hands-on project uses **Qwen3-0.6B** as the base language model while following Molmo2's architecture and training methodology.

🎯 **Primary Goal**: Provide a clear, educational implementation for learning how Vision-Language Models work - from architecture design to multimodal training.

> ⚠️ **Note**: This is a **learning-focused educational project**, not intended for production use.

## Architecture

The model follows the Molmo2 architecture:

- **Base LLM**: Qwen3-0.6B (replacing Molmo2's OLMo base)
- **Vision Encoder**: Same as Molmo2
- **Multimodal Connector**: Same as Molmo2
- **Training Objective**: Same as Molmo2

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

```bash
# Python 3.9+
pip install torch torchvision transformers datasets accelerate opencv-python Pillow

# For distributed training (optional)
pip install deepspeed
```

### Quick Start

**Step 1: Download datasets**

```bash
# Download all datasets for complete training pipeline
python scripts/download_molmo2_data_v2.py --stage all

# Or download specific stage
python scripts/download_molmo2_data_v2.py --stage pretraining
python scripts/download_molmo2_data_v2.py --stage sft
```

**Step 2: Verify data**

```bash
# Inspect downloaded datasets
python scripts/inspect_molmo2_data.py

# Check specific dataset
python scripts/inspect_molmo2_data.py pixmo-cap --num-samples 5
```

**Step 3: Train**

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

## 📚 Documentation

### ⭐ New Documentation (2026-01-15)

- **[SUMMARY.md](./SUMMARY.md)** - ✨ **START HERE** - Complete implementation summary and next steps
- **[MOLMO2_TECH_REPORT_SUMMARY.md](./MOLMO2_TECH_REPORT_SUMMARY.md)** - Comprehensive Molmo2 paper summary (30+ pages)
- **[TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md)** - Visual guide to 3-stage training pipeline
- **[QUICKSTART.md](./QUICKSTART.md)** - Complete guide from setup to training

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
