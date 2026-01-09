# nanoMolmo2

A minimal implementation of Molmo2 Vision-Language Model (VLM) from scratch for educational purposes.

## Overview

**nanoMolmo2** is an educational reimplementation of the [Molmo2](https://molmo.allenai.org/) Vision-Language Model, designed to help developers understand modern VLM architectures. This project uses **Qwen3-0.6B** as the base language model while following Molmo2's architecture and training methodology.

> ⚠️ **Note**: This is an educational project for learning purposes only, not intended for production use.

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

### Quick Start

## References

- [Molmo2 Technical Report](https://molmo.allenai.org/)
- [Qwen3 Model](https://github.com/QwenLM/Qwen3)

## Learning Resources

This project is designed for educational exploration of:
- Vision-language model architectures
- Multimodal pretraining strategies
- Efficient training on limited GPU resources
- Distributed training with PyTorch

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Allen Institute for AI for the Molmo2 architecture and methodology
- Alibaba Cloud for the Qwen3 base model
- The open-source ML community

---

**Educational Use Only** | Built with ❤️ for learning VLM architectures
