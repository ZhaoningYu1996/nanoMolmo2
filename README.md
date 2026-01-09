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

### Quick Start

## References

- [Molmo2 Technical Report](https://molmo.allenai.org/)
- [Qwen3 Model](https://github.com/QwenLM/Qwen3)

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
