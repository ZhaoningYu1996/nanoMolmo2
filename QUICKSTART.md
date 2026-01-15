# nanoMolmo2 Quick Start Guide

Complete guide to getting started with nanoMolmo2 training pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Download Datasets](#step-1-download-datasets)
3. [Step 2: Verify Data](#step-2-verify-data)
4. [Step 3: Set Up Training](#step-3-set-up-training)
5. [Step 4: Train Each Stage](#step-4-train-each-stage)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**For Pre-training (Stage 1)**:
- 2-4 NVIDIA A100 GPUs (40GB or 80GB)
- 500GB+ disk space for datasets
- 128GB+ RAM

**For SFT (Stage 2)**:
- 2-4 NVIDIA A100 GPUs
- Same disk/RAM requirements

**For Long-context (Stage 3)**:
- 4-8 NVIDIA A100 GPUs (80GB recommended)
- Requires Context Parallelism (CP)

### Software Requirements

```bash
# Python 3.9+
python --version

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Transformers and datasets
pip install transformers datasets accelerate

# Computer vision libraries
pip install opencv-python Pillow

# Optional: For distributed training
pip install deepspeed
```

---

## Step 1: Download Datasets

### Download All Datasets (Recommended)

Download all datasets for the complete training pipeline:

```bash
cd /home/yu/projects/nanoMolmo2

# Download pre-training + SFT datasets
python scripts/download_molmo2_data_v2.py --stage all

# With evaluation datasets
python scripts/download_molmo2_data_v2.py --stage all --include-eval
```

**Expected download time**: 4-8 hours (depends on network speed)  
**Total size**: ~150GB

### Download by Stage (Optional)

Download datasets for specific training stages:

```bash
# Stage 1: Pre-training datasets only
python scripts/download_molmo2_data_v2.py --stage pretraining

# Stage 2: SFT datasets only
python scripts/download_molmo2_data_v2.py --stage sft

# Evaluation datasets
python scripts/download_molmo2_data_v2.py --stage eval
```

### Download Specific Datasets

Download only specific datasets:

```bash
# Download only PixMo-Cap for testing
python scripts/download_molmo2_data_v2.py \
    --stage pretraining \
    --datasets pixmo-cap

# Download multiple specific datasets
python scripts/download_molmo2_data_v2.py \
    --stage sft \
    --datasets molmo2-cap molmo2-capqa pixmo-cap
```

### Check Manual Download Instructions

Some datasets require manual download:

```bash
# Show manual download instructions
python scripts/download_molmo2_data_v2.py --show-manual-instructions

# Show for specific stage
python scripts/download_molmo2_data_v2.py \
    --stage pretraining \
    --show-manual-instructions
```

---

## Step 2: Verify Data

### Inspect Downloaded Datasets

Use the inspection script to verify downloads:

```bash
# List all downloaded datasets
python scripts/inspect_molmo2_data.py

# Inspect specific dataset
python scripts/inspect_molmo2_data.py pixmo-cap

# Show more samples
python scripts/inspect_molmo2_data.py pixmo-cap --num-samples 10

# Inspect video dataset
python scripts/inspect_molmo2_data.py molmo2-videocapqa --num-samples 5
```

### Check Dataset Structure

Expected directory structure after download:

```
data/molmo2_datasets/
├── pixmo-cap/
│   ├── train.parquet
│   └── validation.parquet
├── pixmo-points/
│   ├── train.parquet
│   └── validation.parquet
├── molmo2-cap/
│   ├── train.parquet
│   └── validation.parquet
├── molmo2-capqa/
│   └── train.parquet
└── ...
```

### Verify Dataloader (Important!)

Before training, test that dataloaders work correctly:

```bash
# Test Stage 1 dataloader
python examples/train_with_stage_dataloaders.py --stage 1 --inspect

# Test Stage 2 dataloader
python examples/train_with_stage_dataloaders.py --stage 2 --inspect

# Test Stage 3 dataloader
python examples/train_with_stage_dataloaders.py --stage 3 --inspect
```

---

## Step 3: Set Up Training

### Quick Test Run

Run a quick test to ensure everything works:

```python
# test_dataloader.py
from transformers import AutoTokenizer
from torchvision import transforms
from data.stage_dataloaders import get_stage_dataloader

# Setup
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
image_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create Stage 1 dataloader
loader = get_stage_dataloader(
    stage=1,
    tokenizer=tokenizer,
    image_processor=image_processor,
    data_root="./data/molmo2_datasets",
    batch_size=2,
    num_workers=0,
)

# Test fetching a batch
batch = next(iter(loader))
print("✓ Dataloader works!")
print(f"Batch keys: {batch.keys()}")
```

Run the test:

```bash
python test_dataloader.py
```

### Configure Training

Create a training configuration file:

```python
# config/train_config.py

STAGE1_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "max_steps": 100000,
    "warmup_steps": 2000,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 4096,
}

STAGE2_CONFIG = {
    "batch_size": 16,
    "learning_rate": 5e-5,
    "num_epochs": 2,
    "max_steps": 50000,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 4096,
    "max_frames": 128,
    "fps": 2.0,
}

STAGE3_CONFIG = {
    "batch_size": 4,
    "learning_rate": 3e-5,
    "num_epochs": 1,
    "max_steps": 2000,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 36864,
    "max_frames": 384,
    "fps": 2.0,
}
```

---

## Step 4: Train Each Stage

### Stage 1: Pre-training

**Goal**: Learn general visual-language alignment

```bash
# Basic training
python examples/train_with_stage_dataloaders.py \
    --stage 1 \
    --batch-size 32 \
    --num-epochs 3

# With custom data root
python examples/train_with_stage_dataloaders.py \
    --stage 1 \
    --data-root /path/to/datasets \
    --batch-size 32
```

**Data Mixture**:
- 60% Dense captioning (PixMo-Cap)
- 30% Image pointing (PixMo-Points, PixMo-Count)
- 10% NLP data (Tulu)

**Configuration**:
- Sequence length: 4,096 tokens
- Batch size: 32 (per GPU)
- Training steps: ~100K
- Expected time: 1-2 weeks on 32 A100s

**Checkpoints**: Save every 5K steps

### Stage 2: Supervised Fine-Tuning

**Goal**: Instruction following and multimodal tasks

```bash
# Load from Stage 1 checkpoint
python examples/train_with_stage_dataloaders.py \
    --stage 2 \
    --batch-size 16 \
    --num-epochs 2 \
    --checkpoint /path/to/stage1/checkpoint
```

**Data Mixture**:
- Sqrt-proportional sampling across 100+ datasets
- Molmo2 datasets (video cap, QA, pointing, tracking)
- PixMo datasets (image cap, QA, pointing)
- Academic datasets (VQA, DocVQA, ChartQA, etc.)

**Configuration**:
- Sequence length: 4,096 tokens
- Max frames: 128 (videos)
- Frame rate: 2 fps
- Batch size: 16 (per GPU)
- Training steps: ~50K
- Expected time: 3-5 days on 16 A100s

**Checkpoints**: Save every 2K steps

### Stage 3: Long-Context SFT

**Goal**: Handle extended video sequences

```bash
# Requires Context Parallelism (CP)
python examples/train_with_stage_dataloaders.py \
    --stage 3 \
    --batch-size 4 \
    --num-epochs 1 \
    --checkpoint /path/to/stage2/checkpoint
```

**Data Mixture**: Same as Stage 2

**Configuration**:
- Sequence length: 36,864 tokens (9x longer)
- Max frames: 384 (3x more)
- Frame rate: 2 fps
- Batch size: 4 (per GPU)
- Training steps: 2K
- Expected time: ~1 day on 32 A100s

**Special Requirements**:
- Context Parallelism (CP) for long sequences
- Ulysses attention implementation
- High-bandwidth GPU interconnect (NVLink)

**Checkpoints**: Save every 500 steps

---

## Training All Stages Sequentially

Run the complete pipeline:

```bash
python examples/train_with_stage_dataloaders.py --all-stages
```

This will:
1. Train Stage 1 (pre-training)
2. Train Stage 2 (SFT) using Stage 1 checkpoint
3. Train Stage 3 (long-context) using Stage 2 checkpoint

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# View in browser
# http://localhost:6006
```

### Weights & Biases

```python
import wandb

wandb.init(
    project="nanoMolmo2",
    name=f"stage{stage}_run",
    config=train_config,
)

# Log during training
wandb.log({
    "loss": loss.item(),
    "learning_rate": lr,
    "step": step,
})
```

---

## Distributed Training

### Multi-GPU Training (Single Node)

```bash
# Using accelerate
accelerate launch --multi_gpu examples/train_with_stage_dataloaders.py \
    --stage 1 \
    --batch-size 32

# Using torchrun
torchrun --nproc_per_node=4 examples/train_with_stage_dataloaders.py \
    --stage 1 \
    --batch-size 32
```

### Multi-Node Training

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    examples/train_with_stage_dataloaders.py \
    --stage 1

# Node 1, 2, 3 (workers)
# Same command with --node_rank=1, 2, 3
```

### DeepSpeed Training

```bash
deepspeed --num_gpus=8 examples/train_with_stage_dataloaders.py \
    --stage 1 \
    --deepspeed_config config/ds_config.json
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:

1. **Reduce batch size**:
   ```bash
   python examples/train_with_stage_dataloaders.py --stage 2 --batch-size 8
   ```

2. **Increase gradient accumulation**:
   ```python
   gradient_accumulation_steps = 16  # Effective batch = 8 × 16 = 128
   ```

3. **Enable gradient checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

4. **Reduce sequence length**:
   ```python
   max_seq_length = 2048  # Instead of 4096
   ```

5. **Reduce video frames**:
   ```python
   max_frames = 64  # Instead of 128
   ```

### Issue: Dataset Not Found

**Check**:

```bash
# Verify datasets are downloaded
ls -lh data/molmo2_datasets/

# Re-download missing dataset
python scripts/download_molmo2_data_v2.py \
    --stage sft \
    --datasets pixmo-cap
```

### Issue: Slow Data Loading

**Solutions**:

1. **Increase num_workers**:
   ```python
   num_workers = 8  # More parallel loading
   ```

2. **Use faster storage (SSD)**
3. **Pre-process videos** to lower resolution
4. **Enable disk caching**

### Issue: Dataloader Error

**Debug**:

```bash
# Inspect dataloader
python examples/train_with_stage_dataloaders.py --stage 1 --inspect

# Check data format
python scripts/inspect_molmo2_data.py pixmo-cap --num-samples 5
```

### Issue: Training Diverges

**Solutions**:

1. **Lower learning rate**:
   ```python
   learning_rate = 1e-5  # Instead of 1e-4
   ```

2. **Increase warmup steps**:
   ```python
   warmup_steps = 5000  # Gradual warmup
   ```

3. **Use gradient clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Check loss weights** (task weighting might be too high)

---

## Next Steps

After completing training:

1. **Evaluate**: Run evaluation on held-out datasets
   ```bash
   python eval/evaluate_model.py --checkpoint /path/to/checkpoint
   ```

2. **Fine-tune**: Further fine-tune on custom datasets

3. **Deploy**: Export model for inference
   ```bash
   python export/export_model.py --checkpoint /path/to/checkpoint
   ```

4. **Benchmark**: Test on standard VLM benchmarks
   - VQA v2.0
   - Video tracking benchmarks
   - Multi-image reasoning tasks

---

## Useful Commands Summary

```bash
# Download all datasets
python scripts/download_molmo2_data_v2.py --stage all

# Inspect dataset
python scripts/inspect_molmo2_data.py pixmo-cap

# Test dataloader
python examples/train_with_stage_dataloaders.py --stage 1 --inspect

# Train Stage 1
python examples/train_with_stage_dataloaders.py --stage 1 --batch-size 32

# Train Stage 2
python examples/train_with_stage_dataloaders.py --stage 2 --batch-size 16

# Train Stage 3
python examples/train_with_stage_dataloaders.py --stage 3 --batch-size 4

# Train all stages
python examples/train_with_stage_dataloaders.py --all-stages
```

---

## Additional Resources

- **Molmo2 Technical Report**: [MOLMO2_TECH_REPORT_SUMMARY.md](./MOLMO2_TECH_REPORT_SUMMARY.md)
- **Dataloader Documentation**: [data/dataloaders/README.md](./data/dataloaders/README.md)
- **Data Pipeline**: [data/README.md](./data/README.md)
- **Molmo2 Blog**: https://allenai.org/blog/molmo2
- **HuggingFace Collection**: https://huggingface.co/collections/allenai/molmo2-data

---

**Last Updated**: 2026-01-15  
**Version**: 1.0  
**Contributors**: nanoMolmo2 Team
