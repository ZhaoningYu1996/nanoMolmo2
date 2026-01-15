# Your nanoMolmo2 Setup

**Architecture**: Frozen Vision Encoder + Qwen3-0.6B LLM

---

## ✅ Your Configuration

### Model Architecture

```
┌──────────────────────────────────────────────────────┐
│              nanoMolmo2 ARCHITECTURE                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Input: Images/Videos                                │
│         ↓                                            │
│  ┌────────────────────┐                              │
│  │ Vision Encoder     │  Molmo2's CLIP ViT          │
│  │ ~300M params       │  336×336 → 1024-dim         │
│  │ 🔒 FROZEN          │  (No gradients!)            │
│  └────────────────────┘                              │
│         ↓                                            │
│  ┌────────────────────┐                              │
│  │ Connector          │  Linear projection          │
│  │ ~1M params         │  1024-dim → 896-dim         │
│  │ ✏️  TRAINABLE      │                             │
│  └────────────────────┘                              │
│         ↓                                            │
│  ┌────────────────────┐                              │
│  │ Qwen3-0.6B LLM     │  Language generation        │
│  │ ~500M params       │  896-dim hidden             │
│  │ ✏️  TRAINABLE      │  32K context length         │
│  └────────────────────┘                              │
│         ↓                                            │
│  Output: Text                                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Component Details

| Component | Model | Parameters | Trainable? | Memory |
|-----------|-------|------------|------------|--------|
| Vision Encoder | CLIP ViT-L/14@336px | 307M | 🔒 **NO** | 1.2 GB |
| Connector | Linear projection | 1M | ✏️ **YES** | 50 MB |
| Language Model | Qwen3-0.6B | 494M | ✏️ **YES** | 15 GB |
| **Total** | | **802M** | **495M** | **~20 GB** |

---

## 🎯 Why This Setup?

### 1. Memory Efficient
- **Frozen encoder**: No gradients, no optimizer states
- **Saves ~10 GB** per GPU
- **Fits in A100 40GB** with batch_size=32

### 2. Faster Training
- **Skip vision backward pass** → 30-40% speedup
- **More iterations per day**
- **Faster experimentation**

### 3. Good Performance
- **Pre-trained CLIP features** are already excellent
- **Focus learning** on language understanding
- **Proven approach**: LLaVA, MiniGPT-4 use frozen encoders

### 4. Educational Value
- **Simpler to understand** (fewer moving parts)
- **Faster to train** (learn VLM principles quickly)
- **Cost effective** (runs on cheaper GPUs)

---

## 📊 Resource Requirements

### Hardware (Recommended)

**Minimum**:
- 2× NVIDIA A100 40GB GPUs
- 64 GB RAM
- 200 GB storage (datasets) + 50 GB (checkpoints)

**Recommended**:
- 4× NVIDIA A100 40GB GPUs
- 128 GB RAM
- 500 GB storage

**Can also use**:
- 2× A100 80GB
- 4-8× RTX 4090 24GB (with smaller batch sizes)

### Training Time (on 4× A100 40GB)

| Stage | Steps | Time |
|-------|-------|------|
| Pre-training | 100K | 5-7 days |
| SFT | 50K | 2-3 days |
| Long-context | 2K | 4-6 hours |
| **Total** | **152K** | **~10 days** |

**With frozen encoder**: ~30% faster than unfrozen!

---

## 🚀 Getting Started

### Step 1: Verify Model Can Load

```bash
# Install dependencies
pip install -r requirements.txt

# Verify model setup
python scripts/verify_model_setup.py
```

This will:
- ✓ Load CLIP vision encoder
- ✓ Load Qwen3-0.6B LLM
- ✓ Create connector
- ✓ Show memory estimates
- ✓ Verify everything works!

Expected output:
```
Vision Encoder:     307.3M  (🔒 FROZEN)
Connector:            0.9M  (✏️  TRAINABLE)
Language Model:     494.0M  (✏️  TRAINABLE)
────────────────────────────
Total:              802.2M
Trainable:          494.9M (61.7%)

Estimated total: ~20 GB
✓ Fits in A100 40GB!
```

### Step 2: Download Datasets

```bash
# Download all datasets (~150GB, 4-8 hours)
bash scripts/download_all.sh

# Or just essentials (~60GB, 2-4 hours)
python scripts/download_datasets.py --stage pretraining
```

### Step 3: Implement Model

See [MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md) for:
- Complete model implementation code
- Forward pass details
- Training loop structure

### Step 4: Start Training

```bash
# Stage 1: Pre-training
python examples/train_with_stage_dataloaders.py --stage 1 --batch-size 32

# Stage 2: SFT
python examples/train_with_stage_dataloaders.py --stage 2 --batch-size 16

# Stage 3: Long-context
python examples/train_with_stage_dataloaders.py --stage 3 --batch-size 4
```

---

## 📁 Configuration Files

All your settings are in `config/`:

- **`model_config.yaml`** - Model architecture settings
- **`train_config.yaml`** - Training hyperparameters for all 3 stages

Key settings:
```yaml
# Model
freeze_vision_encoder: true  # ← Your setup!
train_connector: true
train_llm: true

# Stage 1 Training
batch_size_per_gpu: 32
learning_rate: 1.0e-4
max_steps: 100000

# Memory
use_gradient_checkpointing: true
use_flash_attention: true
```

---

## 🔧 Fine-Tuning Options

### If You Have More GPUs

You can:
- **Increase batch size**: 32 → 64 per GPU
- **Train longer**: 100K → 150K steps
- **Use larger connector**: Linear → MLP

### If You Want Better Quality

Consider:
- **Unfreeze vision encoder** (requires 8× A100 80GB)
- **Use larger LLM** (Qwen-1.8B instead of 0.6B)
- **Train longer** on each stage

### If Memory is Tight

You can:
- **Reduce batch size**: 32 → 16
- **Use gradient checkpointing**
- **Use CPU offloading** (slower but works)

---

## 📖 Documentation

- **[MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)** - Complete architecture guide
- **[config/model_config.yaml](./config/model_config.yaml)** - Model configuration
- **[config/train_config.yaml](./config/train_config.yaml)** - Training configuration
- **[QUICKSTART.md](./QUICKSTART.md)** - Training guide
- **[TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md)** - Pipeline visualization

---

## 💡 Tips for Success

### 1. Start Small
- Download just pre-training datasets first
- Train for 1K steps to verify everything works
- Then scale up

### 2. Monitor Memory
```python
# In training loop
import torch
print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### 3. Use Gradient Checkpointing
```python
model.llm.gradient_checkpointing_enable()
```
Saves ~30% memory with ~10% slowdown

### 4. Log Everything
```python
import wandb
wandb.init(project="nanoMolmo2")
wandb.log({"loss": loss, "lr": lr, "step": step})
```

### 5. Save Frequently
```python
# Save every 5K steps
if step % 5000 == 0:
    torch.save(model.state_dict(), f"checkpoint_{step}.pt")
```

---

## ✅ Verification Checklist

Before training:

- [ ] Verified model can load: `python scripts/verify_model_setup.py`
- [ ] Downloaded datasets: `ls data/molmo2_datasets/`
- [ ] Tested dataloader: `python examples/train_with_stage_dataloaders.py --stage 1 --inspect`
- [ ] Checked GPU availability: `nvidia-smi`
- [ ] Reviewed configs: `cat config/train_config.yaml`

After verification:

- [ ] Ready to train! 🚀

---

## 🆘 Need Help?

### Common Issues

**Q: Vision encoder won't load**
```bash
# Fix: Install transformers
pip install transformers
huggingface-cli login  # May need to login
```

**Q: Out of memory**
```python
# Fix: Reduce batch size or enable checkpointing
batch_size = 16  # Instead of 32
model.llm.gradient_checkpointing_enable()
```

**Q: Training too slow**
```python
# Fix: Use fewer workers or disable data augmentation
num_workers = 2  # Instead of 4
```

### Documentation

- Model architecture: [MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)
- Training pipeline: [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md)
- Complete summary: [SUMMARY.md](./SUMMARY.md)

---

**Your Setup**: ✅ Optimized for 2-4 A100 40GB GPUs  
**Training Time**: ~10 days total  
**Memory**: ~20 GB per GPU  
**Trainable Params**: ~500M  

**Ready to train!** 🎓
