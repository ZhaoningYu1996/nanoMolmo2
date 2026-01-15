# nanoMolmo2 Implementation Plan

**Pure PyTorch Implementation - Step by Step Guide**

---

## Your Requirements

✅ **Qwen3-0.6B** as language model  
✅ **Frozen vision encoder** (Molmo2's CLIP ViT)  
✅ **Pure PyTorch** (minimal third-party libraries)  
✅ **Educational focus** (understand everything)  

---

## Implementation Phases

### Phase 1: Core Components (Week 1-2)

#### 1.1 Vision Encoder ✏️
**File**: `models/vision_encoder.py`

**Options**:
- **Option A**: Implement ViT from scratch (~200 lines)
  - Best for learning
  - Full control
  - See `PURE_PYTORCH_GUIDE.md` for code

- **Option B**: Load CLIP weights into custom ViT
  - Download weights manually
  - Load into your PyTorch ViT
  - Still pure PyTorch!

**Recommendation**: Start with Option A, load weights later

**Tasks**:
```python
[ ] Implement PatchEmbedding
[ ] Implement TransformerBlock
[ ] Implement complete VisionTransformer
[ ] Add freeze() method
[ ] Test forward pass
[ ] Verify output shapes: [B, 577, 1024]
```

#### 1.2 Language Model ✏️
**File**: `models/language_model.py`

**Implementation**: Transformer Decoder (~300 lines)

**Tasks**:
```python
[ ] Implement RotaryEmbedding (RoPE)
[ ] Implement TransformerDecoderBlock
[ ] Implement complete LanguageModel
[ ] Add causal masking
[ ] Test forward pass
[ ] Verify output shapes: [B, L, vocab_size]
```

**Note**: Start with simpler model, can load Qwen weights later

#### 1.3 Connector ✏️
**File**: `models/connector.py`

**Implementation**: Simple linear projection (~10 lines)

**Tasks**:
```python
[ ] Implement Linear connector (1024 → 896)
[ ] Optional: Implement MLP connector
[ ] Test projection
```

#### 1.4 Complete Model ✏️
**File**: `models/nanomolmo2.py`

**Tasks**:
```python
[ ] Combine all components
[ ] Implement forward pass
[ ] Handle image-text merging
[ ] Add loss computation
[ ] Test end-to-end
```

---

### Phase 2: Data Pipeline (Week 3)

#### 2.1 Tokenizer ✏️
**File**: `data/tokenizer.py`

**Options**:
- **Simple**: Character-level tokenizer (for testing)
- **Better**: BPE tokenizer from scratch
- **Best**: Load Qwen tokenizer (small dependency)

**Tasks**:
```python
[ ] Implement encode() method
[ ] Implement decode() method
[ ] Add special tokens (<image>, <pad>, etc.)
[ ] Test tokenization
```

#### 2.2 Image Datasets ✏️
**File**: `data/image_datasets.py`

**Implementation**: Pure PyTorch Dataset classes

**Tasks**:
```python
[ ] Implement CaptioningDataset
[ ] Implement PointingDataset
[ ] Implement VQADataset
[ ] Add image transforms (torchvision)
[ ] Test data loading
```

**Data Format** (JSONL):
```json
{"image_path": "img.jpg", "caption": "A cat..."}
```

#### 2.3 Video Datasets ✏️
**File**: `data/video_datasets.py`

**Implementation**: Pure PyTorch Dataset with OpenCV

**Tasks**:
```python
[ ] Implement frame sampling (2 fps, max 128 frames)
[ ] Implement VideoCaptioningDataset
[ ] Implement VideoQADataset
[ ] Test video loading
```

#### 2.4 Collator ✏️
**File**: `data/collator.py`

**Implementation**: Custom batching logic

**Tasks**:
```python
[ ] Implement padding
[ ] Implement image-text token merging
[ ] Add task-specific weights
[ ] Test batching
```

---

### Phase 3: Training Loop (Week 4)

#### 3.1 Trainer ✏️
**File**: `training/trainer.py`

**Implementation**: Custom training loop (~200 lines)

**Tasks**:
```python
[ ] Implement train_step()
[ ] Implement validation_step()
[ ] Add gradient accumulation
[ ] Add gradient clipping
[ ] Implement checkpointing
[ ] Add logging (TensorBoard)
```

#### 3.2 Optimizer & Scheduler ✏️
**File**: `training/optimizer.py`

**Tasks**:
```python
[ ] Set up AdamW optimizer
[ ] Implement cosine LR scheduler
[ ] Add warmup schedule
[ ] Test LR scheduling
```

#### 3.3 Training Scripts ✏️
**Files**: `scripts/train_stage*.py`

**Tasks**:
```python
[ ] Stage 1: Pre-training script
[ ] Stage 2: SFT script
[ ] Stage 3: Long-context script
[ ] Add command-line arguments
[ ] Test each stage
```

---

### Phase 4: Testing & Debugging (Week 5)

#### 4.1 Unit Tests ✏️

**Tasks**:
```python
[ ] Test vision encoder shapes
[ ] Test language model shapes
[ ] Test dataset loading
[ ] Test tokenizer
[ ] Test training step
```

#### 4.2 Integration Tests ✏️

**Tasks**:
```python
[ ] Test end-to-end forward pass
[ ] Test backward pass
[ ] Test memory usage
[ ] Profile speed
```

#### 4.3 Small-Scale Training ✏️

**Tasks**:
```python
[ ] Train on 100 samples
[ ] Verify loss decreases
[ ] Check memory usage
[ ] Measure training speed
```

---

### Phase 5: Full Training (Week 6+)

#### 5.1 Stage 1: Pre-training

**Config**:
```yaml
batch_size: 32
learning_rate: 1e-4
max_steps: 100000
data: 60% caption, 30% pointing, 10% NLP
```

**Tasks**:
```python
[ ] Download pre-training datasets
[ ] Start training
[ ] Monitor loss
[ ] Save checkpoints every 5K steps
```

**Time**: ~5-7 days on 4× A100 40GB

#### 5.2 Stage 2: SFT

**Config**:
```yaml
batch_size: 16
learning_rate: 5e-5
max_steps: 50000
data: 100+ datasets, sqrt-proportional
```

**Tasks**:
```python
[ ] Load Stage 1 checkpoint
[ ] Download SFT datasets
[ ] Start training
[ ] Monitor metrics
```

**Time**: ~2-3 days on 4× A100 40GB

#### 5.3 Stage 3: Long-context

**Config**:
```yaml
batch_size: 4
learning_rate: 3e-5
max_steps: 2000
seq_length: 36864
frames: 384
```

**Tasks**:
```python
[ ] Load Stage 2 checkpoint
[ ] Set up Context Parallelism
[ ] Start training
```

**Time**: ~4-6 hours on 4× A100 80GB

---

## Minimal Dependencies

```bash
# Core (REQUIRED)
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
opencv-python>=4.8.0

# Optional (NICE TO HAVE)
tensorboard>=2.14.0
tqdm>=4.66.0
pyarrow>=14.0.0

# NOT NEEDED
# - transformers (we implement from scratch)
# - datasets (we implement from scratch)
# - accelerate (we write training loop)
```

**Total**: 5 core packages!

---

## Project Structure

```
nanoMolmo2/
├── models/
│   ├── vision_encoder.py     ← Week 1
│   ├── language_model.py     ← Week 1-2
│   ├── connector.py           ← Week 1
│   └── nanomolmo2.py         ← Week 2
│
├── data/
│   ├── tokenizer.py          ← Week 3
│   ├── image_datasets.py     ← Week 3
│   ├── video_datasets.py     ← Week 3
│   └── collator.py           ← Week 3
│
├── training/
│   ├── trainer.py            ← Week 4
│   ├── optimizer.py          ← Week 4
│   └── utils.py              ← Week 4
│
└── scripts/
    ├── train_stage1.py       ← Week 4
    ├── train_stage2.py       ← Week 4
    └── train_stage3.py       ← Week 4
```

---

## Quick Start (After Implementation)

```bash
# 1. Install minimal dependencies
pip install -r requirements_minimal.txt

# 2. Download datasets
bash scripts/download_all.sh

# 3. Train Stage 1
python scripts/train_stage1.py \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --max-steps 100000

# 4. Train Stage 2
python scripts/train_stage2.py \
    --batch-size 16 \
    --resume-from checkpoints/stage1/final.pt

# 5. Train Stage 3
python scripts/train_stage3.py \
    --batch-size 4 \
    --resume-from checkpoints/stage2/final.pt
```

---

## Learning Resources

### For Vision Transformer
- **Paper**: "An Image is Worth 16x16 Words" (ViT paper)
- **Code**: `examples/minimal_pure_pytorch.py`
- **Guide**: `PURE_PYTORCH_GUIDE.md`

### For Transformer Decoder
- **Paper**: "Attention is All You Need"
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Code**: `examples/minimal_pure_pytorch.py`

### For Training
- **Mixed Precision**: PyTorch AMP documentation
- **Distributed**: PyTorch DDP documentation
- **Code**: `training/trainer.py`

---

## Milestones

### Milestone 1: Hello World (Week 1)
```python
✓ Vision encoder forward pass works
✓ Language model forward pass works
✓ Loss computation works
✓ Can overfit on 10 samples
```

### Milestone 2: Data Pipeline (Week 3)
```python
✓ Can load images from disk
✓ Can tokenize text
✓ Can create batches
✓ DataLoader works
```

### Milestone 3: Training Loop (Week 4)
```python
✓ Forward + backward pass works
✓ Gradients computed correctly
✓ Can train for 100 steps
✓ Loss decreases
```

### Milestone 4: Small-Scale Training (Week 5)
```python
✓ Train on 1K samples for 1K steps
✓ Model learns something
✓ Can save/load checkpoints
✓ Memory usage acceptable
```

### Milestone 5: Full Training (Week 6+)
```python
✓ Stage 1 training complete
✓ Stage 2 training complete
✓ Stage 3 training complete
✓ Model can do basic VLM tasks
```

---

## Debugging Tips

### Check Shapes Everywhere
```python
print(f"Vision output: {visual_features.shape}")  # [B, 577, 1024]
print(f"LLM output: {logits.shape}")  # [B, L, vocab_size]
```

### Monitor Memory
```python
print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Start Small
```python
# Test with tiny model first
model = NanoMolmo2(
    vision_dim=128,  # Instead of 1024
    llm_dim=128,     # Instead of 896
)
```

### Use Gradient Checkpointing
```python
model.llm.gradient_checkpointing_enable()
```

---

## Success Criteria

**Phase 1-4**: Code works, tests pass  
**Phase 5**: Loss decreases, model learns  
**Final**: Model can caption images and answer visual questions  

---

## Next Steps

1. **Start with minimal example**:
   ```bash
   python examples/minimal_pure_pytorch.py
   ```

2. **Read implementation guide**:
   - [PURE_PYTORCH_GUIDE.md](./PURE_PYTORCH_GUIDE.md)
   - [MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)

3. **Implement Phase 1** (Vision + Language models)

4. **Test everything** before moving to Phase 2

---

**Timeline**: 6-8 weeks (part-time) or 3-4 weeks (full-time)  
**Dependencies**: 5 core packages (pure PyTorch!)  
**Lines of code**: ~2000 lines total  
**Learning**: Maximum! 🎓
