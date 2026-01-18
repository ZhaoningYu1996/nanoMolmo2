# NanoMolmo2 Model Components

Pure PyTorch implementation of a Vision-Language Model (VLM) based on the Molmo2 architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NanoMolmo2 Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘

   Input Image (384×384)
          │
          ▼
┌──────────────────────────┐
│    Vision Encoder        │  SigLIP 2 So400m/14 @ 384px
│    (413M params)         │  
│    🔒 FROZEN             │  Output: [B, 729, 1152]
└──────────────────────────┘  (729 patches × 1152 dim)
          │
          ▼
┌──────────────────────────┐
│    Connector             │  Linear/MLP/Resampler
│    (~1-4M params)        │  
│    ✏️  TRAINABLE         │  Output: [B, 729, 1024]
└──────────────────────────┘  (729 tokens × LLM dim)
          │
          ▼
   Visual Tokens ────────────────┐
                                 │
   Text Tokens ──────────────────┼──▶ Concatenated Sequence
          │                      │
          ▼                      │
┌──────────────────────────┐     │
│    Token Embedding       │     │
│    (Qwen3 vocab)         │     │
└──────────────────────────┘     │
          │                      │
          ▼                      ▼
┌──────────────────────────────────────────────────────┐
│              Language Model (LLM Decoder)             │
│              Qwen3-0.6B-Base                          │
│              (596M params, 28 layers)                 │
│              ✏️  TRAINABLE                            │
│                                                       │
│  Features:                                            │
│  • Grouped Query Attention (16 Q heads, 8 KV heads)  │
│  • Rotary Position Embeddings (RoPE)                  │
│  • Q/K Normalization (Qwen3-specific)                 │
│  • SwiGLU MLP activation                              │
└──────────────────────────────────────────────────────┘
          │
          ▼
   Output Logits [B, seq_len, 151936]
```

---

## Components

### 1. Vision Encoder (`vision_encoder.py`)

**Model**: SigLIP 2 So400m/14 @ 384px  
**Parameters**: 413M (frozen during training)  
**Source**: [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)

#### Architecture

| Component | Value |
|-----------|-------|
| Image Size | 384 × 384 |
| Patch Size | 14 × 14 |
| Num Patches | 729 (27 × 27) |
| Hidden Dim | 1152 |
| MLP Dim | 4304 |
| Layers | 27 |
| Attention Heads | 16 |

#### Data Flow

```
Input: [B, 3, 384, 384] RGB image
         │
         ▼
┌─────────────────────┐
│  Patch Embedding    │  Conv2d(3, 1152, kernel=14, stride=14)
│  + Position Embed   │  Learnable position embeddings
└─────────────────────┘
         │
         ▼  [B, 729, 1152]
┌─────────────────────┐
│  Transformer Layers │  27 × EncoderLayer
│  (Pre-Norm)         │  LayerNorm → Attention → LayerNorm → MLP
└─────────────────────┘
         │
         ▼  [B, 729, 1152]
┌─────────────────────┐
│  Post LayerNorm     │
└─────────────────────┘
         │
         ▼
Output: [B, 729, 1152] visual features
```

#### Usage

```python
from models import VisionEncoder

# Load from local cache (fast) or HuggingFace (slow)
encoder = VisionEncoder.from_pretrained()
encoder.freeze()  # Freeze for VLM training

# Forward pass
images = torch.randn(2, 3, 384, 384)
features = encoder(images)  # [2, 729, 1152]
```

---

### 2. Language Model (`language_model.py`)

**Model**: Qwen3-0.6B-Base  
**Parameters**: 596M (trained during VLM training)  
**Source**: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

#### Architecture

| Component | Value |
|-----------|-------|
| Hidden Dim | 1024 |
| MLP Dim | 3072 |
| Layers | 28 |
| Q Heads | 16 |
| KV Heads | 8 (GQA) |
| Head Dim | 128 (explicit) |
| Vocab Size | 151,936 |
| Max Context | 32,768 |
| RoPE θ | 1,000,000 |

#### Key Features (Qwen3-specific)

1. **Grouped Query Attention (GQA)**: 16 query heads share 8 key-value heads
2. **Explicit Head Dim**: 128 (larger than hidden_size/num_heads = 64)
3. **Q/K Normalization**: RMSNorm applied to Q and K before RoPE
4. **No Attention Bias**: All projection layers have `bias=False`

#### Data Flow

```
Input: [B, L] token IDs
         │
         ▼
┌─────────────────────┐
│  Token Embedding    │  [B, L] → [B, L, 1024]
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Position IDs       │  Generate [0, 1, 2, ..., L-1]
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Decoder Layers (×28)                                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Input Norm (RMSNorm)                                    ││
│  │     ↓                                                   ││
│  │ Self-Attention (GQA + SDPA)                             ││
│  │   • Q: 1024 → 2048 (16 heads × 128 dim)                ││
│  │   • K: 1024 → 1024 (8 heads × 128 dim)                 ││
│  │   • V: 1024 → 1024 (8 heads × 128 dim)                 ││
│  │   • Q/K Norm → RoPE → SDPA → O Proj                    ││
│  │     ↓ (+ residual)                                      ││
│  │ Post-Attention Norm (RMSNorm)                           ││
│  │     ↓                                                   ││
│  │ MLP (SwiGLU)                                            ││
│  │   • gate_proj: 1024 → 3072                             ││
│  │   • up_proj:   1024 → 3072                             ││
│  │   • down_proj: 3072 → 1024                             ││
│  │   • output = down(SiLU(gate) * up)                     ││
│  │     ↓ (+ residual)                                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Final Norm         │  RMSNorm
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  LM Head            │  Linear(1024 → 151936)
│  (tied weights)     │  Shares weights with embedding
└─────────────────────┘
         │
         ▼
Output: [B, L, 151936] logits
```

#### Usage

```python
from models import LanguageModel

# Load from local cache (fast) or HuggingFace (slow)
llm = LanguageModel.from_pretrained()

# Forward pass
input_ids = torch.randint(0, 1000, (2, 64))
logits = llm(input_ids)  # [2, 64, 151936]

# With KV cache for generation
logits, cache = llm(input_ids, use_cache=True)
next_logits, cache = llm(next_token, past_key_values=cache, use_cache=True)
```

---

### 3. Connector (`connector.py`)

**Purpose**: Project vision features to LLM embedding space  
**Parameters**: 1-4M (trained during VLM training)

#### Options

| Type | Parameters | Description |
|------|------------|-------------|
| `linear` | ~1M | Simple linear projection (fastest) |
| `mlp` | ~4M | 2-layer MLP with GELU (more capacity) |
| `resampler` | ~8M | Cross-attention based (reduces tokens) |

#### Data Flow (Linear)

```
Input: [B, 729, 1152] vision features
         │
         ▼
┌─────────────────────┐
│  Linear Projection  │  Linear(1152 → 1024)
└─────────────────────┘
         │
         ▼
Output: [B, 729, 1024] visual tokens (ready for LLM)
```

#### Data Flow (MLP)

```
Input: [B, 729, 1152] vision features
         │
         ▼
┌─────────────────────┐
│  FC1               │  Linear(1152 → 2048)
│  GELU              │
│  FC2               │  Linear(2048 → 1024)
└─────────────────────┘
         │
         ▼
Output: [B, 729, 1024] visual tokens
```

#### Usage

```python
from models import MultimodalConnector, ConnectorConfig

# Linear connector
config = ConnectorConfig(
    vision_dim=1152,
    llm_dim=1024,
    connector_type="linear"
)
connector = MultimodalConnector(config)

vision_features = torch.randn(2, 729, 1152)
visual_tokens = connector(vision_features)  # [2, 729, 1024]
```

---

### 4. Complete Model (`nanomolmo2.py`)

**NanoMolmo2**: Combines all components into a complete VLM.

#### Training Mode

```python
from models import NanoMolmo2

model = NanoMolmo2.from_pretrained()

# Forward pass with loss computation
logits, loss = model(
    input_ids=tokens,        # [B, L] text tokens with <image> placeholders
    pixel_values=images,     # [B, 3, 384, 384] RGB images
    labels=labels,           # [B, L] target tokens (-100 for non-loss positions)
)

# Backward pass
loss.backward()
```

#### Inference Mode

```python
# Generation
generated_ids = model.generate(
    prompt_ids,              # Initial prompt tokens
    pixel_values=images,     # Image(s) to condition on
    max_new_tokens=512,
    temperature=0.7,
)
```

#### What Happens Inside

1. **Image Encoding** (frozen):
   ```
   images → VisionEncoder → [B, 729, 1152]
   ```

2. **Projection** (trained):
   ```
   [B, 729, 1152] → Connector → [B, 729, 1024]
   ```

3. **Token Embedding**:
   ```
   input_ids → Embedding → [B, L, 1024]
   ```

4. **Sequence Construction**:
   ```
   Replace <image> tokens with visual tokens
   Final: [B, L + 729 - 1, 1024]
   ```

5. **LLM Forward** (trained):
   ```
   combined_embeds → LLM Decoder → [B, L', 151936]
   ```

6. **Loss Computation**:
   ```
   CrossEntropy(logits, labels, ignore_index=-100)
   ```

---

## Weight Loading

All models support efficient weight loading:

```bash
# Download weights once (recommended)
python scripts/download_model_weights.py

# Creates:
#   checkpoints/siglip2_so400m_384.pt (1.6 GB)
#   checkpoints/qwen3_0.6b_base.pt (1.1 GB)
```

```python
# Load from local cache (fast, ~5-10s)
vision = VisionEncoder.from_pretrained()
llm = LanguageModel.from_pretrained()

# Or load from HuggingFace (slow, ~30s)
vision = VisionEncoder.from_pretrained(cache_dir=None)
llm = LanguageModel.from_pretrained(cache_dir=None)
```

---

## Training Configuration

### Frozen vs Trainable

| Component | Trainable | Parameters | Gradient Memory |
|-----------|-----------|------------|-----------------|
| Vision Encoder | ❌ | 413M | 0 |
| Connector | ✅ | ~1M | ~4 MB |
| Language Model | ✅ | 596M | ~2.4 GB |
| **Total** | - | **1,010M** | **~2.4 GB** |

### Precision (per Molmo2 tech report)

| Operation | Precision |
|-----------|-----------|
| Most computation | bfloat16 |
| LayerNorm | float32 |
| RoPE | float32 |

---

## File Structure

```
models/
├── __init__.py           # Exports all components
├── README.md             # This file
├── vision_encoder.py     # SigLIP2 implementation
├── language_model.py     # Qwen3-0.6B implementation
├── connector.py          # Vision-to-LLM projection
└── nanomolmo2.py         # Complete VLM
```

---

## References

- [Molmo2 Tech Report](https://www.datocms-assets.com/64837/1766008501-molmo2-tech-report.pdf) - Architecture and training details
- [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384) - Vision encoder
- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) - Language model
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
