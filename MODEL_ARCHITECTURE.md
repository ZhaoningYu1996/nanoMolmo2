# nanoMolmo2 Model Architecture

**Educational implementation using Qwen3-0.6B LLM with frozen Molmo2 vision encoder**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      nanoMolmo2 ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

Input Image/Video
       ↓
┌──────────────────┐
│ Vision Encoder   │  ← Molmo2's ViT (FROZEN during training)
│ (ViT)            │
│ 🔒 FROZEN        │
└──────────────────┘
       ↓
  Visual Tokens
       ↓
┌──────────────────┐
│ Connector        │  ← Learnable projection (TRAINED)
│ (Linear/MLP)     │
└──────────────────┘
       ↓
  Projected Tokens
       ↓
┌──────────────────┐
│ Language Model   │  ← Qwen3-0.6B (TRAINED)
│ (Qwen3-0.6B)     │
└──────────────────┘
       ↓
   Output Text
```

---

## Component Details

### 1. Vision Encoder (FROZEN 🔒)

**Model**: Same as Molmo2 (OpenAI CLIP ViT or similar)

**Configuration**:
- Architecture: Vision Transformer (ViT)
- Image size: 336×336 (or 224×224)
- Patch size: 14×14
- Embedding dim: 1024
- Layers: 24
- Heads: 16
- Parameters: ~300M (frozen, not updated)

**Why Frozen?**:
✅ **Memory savings**: No gradients needed → 50% less memory  
✅ **Faster training**: Skip vision encoder backward pass → 30-40% speedup  
✅ **Stable features**: Pre-trained vision features are already good  
✅ **Focus learning**: All compute goes to LLM and connector  

**Loading**:
```python
from transformers import CLIPVisionModel

vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")

# Freeze all parameters
for param in vision_encoder.parameters():
    param.requires_grad = False

# Set to eval mode
vision_encoder.eval()
```

### 2. Multimodal Connector (TRAINED ✏️)

**Purpose**: Project vision tokens to LLM input space

**Configuration**:
- Type: Linear projection or 2-layer MLP
- Input dim: 1024 (vision encoder output)
- Output dim: 896 (Qwen3-0.6B hidden size)
- Parameters: ~1M (trainable)

**Architecture Options**:

**Option A: Linear (simpler)**:
```python
connector = nn.Linear(1024, 896)
# Parameters: 1024 × 896 = ~900K
```

**Option B: MLP (more capacity)**:
```python
connector = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.GELU(),
    nn.Linear(2048, 896)
)
# Parameters: (1024×2048) + (2048×896) = ~4M
```

**Recommendation**: Start with Linear, upgrade to MLP if needed.

### 3. Language Model (TRAINED ✏️)

**Model**: Qwen3-0.6B

**Configuration**:
- Model: `Qwen/Qwen2.5-0.5B` (closest available)
- Parameters: ~600M (trainable)
- Hidden size: 896
- Num layers: 24
- Num heads: 14
- Context length: 32,768
- Vocabulary: 151,936 tokens

**Why Qwen3-0.6B?**:
✅ **Small & efficient**: 600M params vs Molmo2's 1B-8B  
✅ **Good quality**: Strong base model from Alibaba  
✅ **Long context**: Supports up to 32K tokens  
✅ **Open source**: Apache 2.0 license  
✅ **Educational**: Perfect size for learning  

**Loading**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
)

# All parameters trainable by default
```

---

## Complete Model Implementation

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class NanoMolmo2(nn.Module):
    """
    nanoMolmo2: Educational VLM with frozen vision encoder.
    
    Architecture:
    - Vision: Frozen CLIP ViT (Molmo2's encoder)
    - Connector: Trainable projection
    - LLM: Trainable Qwen3-0.6B
    """
    
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-large-patch14-336",
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        connector_hidden_size: int = None,  # None = linear, else MLP
    ):
        super().__init__()
        
        # 1. Vision Encoder (FROZEN)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
        
        # 2. Language Model
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<image>", "<video>", "<point>", "<timestamp>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # 3. Multimodal Connector
        vision_dim = self.vision_encoder.config.hidden_size  # 1024
        llm_dim = self.llm.config.hidden_size  # 896
        
        if connector_hidden_size is None:
            # Simple linear projection
            self.connector = nn.Linear(vision_dim, llm_dim)
        else:
            # MLP projection
            self.connector = nn.Sequential(
                nn.Linear(vision_dim, connector_hidden_size),
                nn.GELU(),
                nn.Linear(connector_hidden_size, llm_dim)
            )
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual tokens.
        
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            visual_tokens: [batch_size, num_patches, llm_dim]
        """
        # Vision encoder (frozen, no gradients)
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            visual_features = vision_outputs.last_hidden_state  # [B, N, 1024]
        
        # Connector (trainable)
        visual_tokens = self.connector(visual_features)  # [B, N, 896]
        
        return visual_tokens
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass with images and text.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, num_images, 3, H, W]
            labels: [batch_size, seq_len]
        """
        # Encode images if provided
        if pixel_values is not None:
            batch_size, num_images = pixel_values.shape[:2]
            
            # Flatten for encoding
            images = pixel_values.view(-1, 3, pixel_values.shape[-2], pixel_values.shape[-1])
            
            # Get visual tokens
            visual_tokens = self.encode_images(images)  # [B*N, num_patches, llm_dim]
            
            # Reshape back
            visual_tokens = visual_tokens.view(batch_size, num_images, -1, visual_tokens.shape[-1])
        
        # Get text embeddings
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # Merge visual and text tokens (simplified - needs proper interleaving)
        # TODO: Implement proper token merging based on special tokens
        
        # For now, forward through LLM
        outputs = self.llm(
            inputs_embeds=text_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate(self, input_ids, pixel_values=None, **kwargs):
        """Generate text given image and prompt."""
        # Encode images
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
            # Merge with text tokens
            # TODO: Implement
        
        # Generate
        outputs = self.llm.generate(input_ids, **kwargs)
        return outputs
```

---

## Training Configuration

### Memory Breakdown (with frozen vision encoder)

**Per-GPU Memory (A100 40GB)**:

```
Component                      Memory      Trainable?
─────────────────────────────────────────────────────
Vision Encoder (300M)          1.2 GB      🔒 NO
  Parameters (fp16)            600 MB
  Activations                  600 MB
  Gradients                    0 MB        ← Frozen!
  Optimizer states             0 MB        ← Frozen!

Connector (1M-4M)              ~50 MB      ✏️ YES
  Parameters                   4 MB
  Gradients                    4 MB
  Optimizer states (AdamW)     12 MB
  Activations                  30 MB

LLM (600M)                     15 GB       ✏️ YES
  Parameters (bf16)            1.2 GB
  Gradients                    1.2 GB
  Optimizer states (AdamW)     3.6 GB
  Activations (batch)          9 GB

Batch Data                     4 GB
  Images (batch=32)            2 GB
  Tokens (seq=4096)            2 GB

─────────────────────────────────────────────────────
TOTAL                          ~20 GB      ✅ Fits in 40GB!
```

**Comparison with unfrozen vision encoder**:

| Setup | Vision Encoder | Total Memory | Speedup |
|-------|----------------|--------------|---------|
| Frozen (ours) | 🔒 | ~20 GB | 1.3-1.4× |
| Unfrozen | ✏️ | ~30 GB | 1.0× |

**Savings**: ~10 GB per GPU + 30-40% faster!

### Recommended Training Settings

**Stage 1: Pre-training**
```python
config = {
    "batch_size": 32,  # per GPU
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "max_steps": 100000,
    "gradient_accumulation": 4,
    
    # Frozen vision encoder
    "freeze_vision_encoder": True,
    
    # What to train
    "train_connector": True,
    "train_llm": True,
}
```

**Stage 2: SFT**
```python
config = {
    "batch_size": 16,  # per GPU
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "max_steps": 50000,
    "gradient_accumulation": 8,
    
    # Still frozen
    "freeze_vision_encoder": True,
    
    # What to train
    "train_connector": True,
    "train_llm": True,
}
```

**Stage 3: Long-context**
```python
config = {
    "batch_size": 4,  # per GPU
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_steps": 2000,
    "gradient_accumulation": 16,
    
    # Still frozen
    "freeze_vision_encoder": True,
    
    # What to train
    "train_connector": True,
    "train_llm": True,
}
```

---

## Hardware Requirements (with frozen encoder)

### Minimum Setup
- **GPUs**: 2× A100 40GB
- **RAM**: 64 GB
- **Storage**: 200 GB (datasets) + 50 GB (checkpoints)

### Recommended Setup
- **GPUs**: 4× A100 40GB or 2× A100 80GB
- **RAM**: 128 GB
- **Storage**: 500 GB

### Training Time Estimates

| Stage | Steps | GPUs | Time |
|-------|-------|------|------|
| Pre-training | 100K | 4× A100 | 5-7 days |
| SFT | 50K | 4× A100 | 2-3 days |
| Long-context | 2K | 4× A100 | 4-6 hours |

**Total**: ~10 days on 4× A100 40GB GPUs

---

## Model Size Comparison

| Component | nanoMolmo2 | Molmo2-1B | Molmo2-7B |
|-----------|------------|-----------|-----------|
| Vision Encoder | ~300M 🔒 | ~300M ✏️ | ~300M ✏️ |
| Connector | ~1M ✏️ | ~10M ✏️ | ~30M ✏️ |
| LLM | ~600M ✏️ | ~1B ✏️ | ~7B ✏️ |
| **Trainable** | **601M** | **1.3B** | **7.3B** |
| **Total** | **901M** | **1.3B** | **7.3B** |

**Advantage**: Only need to save/load 601M trainable parameters!

---

## Advantages of This Setup

### 1. **Memory Efficiency** 🎯
- Frozen encoder saves ~10 GB per GPU
- Can use larger batch sizes
- Can train on cheaper GPUs (RTX 4090, etc.)

### 2. **Training Speed** ⚡
- 30-40% faster training
- Skip vision encoder backward pass
- More iterations per day

### 3. **Stable Vision Features** 🎨
- Pre-trained CLIP features are already excellent
- Avoid catastrophic forgetting
- Focus learning on language understanding

### 4. **Educational Value** 📚
- Simpler to understand (fewer moving parts)
- Faster experimentation
- Learn VLM principles without huge compute

### 5. **Good Enough Performance** ✅
- Many successful VLMs use frozen encoders
- LLaVA, MiniGPT-4, etc. use this approach
- Performance is still competitive

---

## Potential Limitations

### 1. **Vision-Language Misalignment**
- Vision encoder not adapted to LLM
- May need stronger connector (MLP instead of linear)

**Solution**: Use 2-layer MLP connector

### 2. **Domain Adaptation**
- Vision encoder trained on CLIP data (web images)
- May not transfer perfectly to videos, documents

**Solution**: 
- Use vision encoder pre-trained on diverse data
- Collect more connector training data

### 3. **Can't Learn New Visual Concepts**
- Frozen encoder can't adapt to new visual patterns

**Solution**: This is OK for educational project!

---

## When to Unfreeze Vision Encoder?

Consider unfreezing IF:
- Have abundant GPU resources (8× A100 80GB+)
- Need absolute best performance
- Working with very different visual domains
- In production deployment

For educational purposes: **Keep it frozen!**

---

## Next Steps

1. **Implement model**: See `models/nanomolmo2.py`
2. **Test forward pass**: Verify shapes and memory
3. **Implement connector**: Start with linear, upgrade if needed
4. **Add special token handling**: Properly merge visual/text tokens
5. **Train**: Use stage-specific dataloaders

See `QUICKSTART.md` for training instructions.

---

**Architecture**: Frozen Vision Encoder + Trainable Connector + Trainable Qwen3-0.6B  
**Trainable Parameters**: ~601M  
**Memory Required**: ~20 GB per GPU  
**Training Speed**: 1.3-1.4× faster than unfrozen  
**Perfect for**: Educational VLM implementation!
