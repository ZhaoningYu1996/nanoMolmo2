# Pure PyTorch Implementation Guide

**nanoMolmo2 with minimal dependencies - Educational implementation using only PyTorch**

---

## Philosophy

**Goal**: Implement everything from scratch using pure PyTorch to maximize learning

**Minimal Dependencies**:
- ✅ **PyTorch** - Core framework
- ✅ **torchvision** - Image preprocessing
- ✅ **Pillow (PIL)** - Image loading
- ✅ **numpy** - Numerical operations
- ❌ **NO** HuggingFace Transformers (implement models yourself)
- ❌ **NO** HuggingFace Datasets (implement dataloaders yourself)
- ❌ **NO** PyTorch Lightning (write training loop yourself)
- ❌ **NO** other training frameworks

**Why Pure PyTorch?**
- 🎓 **Educational**: Understand every component
- 🔍 **Transparent**: See exactly what's happening
- 🛠️ **Flexible**: Easy to modify and experiment
- 💪 **Skill building**: Learn core ML engineering

---

## Minimal Dependencies

### Core Requirements

```bash
# requirements_minimal.txt

# Core deep learning
torch>=2.0.0
torchvision>=0.15.0

# Data processing
numpy>=1.24.0
pillow>=10.0.0

# Video processing
opencv-python>=4.8.0

# Optional: Logging and visualization
tensorboard>=2.14.0
tqdm>=4.66.0

# Optional: Fast data loading
pyarrow>=14.0.0  # For parquet files
```

**No Transformers, No Datasets library, No Lightning!**

---

## Project Structure

```
nanoMolmo2/
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py      # Implement ViT from scratch
│   ├── language_model.py      # Implement transformer from scratch
│   ├── connector.py            # Simple projection layer
│   └── nanomolmo2.py          # Complete model
│
├── data/
│   ├── __init__.py
│   ├── image_dataset.py       # Pure PyTorch Dataset
│   ├── video_dataset.py       # Pure PyTorch Dataset
│   ├── collator.py            # Custom batching
│   └── sampler.py             # Custom sampling strategies
│
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Custom training loop
│   ├── optimizer.py           # Optimizer setup
│   ├── scheduler.py           # LR scheduling
│   └── utils.py               # Training utilities
│
└── scripts/
    ├── train_stage1.py        # Stage 1 training script
    ├── train_stage2.py        # Stage 2 training script
    └── train_stage3.py        # Stage 3 training script
```

---

## Implementation Guide

### 1. Vision Encoder (Pure PyTorch ViT)

**Option A: Implement from Scratch** (Most Educational)

```python
# models/vision_encoder.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them."""
    
    def __init__(self, img_size=336, patch_size=14, in_channels=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolution to extract patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, 3, 336, 336]
        x = self.projection(x)  # [B, 1024, 24, 24]
        x = x.flatten(2)  # [B, 1024, 576]
        x = x.transpose(1, 2)  # [B, 576, 1024]
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Pure PyTorch Vision Transformer (ViT)."""
    
    def __init__(
        self,
        img_size=336,
        patch_size=14,
        in_channels=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # x: [B, 3, 336, 336]
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 576, 1024]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 577, 1024]
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        return x  # [B, 577, 1024]
```

**Option B: Load Pre-trained CLIP Weights** (Recommended)

```python
# models/vision_encoder.py
import torch
import torch.nn as nn

class CLIPVisionEncoder(nn.Module):
    """
    Load CLIP vision encoder weights but use pure PyTorch.
    
    We'll download CLIP weights manually and load them into our ViT.
    """
    
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # Create ViT architecture (same as above)
        self.vit = VisionTransformer(
            img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            self.vit.load_state_dict(state_dict)
        
        # Freeze for training
        self.freeze()
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():  # No gradients needed (frozen)
            return self.vit(x)
```

### 2. Language Model (Pure PyTorch Transformer)

**Option A: Implement Transformer from Scratch** (Most Educational)

```python
# models/language_model.py
import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - used in Qwen."""
    
    def __init__(self, dim, max_seq_len=32768):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to queries and keys."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        B, L, D = x.shape
        
        # Self-attention with RoPE
        residual = x
        x = self.norm1(x)
        
        # Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: [B, L, num_heads, head_dim]
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)
        
        # Transpose for attention: [B, num_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Apply mask
        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        
        x = residual + out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class LanguageModel(nn.Module):
    """Pure PyTorch Language Model (Qwen-style)."""
    
    def __init__(
        self,
        vocab_size=151936,
        dim=896,
        num_layers=24,
        num_heads=14,
        mlp_ratio=4.0,
        max_seq_len=32768,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        
        # Rotary embeddings
        self.rope = RotaryEmbedding(dim // num_heads, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final norm and output
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, L]
        B, L = input_ids.shape
        
        # Token embeddings
        x = self.token_embed(input_ids)  # [B, L, dim]
        
        # Get RoPE embeddings
        rope_cos, rope_sin = self.rope(L)
        
        # Expand for batch and heads
        rope_cos = rope_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim]
        rope_sin = rope_sin.unsqueeze(0).unsqueeze(0)
        
        # Causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            ).logical_not()
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin, attention_mask)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]
        
        return logits
```

**Option B: Load Qwen Weights** (Recommended)

```python
# You can download Qwen weights and load them manually
# Or start with random initialization for learning purposes
```

### 3. Complete Model

```python
# models/nanomolmo2.py
import torch
import torch.nn as nn
from .vision_encoder import VisionTransformer
from .language_model import LanguageModel

class NanoMolmo2(nn.Module):
    """
    Complete nanoMolmo2 model - Pure PyTorch implementation.
    
    Architecture:
    - Vision Encoder: ViT (frozen)
    - Connector: Linear projection
    - Language Model: Transformer decoder
    """
    
    def __init__(
        self,
        vision_dim=1024,
        llm_dim=896,
        vocab_size=151936,
        freeze_vision=True,
    ):
        super().__init__()
        
        # Vision encoder (frozen)
        self.vision_encoder = VisionTransformer(
            img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
        
        # Connector
        self.connector = nn.Linear(vision_dim, llm_dim)
        
        # Language model
        self.llm = LanguageModel(
            vocab_size=vocab_size,
            dim=llm_dim,
            num_layers=24,
            num_heads=14,
        )
        
        # Special token IDs (set after creating tokenizer)
        self.image_token_id = None
        self.pad_token_id = None
    
    def encode_images(self, images):
        """Encode images to visual tokens."""
        # images: [B, 3, 336, 336]
        
        with torch.no_grad():  # Vision encoder is frozen
            visual_features = self.vision_encoder(images)  # [B, 577, 1024]
        
        # Project to LLM dimension
        visual_tokens = self.connector(visual_features)  # [B, 577, 896]
        
        return visual_tokens
    
    def forward(self, input_ids, images=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] - Text token IDs
            images: [B, 3, H, W] - Images (optional)
            labels: [B, L] - Target token IDs for training
        
        Returns:
            logits: [B, L, vocab_size]
            loss: scalar (if labels provided)
        """
        # Get text embeddings
        text_embeds = self.llm.token_embed(input_ids)  # [B, L, 896]
        
        # If images provided, merge visual tokens
        if images is not None:
            visual_tokens = self.encode_images(images)  # [B, 577, 896]
            
            # TODO: Properly merge visual and text tokens
            # For now, simple concatenation
            # In practice, need to handle <image> tokens properly
            pass
        
        # Forward through LLM
        logits = self.llm(input_ids)  # [B, L, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.pad_token_id if self.pad_token_id else -100
            )
        
        return logits, loss
```

### 4. Pure PyTorch Dataset

```python
# data/image_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path

class ImageCaptionDataset(Dataset):
    """Pure PyTorch dataset for image captioning."""
    
    def __init__(self, data_file, image_transform=None):
        """
        Args:
            data_file: Path to JSONL file
            image_transform: torchvision transforms
        """
        self.data_file = Path(data_file)
        self.image_transform = image_transform
        
        # Load data
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        
        # Get text
        text = sample['caption']
        
        return {
            'image': image,
            'text': text,
            'image_path': sample['image_path'],
        }
```

### 5. Custom Training Loop

```python
# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    """Pure PyTorch trainer - no frameworks!"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device='cuda',
        max_steps=100000,
        save_every=5000,
        log_every=100,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_steps = max_steps
        self.save_every = save_every
        self.log_every = log_every
        
        self.step = 0
        self.epoch = 0
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        images = batch['images'].to(self.device) if 'images' in batch else None
        labels = batch['labels'].to(self.device)
        
        # Forward
        logits, loss = self.model(input_ids, images, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps...")
        
        pbar = tqdm(total=self.max_steps, desc="Training")
        
        while self.step < self.max_steps:
            for batch in self.train_dataloader:
                # Train step
                loss = self.train_step(batch)
                
                self.step += 1
                pbar.update(1)
                
                # Logging
                if self.step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{lr:.2e}'})
                
                # Save checkpoint
                if self.step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
                
                if self.step >= self.max_steps:
                    break
            
            self.epoch += 1
        
        pbar.close()
        print("Training complete!")
    
    def save_checkpoint(self, path):
        """Save checkpoint."""
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")
```

---

## Complete Training Script Example

```python
# scripts/train_stage1.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Your implementations
from models.nanomolmo2 import NanoMolmo2
from data.image_dataset import ImageCaptionDataset
from training.trainer import Trainer

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 1e-4
    max_steps = 100000
    
    # Model
    model = NanoMolmo2(
        vision_dim=1024,
        llm_dim=896,
        vocab_size=151936,
        freeze_vision=True,
    )
    
    # Image transforms
    image_transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Dataset
    train_dataset = ImageCaptionDataset(
        data_file='data/molmo2_datasets/pixmo-cap/train.jsonl',
        image_transform=image_transform,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=max_steps,
        save_every=5000,
        log_every=100,
    )
    
    # Train!
    trainer.train()

if __name__ == '__main__':
    main()
```

---

## Benefits of Pure PyTorch

### 1. **Full Control** 🎮
- Understand every line of code
- Easy to debug and modify
- No hidden behaviors

### 2. **Educational** 🎓
- Learn transformer architecture deeply
- Understand training loop mechanics
- Build ML engineering skills

### 3. **Flexible** 🛠️
- Add custom features easily
- Experiment with architectures
- No framework constraints

### 4. **Portable** 📦
- Minimal dependencies
- Easy to deploy
- Works anywhere PyTorch works

---

## Next Steps

1. **Implement Vision Encoder**
   - Start with simple ViT
   - Add RoPE if needed
   - Load CLIP weights later

2. **Implement Language Model**
   - Build transformer decoder
   - Add RoPE for position encoding
   - Test with small examples

3. **Implement Datasets**
   - Simple JSONL reader
   - Image loading with PIL
   - Video loading with OpenCV

4. **Write Training Loop**
   - Forward/backward pass
   - Gradient clipping
   - Checkpointing

5. **Test and Debug**
   - Verify shapes
   - Check memory usage
   - Profile performance

---

## Example: Minimal Working Implementation

See `examples/minimal_pure_pytorch.py` for a complete working example with:
- ✅ Vision encoder (ViT)
- ✅ Language model (Transformer)
- ✅ Dataset (JSONL)
- ✅ Training loop
- ✅ < 500 lines of code!

---

**Pure PyTorch = Maximum Learning!** 🎓

No magic, just code you can understand and control.
