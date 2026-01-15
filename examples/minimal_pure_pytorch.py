#!/usr/bin/env python3
"""
Minimal Pure PyTorch Implementation of nanoMolmo2

This is a simplified, educational implementation showing the core concepts.
Total: ~400 lines of pure PyTorch code!

Usage:
    python examples/minimal_pure_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# 1. VISION ENCODER (Pure PyTorch ViT)
# =============================================================================

class VisionTransformer(nn.Module):
    """Simplified Vision Transformer."""
    
    def __init__(self, img_size=224, patch_size=16, dim=768, depth=12, heads=12):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: [B, 3, H, W] -> [B, dim, h, w] -> [B, num_patches, dim]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x)


# =============================================================================
# 2. LANGUAGE MODEL (Pure PyTorch Transformer Decoder)
# =============================================================================

class TransformerDecoder(nn.Module):
    """Simplified Transformer Decoder (GPT-style)."""
    
    def __init__(self, vocab_size=50000, dim=512, depth=12, heads=8, max_len=2048):
        super().__init__()
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        
        # Transformer decoder blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
    
    def forward(self, input_ids, memory=None):
        B, L = input_ids.shape
        
        # Token + position embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Transformer
        if memory is None:
            memory = x  # Self-attention only
        
        x = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
        )
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


# =============================================================================
# 3. COMPLETE MODEL (nanoMolmo2)
# =============================================================================

class NanoMolmo2(nn.Module):
    """
    Minimal nanoMolmo2: Vision Encoder + Connector + Language Model.
    
    Pure PyTorch implementation for educational purposes.
    """
    
    def __init__(
        self,
        img_size=224,
        vision_dim=768,
        llm_dim=512,
        vocab_size=50000,
        freeze_vision=True,
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VisionTransformer(
            img_size=img_size,
            dim=vision_dim,
            depth=12,
            heads=12,
        )
        
        # Freeze vision encoder
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
        
        # Connector (project vision to LLM dim)
        self.connector = nn.Linear(vision_dim, llm_dim)
        
        # Language model
        self.llm = TransformerDecoder(
            vocab_size=vocab_size,
            dim=llm_dim,
            depth=12,
            heads=8,
        )
        
        self.pad_token_id = 0
    
    def forward(self, input_ids, images=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            images: [B, 3, H, W] images (optional)
            labels: [B, L] target tokens (optional)
        
        Returns:
            logits: [B, L, vocab_size]
            loss: scalar (if labels provided)
        """
        # Encode images if provided
        memory = None
        if images is not None:
            with torch.no_grad():  # Vision frozen
                visual_features = self.vision_encoder(images)  # [B, N, vision_dim]
            
            # Project to LLM dimension
            memory = self.connector(visual_features)  # [B, N, llm_dim]
        
        # Forward through LLM
        logits = self.llm(input_ids, memory=memory)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.pad_token_id,
            )
        
        return logits, loss


# =============================================================================
# 4. SIMPLE TOKENIZER
# =============================================================================

class SimpleTokenizer:
    """Dead simple character-level tokenizer for demonstration."""
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    
    def encode(self, text, max_length=512):
        """Encode text to token IDs (character-level)."""
        # Simple: use ASCII values offset by 3
        tokens = [self.bos_token_id]
        tokens.extend([min(ord(c) + 3, self.vocab_size - 1) for c in text[:max_length-2]])
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, tokens):
        """Decode token IDs to text."""
        text = []
        for t in tokens:
            if t == self.pad_token_id:
                continue
            if t == self.bos_token_id or t == self.eos_token_id:
                continue
            text.append(chr(t - 3))
        return ''.join(text)


# =============================================================================
# 5. DATASET
# =============================================================================

class SimpleImageCaptionDataset(Dataset):
    """Simple dataset for image captioning."""
    
    def __init__(self, data_file, tokenizer, image_transform, max_length=512):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        
        # Load data (JSONL format)
        self.samples = []
        if Path(data_file).exists():
            with open(data_file, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        
        # Fallback: create dummy data if file doesn't exist
        if len(self.samples) == 0:
            print("Warning: No data file found, using dummy data")
            self.samples = [
                {"image_path": "dummy.jpg", "caption": "A cat sitting on a mat"}
            ] * 100
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (or create dummy)
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='red')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Tokenize text
        tokens = self.tokenizer.encode(sample['caption'], max_length=self.max_length)
        
        # Pad to max_length
        input_ids = tokens[:-1] + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens) + 1)
        labels = tokens[1:] + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens) + 1)
        
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'images': image,
        }


# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

def train(model, dataloader, optimizer, device, num_steps=1000):
    """Simple training loop."""
    model.train()
    model.to(device)
    
    pbar = tqdm(total=num_steps, desc="Training")
    step = 0
    
    while step < num_steps:
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            images = batch['images'].to(device)
            
            # Forward
            logits, loss = model(input_ids, images=images, labels=labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Log
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if step >= num_steps:
                break
    
    pbar.close()
    return model


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Minimal Pure PyTorch nanoMolmo2 Implementation")
    print("=" * 70)
    print()
    
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=512)
    
    # Image transform
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset
    dataset = SimpleImageCaptionDataset(
        data_file='data/dummy_train.jsonl',  # Will use dummy data if not exists
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=128,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Model
    model = NanoMolmo2(
        img_size=224,
        vision_dim=768,
        llm_dim=512,
        vocab_size=512,
        freeze_vision=True,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )
    
    # Train!
    print("Starting training...")
    model = train(model, dataloader, optimizer, device, num_steps=100)
    
    print("\n✓ Training complete!")
    print("\nThis minimal implementation shows:")
    print("  - Pure PyTorch Vision Transformer")
    print("  - Pure PyTorch Transformer Decoder")
    print("  - Frozen vision encoder")
    print("  - Custom training loop")
    print("  - No external frameworks!")
    print()
    print("Total code: ~400 lines of pure PyTorch ✨")


if __name__ == '__main__':
    main()
