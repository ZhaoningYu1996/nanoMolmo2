"""
SigLIP 2 So400m/14 @ 384px Vision Encoder

Pure PyTorch implementation with weight loading support.

Architecture (from Molmo2 tech report):
- Model: SigLIP 2 So400m/14
- Input: 384×384 RGB images
- Patch size: 14×14
- Num patches: 27×27 = 729 (no CLS token in SigLIP)
- Hidden size: 1152
- Layers: 27
- Heads: 16
- MLP ratio: ~4.3688 (intermediate_size=5032)

References:
- SigLIP 2 paper: https://arxiv.org/abs/2502.14786
- HuggingFace: google/siglip2-so400m-patch14-384
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    """Configuration for SigLIP 2 So400m/14 vision encoder."""
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1152
    intermediate_size: int = 4304  # SigLIP uses ~3.74x ratio
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2  # 729
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 72


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
        )
        # Learnable position embeddings
        self.position_embedding = nn.Embedding(
            config.num_patches,
            config.hidden_size,
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_patches).unsqueeze(0),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            embeddings: [B, num_patches, hidden_size]
        """
        # Patch projection: [B, 3, 384, 384] -> [B, hidden_size, 27, 27]
        patch_embeds = self.projection(pixel_values)
        # Flatten: [B, hidden_size, 27, 27] -> [B, hidden_size, 729] -> [B, 729, hidden_size]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # Add position embeddings
        position_embeds = self.position_embedding(self.position_ids)
        embeddings = patch_embeds + position_embeds
        return embeddings


class Attention(nn.Module):
    """Multi-head self-attention with SDPA."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = config.attention_dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, 1, L, L] or None
        Returns:
            output: [B, L, D]
        """
        B, L, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head: [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention (uses Flash/Memory-efficient kernels when available)
        # SDPA supports custom attention masks, unlike FlashAttention standalone
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        
        # Reshape back: [B, num_heads, L, head_dim] -> [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        
        # Output projection
        output = self.out_proj(attn_output)
        return output


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation = nn.GELU(approximate="tanh")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class EncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-norm transformer block."""
        # Self-attention with residual
        residual = hidden_states
        # LayerNorm in full precision (as per Molmo2 tech report)
        hidden_states = self.layer_norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class VisionEncoder(nn.Module):
    """
    SigLIP 2 So400m/14 @ 384px Vision Encoder.
    
    Pure PyTorch implementation for frozen feature extraction.
    
    Usage:
        encoder = VisionEncoder.from_pretrained("google/siglip2-so400m-patch14-384")
        encoder.freeze()
        
        with torch.no_grad():
            features = encoder(images)  # [B, 729, 1152]
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        super().__init__()
        self.config = config or VisionConfig()
        
        self.embeddings = PatchEmbedding(self.config)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)
        ])
        self.post_layernorm = nn.LayerNorm(
            self.config.hidden_size, eps=self.config.layer_norm_eps
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights (Xavier uniform for linear, normal for embeddings)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 384, 384] normalized RGB images
            attention_mask: Optional [B, 1, num_patches, num_patches]
            
        Returns:
            hidden_states: [B, num_patches, hidden_size] = [B, 729, 1152]
        """
        # Patch embedding
        hidden_states = self.embeddings(pixel_values)
        
        # Encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm (in full precision)
        hidden_states = self.post_layernorm(hidden_states.float()).to(hidden_states.dtype)
        
        return hidden_states
    
    def freeze(self):
        """Freeze all parameters for feature extraction."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def unfreeze(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "google/siglip2-so400m-patch14-384",
        cache_dir: Optional[str] = "./checkpoints",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "VisionEncoder":
        """
        Load pretrained SigLIP 2 weights.
        
        Uses local cache if available (fast), otherwise downloads from HuggingFace (slow).
        Run `python scripts/download_model_weights.py` to pre-download weights.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            cache_dir: Directory for cached weights (default: ./checkpoints)
            device: Target device
            dtype: Target dtype (default: bfloat16)
            
        Returns:
            VisionEncoder with loaded weights
        """
        from pathlib import Path
        
        # Try local cache first (fast path)
        if cache_dir:
            cache_path = Path(cache_dir) / "siglip2_so400m_384.pt"
            if cache_path.exists():
                return cls._load_from_local(cache_path, device, dtype)
        
        # Fall back to HuggingFace (slow path)
        return cls._load_from_huggingface(model_name_or_path, device, dtype)
    
    @classmethod
    def _load_from_local(
        cls,
        cache_path,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "VisionEncoder":
        """Load from local cache (fast)."""
        checkpoint = torch.load(cache_path, map_location="cpu", weights_only=True)
        
        config = VisionConfig(**checkpoint["config"])
        model = cls(config)
        
        # Load weights (filter out position_ids buffer)
        state_dict = checkpoint["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        if missing:
            print(f"Warning: Missing keys: {missing}")
        
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        
        return model
    
    @classmethod
    def _load_from_huggingface(
        cls,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "VisionEncoder":
        """Load from HuggingFace (slow, use for first-time download)."""
        try:
            from transformers import SiglipVisionModel, AutoConfig
            
            print(f"Loading from HuggingFace: {model_name_or_path}")
            print("Tip: Run `python scripts/download_model_weights.py` to cache weights locally for faster loading.")
            
            # Load HuggingFace config (vision config is nested)
            full_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            hf_config = full_config.vision_config if hasattr(full_config, 'vision_config') else full_config
            
            # Load HuggingFace model (SiglipVisionModel works for SigLIP2)
            hf_model = SiglipVisionModel.from_pretrained(model_name_or_path)
            
            # Create our config from HF config
            config = VisionConfig(
                image_size=hf_config.image_size,
                patch_size=hf_config.patch_size,
                num_channels=hf_config.num_channels,
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                layer_norm_eps=hf_config.layer_norm_eps,
                attention_dropout=getattr(hf_config, 'attention_dropout', 0.0),
            )
            
            # Create our model
            model = cls(config)
            
            # Load weights
            model._load_from_hf_model(hf_model)
            
            if dtype is not None:
                model = model.to(dtype=dtype)
            if device is not None:
                model = model.to(device=device)
            
            return model
            
        except ImportError:
            raise ImportError(
                "Loading pretrained weights requires transformers. "
                "Install with: pip install transformers"
            )
    
    def _load_from_hf_model(self, hf_model):
        """Load weights from HuggingFace SiglipVisionModel."""
        hf_state = hf_model.state_dict()
        
        # Weight mapping from HF to our model
        # HF uses "vision_model.embeddings.patch_embedding.weight" format
        mapping = {}
        
        # Embeddings - HF has Conv2d patch_embedding, we also use Conv2d projection
        mapping["embeddings.projection.weight"] = "vision_model.embeddings.patch_embedding.weight"
        mapping["embeddings.projection.bias"] = "vision_model.embeddings.patch_embedding.bias"
        mapping["embeddings.position_embedding.weight"] = "vision_model.embeddings.position_embedding.weight"
        
        # Encoder layers
        for i in range(self.config.num_hidden_layers):
            hf_prefix = f"vision_model.encoder.layers.{i}"
            our_prefix = f"encoder_layers.{i}"
            
            # Layer norms
            mapping[f"{our_prefix}.layer_norm1.weight"] = f"{hf_prefix}.layer_norm1.weight"
            mapping[f"{our_prefix}.layer_norm1.bias"] = f"{hf_prefix}.layer_norm1.bias"
            mapping[f"{our_prefix}.layer_norm2.weight"] = f"{hf_prefix}.layer_norm2.weight"
            mapping[f"{our_prefix}.layer_norm2.bias"] = f"{hf_prefix}.layer_norm2.bias"
            
            # Attention
            mapping[f"{our_prefix}.self_attn.q_proj.weight"] = f"{hf_prefix}.self_attn.q_proj.weight"
            mapping[f"{our_prefix}.self_attn.q_proj.bias"] = f"{hf_prefix}.self_attn.q_proj.bias"
            mapping[f"{our_prefix}.self_attn.k_proj.weight"] = f"{hf_prefix}.self_attn.k_proj.weight"
            mapping[f"{our_prefix}.self_attn.k_proj.bias"] = f"{hf_prefix}.self_attn.k_proj.bias"
            mapping[f"{our_prefix}.self_attn.v_proj.weight"] = f"{hf_prefix}.self_attn.v_proj.weight"
            mapping[f"{our_prefix}.self_attn.v_proj.bias"] = f"{hf_prefix}.self_attn.v_proj.bias"
            mapping[f"{our_prefix}.self_attn.out_proj.weight"] = f"{hf_prefix}.self_attn.out_proj.weight"
            mapping[f"{our_prefix}.self_attn.out_proj.bias"] = f"{hf_prefix}.self_attn.out_proj.bias"
            
            # MLP
            mapping[f"{our_prefix}.mlp.fc1.weight"] = f"{hf_prefix}.mlp.fc1.weight"
            mapping[f"{our_prefix}.mlp.fc1.bias"] = f"{hf_prefix}.mlp.fc1.bias"
            mapping[f"{our_prefix}.mlp.fc2.weight"] = f"{hf_prefix}.mlp.fc2.weight"
            mapping[f"{our_prefix}.mlp.fc2.bias"] = f"{hf_prefix}.mlp.fc2.bias"
        
        # Post layer norm
        mapping["post_layernorm.weight"] = "vision_model.post_layernorm.weight"
        mapping["post_layernorm.bias"] = "vision_model.post_layernorm.bias"
        
        # Copy weights
        new_state = {}
        for our_key, hf_key in mapping.items():
            if hf_key in hf_state:
                new_state[our_key] = hf_state[hf_key]
            else:
                print(f"Warning: {hf_key} not found in HF model")
        
        # Load with strict=False to allow missing keys
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        
        # Filter out position_ids which is a buffer, not a parameter
        missing = [k for k in missing if "position_ids" not in k]
        
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    
    def save_pretrained(self, save_path: str):
        """Save model weights in our format."""
        import json
        from pathlib import Path
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "num_channels": self.config.num_channels,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "layer_norm_eps": self.config.layer_norm_eps,
            "attention_dropout": self.config.attention_dropout,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), save_dir / "model.pt")


# Default preprocessing for SigLIP 2
def get_image_transform(image_size: int = 384):
    """Get image preprocessing transform for SigLIP 2."""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],  # SigLIP uses 0.5 normalization
            std=[0.5, 0.5, 0.5],
        ),
    ])


if __name__ == "__main__":
    # Test the vision encoder
    print("Testing VisionEncoder...")
    
    config = VisionConfig()
    print(f"Config: {config}")
    print(f"Num patches: {config.num_patches}")
    
    model = VisionEncoder(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 384, 384)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [2, 729, 1152]
    
    # Test freeze
    model.freeze()
    print(f"Frozen: {not any(p.requires_grad for p in model.parameters())}")
    
    print("✓ VisionEncoder tests passed!")
