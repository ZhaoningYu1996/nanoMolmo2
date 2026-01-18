"""
Multimodal Connector

Projects vision encoder outputs to language model embedding space.

Options:
- Linear: Simple linear projection (fastest, ~1M params)
- MLP: 2-layer MLP with GELU (more capacity, ~4M params)
- Resampler: Cross-attention based (Molmo2 style, most flexible)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConnectorConfig:
    """Configuration for multimodal connector."""
    vision_dim: int = 1152  # SigLIP 2 So400m hidden size
    llm_dim: int = 896      # Qwen3-0.6B hidden size
    hidden_dim: int = 2048  # MLP hidden size (if using MLP)
    num_tokens: int = 64    # Number of output tokens (if using pooling)
    connector_type: str = "linear"  # "linear", "mlp", or "resampler"


class LinearConnector(nn.Module):
    """Simple linear projection from vision to LLM space."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.proj = nn.Linear(config.vision_dim, config.llm_dim, bias=True)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, num_patches, vision_dim]
        Returns:
            projected: [B, num_patches, llm_dim]
        """
        return self.proj(vision_features)


class MLPConnector(nn.Module):
    """2-layer MLP projection with GELU activation."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_dim, config.hidden_dim, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(config.hidden_dim, config.llm_dim, bias=True)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, num_patches, vision_dim]
        Returns:
            projected: [B, num_patches, llm_dim]
        """
        x = self.fc1(vision_features)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ResamplerConnector(nn.Module):
    """
    Cross-attention based resampler (Molmo2 style).
    
    Uses learnable query tokens to attend to vision features,
    reducing the number of vision tokens passed to LLM.
    """
    
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.config = config
        
        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, config.num_tokens, config.llm_dim) * 0.02)
        
        # Project vision features to LLM dim for cross-attention
        self.vision_proj = nn.Linear(config.vision_dim, config.llm_dim, bias=True)
        
        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.llm_dim,
            num_heads=config.llm_dim // 64,  # 14 heads for 896 dim
            batch_first=True,
        )
        
        # Feed-forward
        self.norm1 = nn.LayerNorm(config.llm_dim)
        self.norm2 = nn.LayerNorm(config.llm_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.llm_dim, config.llm_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.llm_dim * 4, config.llm_dim),
        )
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, num_patches, vision_dim]
        Returns:
            resampled: [B, num_tokens, llm_dim]
        """
        B = vision_features.shape[0]
        
        # Project vision features
        kv = self.vision_proj(vision_features)  # [B, num_patches, llm_dim]
        
        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # [B, num_tokens, llm_dim]
        
        # Cross-attention: queries attend to vision features
        attn_out, _ = self.cross_attn(queries, kv, kv)
        x = self.norm1(queries + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


class MultimodalConnector(nn.Module):
    """
    Unified multimodal connector interface.
    
    Supports multiple connector types:
    - linear: Fast, preserves all vision tokens
    - mlp: More capacity, preserves all vision tokens
    - resampler: Reduces vision tokens via cross-attention
    
    Usage:
        connector = MultimodalConnector(config)
        llm_inputs = connector(vision_features)
    """
    
    def __init__(self, config: Optional[ConnectorConfig] = None):
        super().__init__()
        self.config = config or ConnectorConfig()
        
        if self.config.connector_type == "linear":
            self.connector = LinearConnector(self.config)
        elif self.config.connector_type == "mlp":
            self.connector = MLPConnector(self.config)
        elif self.config.connector_type == "resampler":
            self.connector = ResamplerConnector(self.config)
        else:
            raise ValueError(f"Unknown connector type: {self.config.connector_type}")
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, num_patches, vision_dim]
        Returns:
            llm_inputs: [B, num_tokens, llm_dim]
                - For linear/mlp: num_tokens = num_patches
                - For resampler: num_tokens = config.num_tokens
        """
        return self.connector(vision_features)


def create_connector(
    vision_dim: int = 1152,
    llm_dim: int = 896,
    connector_type: str = "linear",
    **kwargs,
) -> MultimodalConnector:
    """
    Create a multimodal connector.
    
    Args:
        vision_dim: Vision encoder hidden size (SigLIP 2: 1152)
        llm_dim: LLM hidden size (Qwen3-0.6B: 896)
        connector_type: "linear", "mlp", or "resampler"
        **kwargs: Additional config options
        
    Returns:
        MultimodalConnector instance
    """
    config = ConnectorConfig(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        connector_type=connector_type,
        **kwargs,
    )
    return MultimodalConnector(config)


if __name__ == "__main__":
    # Test connectors
    print("Testing MultimodalConnector...")
    
    # Vision features: [B=2, num_patches=729, vision_dim=1152]
    vision_features = torch.randn(2, 729, 1152)
    
    # Test linear connector
    linear = create_connector(connector_type="linear")
    out = linear(vision_features)
    print(f"Linear connector: {vision_features.shape} -> {out.shape}")
    print(f"  Params: {sum(p.numel() for p in linear.parameters()) / 1e6:.2f}M")
    
    # Test MLP connector
    mlp = create_connector(connector_type="mlp")
    out = mlp(vision_features)
    print(f"MLP connector: {vision_features.shape} -> {out.shape}")
    print(f"  Params: {sum(p.numel() for p in mlp.parameters()) / 1e6:.2f}M")
    
    # Test resampler connector
    resampler = create_connector(connector_type="resampler", num_tokens=64)
    out = resampler(vision_features)
    print(f"Resampler connector: {vision_features.shape} -> {out.shape}")
    print(f"  Params: {sum(p.numel() for p in resampler.parameters()) / 1e6:.2f}M")
    
    print("✓ MultimodalConnector tests passed!")
