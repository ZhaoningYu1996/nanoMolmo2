"""
Qwen3-0.6B-Base Language Model

Pure PyTorch implementation with RoPE and weight loading support.
Uses the base model (not instruct) for better VLM pre-training.

Architecture (from Qwen3-0.6B-Base):
- Hidden size: 1024
- Intermediate size: 3072
- Num layers: 28
- Num attention heads: 16
- Num key-value heads: 8 (GQA)
- Head dim: 128 (explicit, not hidden_size/heads)
- Max position embeddings: 32768
- Vocabulary size: 151936
- RoPE theta: 1000000.0

References:
- Qwen3-Base: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- RoPE: https://arxiv.org/abs/2104.09864
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    """Configuration for Qwen3-0.6B-Base language model."""
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # Grouped Query Attention
    head_dim: int = 128  # Qwen3 uses explicit head_dim (not hidden_size/heads)
    max_position_embeddings: int = 32768  # Base model context length
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 2


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Llama/Qwen)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RoPE and LayerNorm in full precision (as per Molmo2 tech report)
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._set_cos_sin_cache(max_position_embeddings, device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate to get [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, num_heads, L, head_dim] - used for dtype/device
            position_ids: [B, L]
        Returns:
            cos, sin: [1, 1, L, head_dim]
        """
        seq_len = position_ids.max() + 1
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        
        # Gather cos/sin values for the positions
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, L, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # RoPE in full precision (as per Molmo2 tech report)
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.float(), sin.float()
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and SDPA."""
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        # Qwen3 uses explicit head_dim which may differ from hidden_size/num_heads
        # Q proj: hidden_size -> num_heads * head_dim (1024 -> 16*128 = 2048)
        # K proj: hidden_size -> num_kv_heads * head_dim (1024 -> 8*128 = 1024)
        # V proj: hidden_size -> num_kv_heads * head_dim (1024 -> 8*128 = 1024)
        # O proj: num_heads * head_dim -> hidden_size (2048 -> 1024)
        # Qwen3 has no biases in attention projections
        
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        
        # Qwen3 has Q and K normalization before RoPE
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        self.dropout = config.attention_dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, 1, L, L] causal mask
            position_ids: [B, L]
            past_key_value: Optional cached (K, V)
            use_cache: Whether to return updated cache
        Returns:
            output: [B, L, D]
            past_key_value: Optional updated cache
        """
        B, L, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape: [B, L, num_heads * head_dim] -> [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Qwen3: Apply Q/K normalization before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0)
        
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        # Repeat KV for GQA: [B, num_kv_heads, L, head_dim] -> [B, num_heads, L, head_dim]
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Scaled dot-product attention with SDPA
        # SDPA supports custom masks (unlike standalone FlashAttention)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,  # Use built-in causal if no mask provided
        )
        
        # Reshape: [B, num_heads, L, head_dim] -> [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, new_cache


class MLP(nn.Module):
    """Feed-forward network with SiLU gate (SwiGLU variant)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(act(gate(x)) * up(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Pre-norm transformer decoder block."""
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_cache


class LanguageModel(nn.Module):
    """
    Qwen3-0.6B Style Language Model.
    
    Pure PyTorch implementation with RoPE and Grouped Query Attention.
    
    Usage:
        model = LanguageModel.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        logits = model(input_ids)  # [B, L, vocab_size]
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__()
        self.config = config or LLMConfig()
        
        # Token embedding
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(self.config, layer_idx)
            for layer_idx in range(self.config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        
        # Tie weights if configured
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def _make_causal_mask(
        self, input_ids: torch.Tensor, past_length: int = 0
    ) -> torch.Tensor:
        """Create causal attention mask."""
        B, L = input_ids.shape
        total_length = L + past_length
        
        # Create causal mask: positions can only attend to earlier positions
        mask = torch.triu(
            torch.ones(L, total_length, device=input_ids.device, dtype=torch.bool),
            diagonal=past_length + 1,
        )
        
        # Convert to attention mask format (0 = attend, -inf = mask)
        mask = mask.float().masked_fill(mask, float("-inf"))
        
        # Expand for batch and heads: [1, 1, L, total_length]
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] padding mask (1 = attend, 0 = pad)
            position_ids: [B, L] position indices
            past_key_values: List of (K, V) caches per layer
            use_cache: Whether to return updated caches
            return_hidden_states: Whether to return hidden states
            
        Returns:
            logits: [B, L, vocab_size]
            past_key_values: Optional list of (K, V) caches
            hidden_states: Optional [B, L, hidden_size]
        """
        B, L = input_ids.shape
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + L, device=input_ids.device
            ).unsqueeze(0).expand(B, -1)
        
        # Causal mask
        if attention_mask is None:
            causal_mask = self._make_causal_mask(input_ids, past_length)
        else:
            # Combine padding mask with causal mask
            causal_mask = self._make_causal_mask(input_ids, past_length)
            # Expand padding mask: [B, L] -> [B, 1, 1, L]
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * float("-inf")
            # Pad to total length if needed
            if past_length > 0:
                padding_mask = F.pad(padding_mask, (past_length, 0), value=0.0)
            causal_mask = causal_mask + padding_mask
        
        # Decoder layers
        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_cache = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_caches.append(new_cache)
        
        # Final norm
        final_hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(final_hidden_states)
        
        outputs = (logits,)
        if use_cache:
            outputs += (new_caches,)
        if return_hidden_states:
            outputs += (final_hidden_states,)
        
        return outputs if len(outputs) > 1 else logits
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get token embeddings for external use."""
        return self.embed_tokens
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen3-0.6B-Base",
        cache_dir: Optional[str] = "./checkpoints",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "LanguageModel":
        """
        Load pretrained Qwen3-Base weights.
        
        Uses local cache if available (fast), otherwise downloads from HuggingFace (slow).
        Run `python scripts/download_model_weights.py` to pre-download weights.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            cache_dir: Directory for cached weights (default: ./checkpoints)
            device: Target device
            dtype: Target dtype (default: bfloat16)
            
        Returns:
            LanguageModel with loaded weights
        """
        from pathlib import Path
        
        # Try local cache first (fast path)
        if cache_dir:
            cache_path = Path(cache_dir) / "qwen3_0.6b_base.pt"
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
    ) -> "LanguageModel":
        """Load from local cache (fast)."""
        checkpoint = torch.load(cache_path, map_location="cpu", weights_only=True)
        
        config = LLMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        
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
    ) -> "LanguageModel":
        """Load from HuggingFace (slow, use for first-time download)."""
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            
            print(f"Loading from HuggingFace: {model_name_or_path}")
            print("Tip: Run `python scripts/download_model_weights.py` to cache weights locally for faster loading.")
            
            # Load HuggingFace config
            hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            
            # Create our config from HF config
            # Qwen3 has explicit head_dim (128) which differs from hidden_size/num_heads
            config = LLMConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
                max_position_embeddings=hf_config.max_position_embeddings,
                rms_norm_eps=hf_config.rms_norm_eps,
                rope_theta=hf_config.rope_theta,
                tie_word_embeddings=hf_config.tie_word_embeddings,
            )
            
            # Create our model
            model = cls(config)
            
            # Load HuggingFace model
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
            
            # Copy weights
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
        """Load weights from HuggingFace Qwen3 model."""
        hf_state = hf_model.state_dict()
        
        # Weight mapping
        mapping = {}
        
        # Embeddings
        mapping["embed_tokens.weight"] = "model.embed_tokens.weight"
        
        # Layers
        for i in range(self.config.num_hidden_layers):
            our_prefix = f"layers.{i}"
            hf_prefix = f"model.layers.{i}"
            
            # Norms
            mapping[f"{our_prefix}.input_layernorm.weight"] = f"{hf_prefix}.input_layernorm.weight"
            mapping[f"{our_prefix}.post_attention_layernorm.weight"] = f"{hf_prefix}.post_attention_layernorm.weight"
            
            # Attention (Qwen3 has no biases)
            mapping[f"{our_prefix}.self_attn.q_proj.weight"] = f"{hf_prefix}.self_attn.q_proj.weight"
            mapping[f"{our_prefix}.self_attn.k_proj.weight"] = f"{hf_prefix}.self_attn.k_proj.weight"
            mapping[f"{our_prefix}.self_attn.v_proj.weight"] = f"{hf_prefix}.self_attn.v_proj.weight"
            mapping[f"{our_prefix}.self_attn.o_proj.weight"] = f"{hf_prefix}.self_attn.o_proj.weight"
            
            # Qwen3 Q/K normalization
            mapping[f"{our_prefix}.self_attn.q_norm.weight"] = f"{hf_prefix}.self_attn.q_norm.weight"
            mapping[f"{our_prefix}.self_attn.k_norm.weight"] = f"{hf_prefix}.self_attn.k_norm.weight"
            
            # MLP
            mapping[f"{our_prefix}.mlp.gate_proj.weight"] = f"{hf_prefix}.mlp.gate_proj.weight"
            mapping[f"{our_prefix}.mlp.up_proj.weight"] = f"{hf_prefix}.mlp.up_proj.weight"
            mapping[f"{our_prefix}.mlp.down_proj.weight"] = f"{hf_prefix}.mlp.down_proj.weight"
        
        # Final norm and LM head
        mapping["norm.weight"] = "model.norm.weight"
        mapping["lm_head.weight"] = "lm_head.weight"
        
        # Copy weights
        new_state = {}
        for our_key, hf_key in mapping.items():
            if hf_key in hf_state:
                new_state[our_key] = hf_state[hf_key]
            else:
                # Handle tied weights
                if our_key == "lm_head.weight" and self.config.tie_word_embeddings:
                    continue  # Will be tied to embed_tokens
                print(f"Warning: {hf_key} not found in HF model")
        
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        
        # Filter out expected missing keys (tied weights)
        if self.config.tie_word_embeddings:
            missing = [k for k in missing if k != "lm_head.weight"]
        
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
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "head_dim": self.config.head_dim,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rms_norm_eps": self.config.rms_norm_eps,
            "rope_theta": self.config.rope_theta,
            "tie_word_embeddings": self.config.tie_word_embeddings,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), save_dir / "model.pt")


if __name__ == "__main__":
    # Test the language model
    print("Testing LanguageModel (Qwen3-0.6B architecture)...")
    
    config = LLMConfig()
    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"GQA: {config.num_attention_heads} Q heads, {config.num_key_value_heads} KV heads")
    print(f"Head dim: {config.head_dim} (explicit)")
    print(f"Q proj dim: {config.num_attention_heads * config.head_dim}")
    print(f"KV proj dim: {config.num_key_value_heads * config.head_dim}")
    
    model = LanguageModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count / 1e6:.1f}M params ({param_count / 1e9:.2f}B)")
    
    # Test forward pass
    dummy_input = torch.randint(0, 1000, (2, 64))
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [2, 64, 151936]
    
    # Test with cache
    output, cache = model(dummy_input, use_cache=True)
    print(f"Cache layers: {len(cache)}")
    print(f"Cache shape: K={cache[0][0].shape}, V={cache[0][1].shape}")
    
    # Test generation step
    next_token = torch.randint(0, 1000, (2, 1))
    output2, cache2 = model(next_token, past_key_values=cache, use_cache=True)
    print(f"Next token output shape: {output2.shape}")  # Should be [2, 1, 151936]
    
    print("✓ LanguageModel tests passed!")
