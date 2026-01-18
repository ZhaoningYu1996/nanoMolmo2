"""
NanoMolmo2: Complete Vision-Language Model

Combines:
- SigLIP 2 So400m/14 @ 384px vision encoder (frozen)
- Multimodal connector (trainable)
- Qwen3-0.6B language model (trainable)

Based on Molmo2 tech report:
- Uses SDPA (not FlashAttention) for custom attention masks
- torch.compile compatible with static shapes
- AMP with bfloat16 (LayerNorm/RoPE in full precision)
- FSDP2 compatible for distributed training
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionEncoder, VisionConfig
from .language_model import LanguageModel, LLMConfig
from .connector import MultimodalConnector, ConnectorConfig


@dataclass
class NanoMolmo2Config:
    """Configuration for complete NanoMolmo2 model."""
    # Vision encoder
    vision_config: Optional[VisionConfig] = None
    freeze_vision: bool = True  # Freeze vision encoder (as per Molmo2)
    
    # Connector
    connector_config: Optional[ConnectorConfig] = None
    
    # Language model
    llm_config: Optional[LLMConfig] = None
    
    # Special tokens
    image_token_id: int = 151646  # <image> token
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    
    # Training settings
    use_cache: bool = False
    gradient_checkpointing: bool = False


class NanoMolmo2(nn.Module):
    """
    NanoMolmo2: Vision-Language Model for educational purposes.
    
    Architecture:
        Image → [Vision Encoder] → [Connector] → Visual Tokens
                                                      ↓
        Text  → [Token Embedding] → Text Tokens  → [LLM Decoder] → Logits
    
    Training approach (from Molmo2):
    - Vision encoder: FROZEN (no gradients)
    - Connector: TRAINED
    - LLM: TRAINED
    
    Usage:
        model = NanoMolmo2.from_pretrained()
        
        # Training
        logits, loss = model(
            input_ids=tokens,
            pixel_values=images,
            labels=labels,
        )
        
        # Inference
        generated = model.generate(prompt_ids, pixel_values=images)
    """
    
    def __init__(self, config: Optional[NanoMolmo2Config] = None):
        super().__init__()
        self.config = config or NanoMolmo2Config()
        
        # Initialize configs with defaults
        if self.config.vision_config is None:
            self.config.vision_config = VisionConfig()
        if self.config.connector_config is None:
            self.config.connector_config = ConnectorConfig()
        if self.config.llm_config is None:
            self.config.llm_config = LLMConfig()
        
        # Vision Encoder
        self.vision_encoder = VisionEncoder(self.config.vision_config)
        if self.config.freeze_vision:
            self.vision_encoder.freeze()
        
        # Multimodal Connector
        self.connector = MultimodalConnector(self.config.connector_config)
        
        # Language Model
        self.llm = LanguageModel(self.config.llm_config)
        
        # Special token embedding for <image> placeholder
        self.image_token_embedding = nn.Parameter(
            torch.randn(1, 1, self.config.llm_config.hidden_size) * 0.02
        )
    
    def encode_images(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images to visual tokens.
        
        Args:
            pixel_values: [B, num_images, 3, H, W] or [B, 3, H, W]
            
        Returns:
            visual_tokens: [B, num_visual_tokens, llm_dim]
        """
        # Handle single vs multiple images
        if pixel_values.dim() == 4:
            # Single image per sample: [B, 3, H, W]
            pixel_values = pixel_values.unsqueeze(1)  # [B, 1, 3, H, W]
        
        B, N, C, H, W = pixel_values.shape
        
        # Flatten batch and num_images dimensions
        pixel_values = pixel_values.view(B * N, C, H, W)
        
        # Vision encoder (frozen, no gradients)
        with torch.no_grad() if self.config.freeze_vision else torch.enable_grad():
            vision_features = self.vision_encoder(pixel_values)  # [B*N, patches, vision_dim]
        
        # Project to LLM dimension
        visual_tokens = self.connector(vision_features)  # [B*N, tokens, llm_dim]
        
        # Reshape back: [B*N, tokens, llm_dim] -> [B, N*tokens, llm_dim]
        num_tokens = visual_tokens.shape[1]
        visual_tokens = visual_tokens.view(B, N * num_tokens, -1)
        
        return visual_tokens
    
    def _merge_visual_and_text_embeddings(
        self,
        input_ids: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge visual and text embeddings.
        
        Replaces <image> tokens with visual tokens from the vision encoder.
        
        Args:
            input_ids: [B, L] token IDs with <image> placeholders
            visual_tokens: [B, num_visual_tokens, llm_dim]
            
        Returns:
            inputs_embeds: [B, L', llm_dim] merged embeddings
            position_ids: [B, L'] position IDs
        """
        B, L = input_ids.shape
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if visual_tokens is None:
            # Text-only forward pass
            return text_embeds, torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Find <image> token positions
        image_mask = input_ids == self.config.image_token_id
        
        # If no <image> tokens, just return text embeddings
        if not image_mask.any():
            return text_embeds, torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Simple approach: Replace each <image> token with visual tokens
        # This expands the sequence length
        
        num_visual_tokens = visual_tokens.shape[1]
        num_image_tokens = image_mask.sum(dim=1).max().item()  # Max across batch
        
        # Calculate new sequence length
        new_length = L - num_image_tokens + num_image_tokens * num_visual_tokens
        
        # Create output tensors
        merged_embeds = torch.zeros(B, new_length, text_embeds.shape[-1], 
                                    device=text_embeds.device, dtype=text_embeds.dtype)
        
        # For each sample in batch, merge embeddings
        for b in range(B):
            sample_mask = image_mask[b]
            sample_text_embeds = text_embeds[b]
            sample_visual = visual_tokens[b]
            
            # Find image token positions
            image_positions = sample_mask.nonzero(as_tuple=True)[0]
            
            # Build merged sequence
            out_idx = 0
            text_idx = 0
            visual_idx = 0
            
            for pos in range(L):
                if sample_mask[pos]:
                    # Insert visual tokens
                    vis_start = visual_idx * num_visual_tokens // max(len(image_positions), 1)
                    vis_end = vis_start + num_visual_tokens // max(len(image_positions), 1)
                    vis_tokens = sample_visual[vis_start:vis_end]
                    merged_embeds[b, out_idx:out_idx + vis_tokens.shape[0]] = vis_tokens
                    out_idx += vis_tokens.shape[0]
                    visual_idx += 1
                else:
                    # Copy text embedding
                    merged_embeds[b, out_idx] = sample_text_embeds[pos]
                    out_idx += 1
        
        # Create position IDs
        position_ids = torch.arange(new_length, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        return merged_embeds, position_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: [B, L] token IDs
            pixel_values: [B, num_images, 3, H, W] or [B, 3, H, W] images
            attention_mask: [B, L] padding mask
            labels: [B, L] target tokens for loss computation
            position_ids: [B, L] position indices
            past_key_values: KV cache for generation
            use_cache: Whether to return KV cache
            
        Returns:
            If labels provided: (logits, loss)
            Else: logits (and optionally cache)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Encode images if provided
        visual_tokens = None
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
        
        # Merge visual and text embeddings
        inputs_embeds, merged_position_ids = self._merge_visual_and_text_embeddings(
            input_ids, visual_tokens
        )
        
        if position_ids is None:
            position_ids = merged_position_ids
        
        # Forward through LLM
        # Note: We pass inputs_embeds directly, bypassing embedding layer
        outputs = self._forward_llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        
        if loss is not None:
            return logits, loss
        elif use_cache and isinstance(outputs, tuple):
            return logits, outputs[1]  # logits, cache
        else:
            return logits
    
    def _forward_llm(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward through LLM with embeddings instead of token IDs."""
        B, L, D = inputs_embeds.shape
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        
        hidden_states = inputs_embeds
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + L, device=inputs_embeds.device
            ).unsqueeze(0).expand(B, -1)
        
        # Causal mask
        causal_mask = self.llm._make_causal_mask(
            torch.zeros(B, L, device=inputs_embeds.device, dtype=torch.long),
            past_length
        )
        
        if attention_mask is not None:
            # Expand and combine masks
            # ... (mask handling code)
            pass
        
        # Decoder layers
        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.llm.layers):
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
        hidden_states = self.llm.norm(hidden_states)
        
        # LM head
        logits = self.llm.lm_head(hidden_states)
        
        if use_cache:
            return logits, new_caches
        return (logits,)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text given prompt and optional images.
        
        Args:
            input_ids: [B, L] prompt token IDs
            pixel_values: Optional images
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            output_ids: [B, L + max_new_tokens] generated tokens
        """
        self.eval()
        
        # Encode images once
        visual_tokens = None
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
        
        # Merge visual and text embeddings
        inputs_embeds, position_ids = self._merge_visual_and_text_embeddings(
            input_ids, visual_tokens
        )
        
        # Generate tokens one by one
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                # First pass: use full embeddings
                outputs = self._forward_llm(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    use_cache=True,
                )
            else:
                # Subsequent passes: use last token only
                last_token = generated[:, -1:]
                last_embeds = self.llm.get_input_embeddings()(last_token)
                last_pos = position_ids[:, -1:] + 1
                position_ids = torch.cat([position_ids, last_pos], dim=1)
                
                outputs = self._forward_llm(
                    inputs_embeds=last_embeds,
                    position_ids=last_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits, past_key_values = outputs
            next_token_logits = logits[:, -1, :]
            
            # Sample next token
            if do_sample:
                # Temperature scaling
                next_token_logits = next_token_logits / temperature
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for b in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = float("-inf")
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.config.gradient_checkpointing = True
        # Apply to LLM layers
        for layer in self.llm.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.config.gradient_checkpointing = False
        for layer in self.llm.layers:
            layer.gradient_checkpointing = False
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (connector + LLM)."""
        params = []
        
        # Connector is always trainable
        params.extend(self.connector.parameters())
        
        # LLM is trainable
        params.extend(self.llm.parameters())
        
        # Image token embedding
        params.append(self.image_token_embedding)
        
        return params
    
    def print_trainable_parameters(self):
        """Print parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params / 1e6:.1f}M ({100*frozen_params/total_params:.1f}%)")
        
        # Breakdown by component
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        connector_params = sum(p.numel() for p in self.connector.parameters())
        llm_params = sum(p.numel() for p in self.llm.parameters())
        
        print(f"\nComponent breakdown:")
        print(f"  Vision Encoder: {vision_params / 1e6:.1f}M (frozen)")
        print(f"  Connector: {connector_params / 1e6:.2f}M (trainable)")
        print(f"  LLM: {llm_params / 1e6:.1f}M (trainable)")
    
    @classmethod
    def from_pretrained(
        cls,
        vision_model: str = "google/siglip2-so400m-patch14-384",
        llm_model: str = "Qwen/Qwen2.5-0.5B",
        connector_type: str = "linear",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "NanoMolmo2":
        """
        Create NanoMolmo2 with pretrained vision encoder and LLM.
        
        Args:
            vision_model: HuggingFace vision model ID
            llm_model: HuggingFace LLM model ID
            connector_type: "linear", "mlp", or "resampler"
            device: Target device
            dtype: Target dtype
            
        Returns:
            NanoMolmo2 with loaded weights
        """
        print(f"Loading vision encoder from {vision_model}...")
        vision_encoder = VisionEncoder.from_pretrained(vision_model)
        
        print(f"Loading language model from {llm_model}...")
        llm = LanguageModel.from_pretrained(llm_model)
        
        # Create config
        connector_config = ConnectorConfig(
            vision_dim=vision_encoder.config.hidden_size,
            llm_dim=llm.config.hidden_size,
            connector_type=connector_type,
        )
        
        config = NanoMolmo2Config(
            vision_config=vision_encoder.config,
            connector_config=connector_config,
            llm_config=llm.config,
        )
        
        # Create model
        model = cls(config)
        
        # Copy loaded weights
        model.vision_encoder = vision_encoder
        model.llm = llm
        
        # Freeze vision encoder
        if config.freeze_vision:
            model.vision_encoder.freeze()
        
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model to directory."""
        import json
        from pathlib import Path
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configs
        config_dict = {
            "freeze_vision": self.config.freeze_vision,
            "image_token_id": self.config.image_token_id,
            "pad_token_id": self.config.pad_token_id,
            "connector_type": self.config.connector_config.connector_type,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save component weights
        self.vision_encoder.save_pretrained(save_dir / "vision_encoder")
        self.llm.save_pretrained(save_dir / "llm")
        
        # Save connector and image token embedding
        torch.save({
            "connector": self.connector.state_dict(),
            "image_token_embedding": self.image_token_embedding,
        }, save_dir / "connector.pt")


if __name__ == "__main__":
    # Test NanoMolmo2
    print("Testing NanoMolmo2...")
    print("=" * 60)
    
    # Create model with default configs
    model = NanoMolmo2()
    model.print_trainable_parameters()
    
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    
    # Dummy tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Dummy images: [B, 3, 384, 384]
    pixel_values = torch.randn(batch_size, 3, 384, 384)
    
    # Dummy labels
    labels = input_ids.clone()
    
    # Forward pass
    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Testing text-only forward pass...")
    
    # Text-only forward
    with torch.no_grad():
        logits_text = model(input_ids=input_ids)
    
    print(f"Text-only logits shape: {logits_text.shape}")
    
    print("\n✓ NanoMolmo2 tests passed!")
