#!/usr/bin/env python3
"""
Download and convert model weights for NanoMolmo2.

Downloads weights from HuggingFace once and saves in optimized local format.
Subsequent loads are ~30x faster using pure PyTorch.

Usage:
    python scripts/download_model_weights.py                    # Download all
    python scripts/download_model_weights.py --vision-only      # Vision encoder only
    python scripts/download_model_weights.py --llm-only         # LLM only
    python scripts/download_model_weights.py --cache-dir ./my_cache
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def download_vision_encoder(cache_dir: Path, force: bool = False) -> Path:
    """Download and convert SigLIP2 vision encoder weights."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    
    cache_path = cache_dir / "siglip2_so400m_384.pt"
    
    if cache_path.exists() and not force:
        print(f"✓ Vision encoder already cached: {cache_path}")
        return cache_path
    
    print("Downloading SigLIP2 So400m vision encoder...")
    model_name = "google/siglip2-so400m-patch14-384"
    
    # Load config
    full_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf_config = full_config.vision_config if hasattr(full_config, 'vision_config') else full_config
    
    config = {
        "image_size": hf_config.image_size,
        "patch_size": hf_config.patch_size,
        "num_channels": hf_config.num_channels,
        "hidden_size": hf_config.hidden_size,
        "intermediate_size": hf_config.intermediate_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "layer_norm_eps": hf_config.layer_norm_eps,
        "attention_dropout": getattr(hf_config, 'attention_dropout', 0.0),
    }
    
    print(f"  Config: {config['hidden_size']}d, {config['num_hidden_layers']} layers")
    
    # Download weights directly
    print("  Downloading safetensors...")
    weights_file = hf_hub_download(model_name, "model.safetensors")
    hf_state = load_file(weights_file)
    
    # Convert to our format
    print("  Converting weights...")
    state_dict = convert_vision_weights(hf_state, config["num_hidden_layers"])
    
    # Save
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": state_dict}, cache_path)
    
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"✓ Vision encoder saved: {cache_path} ({size_mb:.1f} MB)")
    return cache_path


def convert_vision_weights(hf_state: dict, num_layers: int) -> dict:
    """Convert HuggingFace SigLIP weights to our format."""
    mapping = {
        "embeddings.projection.weight": "vision_model.embeddings.patch_embedding.weight",
        "embeddings.projection.bias": "vision_model.embeddings.patch_embedding.bias",
        "embeddings.position_embedding.weight": "vision_model.embeddings.position_embedding.weight",
        "post_layernorm.weight": "vision_model.post_layernorm.weight",
        "post_layernorm.bias": "vision_model.post_layernorm.bias",
    }
    
    for i in range(num_layers):
        hf_prefix = f"vision_model.encoder.layers.{i}"
        our_prefix = f"encoder_layers.{i}"
        
        mapping.update({
            f"{our_prefix}.layer_norm1.weight": f"{hf_prefix}.layer_norm1.weight",
            f"{our_prefix}.layer_norm1.bias": f"{hf_prefix}.layer_norm1.bias",
            f"{our_prefix}.layer_norm2.weight": f"{hf_prefix}.layer_norm2.weight",
            f"{our_prefix}.layer_norm2.bias": f"{hf_prefix}.layer_norm2.bias",
            f"{our_prefix}.self_attn.q_proj.weight": f"{hf_prefix}.self_attn.q_proj.weight",
            f"{our_prefix}.self_attn.q_proj.bias": f"{hf_prefix}.self_attn.q_proj.bias",
            f"{our_prefix}.self_attn.k_proj.weight": f"{hf_prefix}.self_attn.k_proj.weight",
            f"{our_prefix}.self_attn.k_proj.bias": f"{hf_prefix}.self_attn.k_proj.bias",
            f"{our_prefix}.self_attn.v_proj.weight": f"{hf_prefix}.self_attn.v_proj.weight",
            f"{our_prefix}.self_attn.v_proj.bias": f"{hf_prefix}.self_attn.v_proj.bias",
            f"{our_prefix}.self_attn.out_proj.weight": f"{hf_prefix}.self_attn.out_proj.weight",
            f"{our_prefix}.self_attn.out_proj.bias": f"{hf_prefix}.self_attn.out_proj.bias",
            f"{our_prefix}.mlp.fc1.weight": f"{hf_prefix}.mlp.fc1.weight",
            f"{our_prefix}.mlp.fc1.bias": f"{hf_prefix}.mlp.fc1.bias",
            f"{our_prefix}.mlp.fc2.weight": f"{hf_prefix}.mlp.fc2.weight",
            f"{our_prefix}.mlp.fc2.bias": f"{hf_prefix}.mlp.fc2.bias",
        })
    
    state_dict = {}
    for our_key, hf_key in mapping.items():
        if hf_key in hf_state:
            state_dict[our_key] = hf_state[hf_key]
        else:
            print(f"  Warning: {hf_key} not found")
    
    return state_dict


def download_llm(cache_dir: Path, force: bool = False) -> Path:
    """Download and convert Qwen3-0.6B-Base weights."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoConfig
    
    cache_path = cache_dir / "qwen3_0.6b_base.pt"
    
    if cache_path.exists() and not force:
        print(f"✓ LLM already cached: {cache_path}")
        return cache_path
    
    print("Downloading Qwen3-0.6B-Base...")
    model_name = "Qwen/Qwen3-0.6B-Base"
    
    # Load config
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    config = {
        "vocab_size": hf_config.vocab_size,
        "hidden_size": hf_config.hidden_size,
        "intermediate_size": hf_config.intermediate_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "head_dim": getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
        "max_position_embeddings": hf_config.max_position_embeddings,
        "rms_norm_eps": hf_config.rms_norm_eps,
        "rope_theta": hf_config.rope_theta,
        "tie_word_embeddings": hf_config.tie_word_embeddings,
    }
    
    print(f"  Config: {config['hidden_size']}d, {config['num_hidden_layers']} layers, {config['vocab_size']} vocab")
    
    # Download weights - check for sharded files
    print("  Downloading safetensors...")
    repo_files = list_repo_files(model_name)
    safetensor_files = [f for f in repo_files if f.endswith('.safetensors')]
    
    hf_state = {}
    for sf in safetensor_files:
        print(f"    Loading {sf}...")
        local_file = hf_hub_download(model_name, sf)
        shard_state = load_file(local_file)
        hf_state.update(shard_state)
    
    # Convert to our format
    print("  Converting weights...")
    state_dict = convert_llm_weights(hf_state, config)
    
    # Save
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": state_dict}, cache_path)
    
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"✓ LLM saved: {cache_path} ({size_mb:.1f} MB)")
    return cache_path


def convert_llm_weights(hf_state: dict, config: dict) -> dict:
    """Convert HuggingFace Qwen3 weights to our format."""
    mapping = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
    }
    
    for i in range(config["num_hidden_layers"]):
        our_prefix = f"layers.{i}"
        hf_prefix = f"model.layers.{i}"
        
        mapping.update({
            f"{our_prefix}.input_layernorm.weight": f"{hf_prefix}.input_layernorm.weight",
            f"{our_prefix}.post_attention_layernorm.weight": f"{hf_prefix}.post_attention_layernorm.weight",
            f"{our_prefix}.self_attn.q_proj.weight": f"{hf_prefix}.self_attn.q_proj.weight",
            f"{our_prefix}.self_attn.k_proj.weight": f"{hf_prefix}.self_attn.k_proj.weight",
            f"{our_prefix}.self_attn.v_proj.weight": f"{hf_prefix}.self_attn.v_proj.weight",
            f"{our_prefix}.self_attn.o_proj.weight": f"{hf_prefix}.self_attn.o_proj.weight",
            f"{our_prefix}.self_attn.q_norm.weight": f"{hf_prefix}.self_attn.q_norm.weight",
            f"{our_prefix}.self_attn.k_norm.weight": f"{hf_prefix}.self_attn.k_norm.weight",
            f"{our_prefix}.mlp.gate_proj.weight": f"{hf_prefix}.mlp.gate_proj.weight",
            f"{our_prefix}.mlp.up_proj.weight": f"{hf_prefix}.mlp.up_proj.weight",
            f"{our_prefix}.mlp.down_proj.weight": f"{hf_prefix}.mlp.down_proj.weight",
        })
    
    state_dict = {}
    for our_key, hf_key in mapping.items():
        if hf_key in hf_state:
            state_dict[our_key] = hf_state[hf_key]
        else:
            print(f"  Warning: {hf_key} not found")
    
    # Handle tied weights
    if config["tie_word_embeddings"]:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]
    
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Download model weights for NanoMolmo2")
    parser.add_argument("--cache-dir", type=str, default="./checkpoints",
                        help="Directory to save weights (default: ./checkpoints)")
    parser.add_argument("--vision-only", action="store_true",
                        help="Download only vision encoder")
    parser.add_argument("--llm-only", action="store_true",
                        help="Download only LLM")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cached")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    print("=" * 60)
    print("NanoMolmo2 Model Weight Downloader")
    print("=" * 60)
    print(f"Cache directory: {cache_dir.absolute()}")
    print()
    
    try:
        if not args.llm_only:
            download_vision_encoder(cache_dir, args.force)
            print()
        
        if not args.vision_only:
            download_llm(cache_dir, args.force)
            print()
        
        print("=" * 60)
        print("Download complete!")
        print(f"Weights saved to: {cache_dir.absolute()}")
        print()
        print("To use in code:")
        print("  from models import VisionEncoder, LanguageModel")
        print(f'  vision = VisionEncoder.from_pretrained(cache_dir="{cache_dir}")')
        print(f'  llm = LanguageModel.from_pretrained(cache_dir="{cache_dir}")')
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
