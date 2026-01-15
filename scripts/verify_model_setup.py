#!/usr/bin/env python3
"""
Verify nanoMolmo2 model setup and estimate memory usage.

Usage:
    python scripts/verify_model_setup.py
"""

import torch
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

def count_parameters(model, trainable_only=False):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def get_model_memory(model, dtype=torch.float32):
    """Estimate model memory in GB."""
    params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4 if dtype == torch.float32 else 2  # fp32 or fp16/bf16
    return params * bytes_per_param / (1024**3)  # Convert to GB

def verify_setup():
    """Verify model components can be loaded."""
    print("="*70)
    print("nanoMolmo2 Model Setup Verification")
    print("="*70)
    print()
    
    # 1. Vision Encoder
    print("1. Loading Vision Encoder (CLIP ViT)...")
    try:
        vision_model_name = "openai/clip-vit-large-patch14-336"
        vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        
        # Freeze it
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()
        
        vision_params_total = count_parameters(vision_encoder, trainable_only=False)
        vision_params_trainable = count_parameters(vision_encoder, trainable_only=True)
        vision_memory = get_model_memory(vision_encoder, dtype=torch.float16)
        
        print(f"   ✓ Loaded: {vision_model_name}")
        print(f"   Total parameters: {vision_params_total / 1e6:.1f}M")
        print(f"   Trainable parameters: {vision_params_trainable / 1e6:.1f}M (FROZEN)")
        print(f"   Memory (fp16): {vision_memory:.2f} GB")
        print(f"   Output dim: {vision_encoder.config.hidden_size}")
        print()
    except Exception as e:
        print(f"   ✗ Failed to load vision encoder: {e}")
        return False
    
    # 2. Language Model
    print("2. Loading Language Model (Qwen3-0.6B)...")
    try:
        llm_model_name = "Qwen/Qwen2.5-0.5B"
        llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": ["<image>", "<video>", "<point>", "<timestamp>"]
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
        llm.resize_token_embeddings(len(tokenizer))
        
        llm_params_total = count_parameters(llm, trainable_only=False)
        llm_params_trainable = count_parameters(llm, trainable_only=True)
        llm_memory = get_model_memory(llm, dtype=torch.bfloat16)
        
        print(f"   ✓ Loaded: {llm_model_name}")
        print(f"   Total parameters: {llm_params_total / 1e6:.1f}M")
        print(f"   Trainable parameters: {llm_params_trainable / 1e6:.1f}M")
        print(f"   Memory (bf16): {llm_memory:.2f} GB")
        print(f"   Hidden dim: {llm.config.hidden_size}")
        print(f"   Vocab size: {len(tokenizer)} (added {num_added} special tokens)")
        print()
    except Exception as e:
        print(f"   ✗ Failed to load language model: {e}")
        return False
    
    # 3. Connector
    print("3. Creating Multimodal Connector...")
    try:
        import torch.nn as nn
        
        vision_dim = vision_encoder.config.hidden_size  # 1024
        llm_dim = llm.config.hidden_size  # 896
        
        # Linear connector
        connector_linear = nn.Linear(vision_dim, llm_dim)
        connector_linear_params = count_parameters(connector_linear)
        connector_linear_memory = get_model_memory(connector_linear, dtype=torch.bfloat16)
        
        # MLP connector
        connector_mlp = nn.Sequential(
            nn.Linear(vision_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, llm_dim)
        )
        connector_mlp_params = count_parameters(connector_mlp)
        connector_mlp_memory = get_model_memory(connector_mlp, dtype=torch.bfloat16)
        
        print(f"   ✓ Connector dimensions: {vision_dim} → {llm_dim}")
        print(f"   Linear option:")
        print(f"     Parameters: {connector_linear_params / 1e6:.2f}M")
        print(f"     Memory: {connector_linear_memory * 1000:.1f} MB")
        print(f"   MLP option (2-layer):")
        print(f"     Parameters: {connector_mlp_params / 1e6:.2f}M")
        print(f"     Memory: {connector_mlp_memory * 1000:.1f} MB")
        print()
    except Exception as e:
        print(f"   ✗ Failed to create connector: {e}")
        return False
    
    # 4. Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    total_params = vision_params_total + connector_linear_params + llm_params_total
    trainable_params = vision_params_trainable + connector_linear_params + llm_params_trainable
    
    print(f"\nModel Components:")
    print(f"  Vision Encoder:   {vision_params_total / 1e6:7.1f}M  (🔒 FROZEN)")
    print(f"  Connector:        {connector_linear_params / 1e6:7.1f}M  (✏️  TRAINABLE)")
    print(f"  Language Model:   {llm_params_total / 1e6:7.1f}M  (✏️  TRAINABLE)")
    print(f"  " + "-"*40)
    print(f"  Total:            {total_params / 1e6:7.1f}M")
    print(f"  Trainable:        {trainable_params / 1e6:7.1f}M ({trainable_params/total_params*100:.1f}%)")
    
    print(f"\nMemory Estimates (per GPU):")
    print(f"  Vision Encoder (fp16, frozen):  {vision_memory:.2f} GB")
    print(f"    - No gradients needed!")
    print(f"    - No optimizer states!")
    print(f"  Connector (bf16):               {connector_linear_memory * 1000:.0f} MB")
    print(f"  Language Model (bf16):          {llm_memory:.2f} GB")
    print(f"  Activations (batch=32):         ~4 GB")
    print(f"  Batch data:                     ~4 GB")
    print(f"  " + "-"*40)
    print(f"  Estimated total:                ~{vision_memory + llm_memory + 8:.0f} GB")
    
    print(f"\nHardware Requirements:")
    if vision_memory + llm_memory + 8 < 40:
        print(f"  ✓ Fits in A100 40GB!")
        print(f"  ✓ Can use batch_size=32")
    else:
        print(f"  ⚠️  Requires A100 80GB")
        print(f"  ⚠️  Or reduce batch size")
    
    print(f"\nTraining Speed Benefit:")
    print(f"  Frozen vision encoder → ~30-40% faster training")
    print(f"  Skip vision backward pass = fewer FLOPs")
    
    print("\n" + "="*70)
    print("✓ All components verified successfully!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Download datasets: bash scripts/download_all.sh")
    print("  2. Implement model: See MODEL_ARCHITECTURE.md")
    print("  3. Start training: python examples/train_with_stage_dataloaders.py --stage 1")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    success = verify_setup()
    sys.exit(0 if success else 1)
