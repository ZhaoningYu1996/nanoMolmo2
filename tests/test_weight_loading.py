"""
Test weight loading for NanoMolmo2 models.

Verifies that:
1. Vision encoder (SigLIP2 So400m) loads correctly
2. Language model (Qwen3-0.6B-Base) loads correctly
3. Forward passes produce valid outputs

Uses local cache in ./checkpoints if available (fast),
falls back to HuggingFace download if not (slow).
Run `python scripts/download_model_weights.py` to cache weights locally.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_vision_encoder_config():
    """Test VisionEncoder config matches SigLIP2."""
    from models.vision_encoder import VisionConfig, VisionEncoder
    
    config = VisionConfig()
    print("=== Vision Encoder Config ===")
    print(f"  image_size: {config.image_size}")
    print(f"  patch_size: {config.patch_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_patches: {config.num_patches}")
    
    # Expected values from google/siglip2-so400m-patch14-384
    assert config.image_size == 384, f"Expected 384, got {config.image_size}"
    assert config.patch_size == 14, f"Expected 14, got {config.patch_size}"
    assert config.hidden_size == 1152, f"Expected 1152, got {config.hidden_size}"
    assert config.intermediate_size == 4304, f"Expected 4304, got {config.intermediate_size}"
    assert config.num_hidden_layers == 27, f"Expected 27, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 16, f"Expected 16, got {config.num_attention_heads}"
    print("✓ Vision encoder config matches SigLIP2 So400m")


def test_llm_config():
    """Test LLMConfig matches Qwen3-0.6B-Base."""
    from models.language_model import LLMConfig, LanguageModel
    
    config = LLMConfig()
    print("\n=== Language Model Config ===")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  Q proj output: {config.num_attention_heads * config.head_dim}")
    print(f"  KV proj output: {config.num_key_value_heads * config.head_dim}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  rope_theta: {config.rope_theta}")
    
    # Expected values from Qwen/Qwen3-0.6B-Base
    assert config.hidden_size == 1024, f"Expected 1024, got {config.hidden_size}"
    assert config.intermediate_size == 3072, f"Expected 3072, got {config.intermediate_size}"
    assert config.num_hidden_layers == 28, f"Expected 28, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 16, f"Expected 16, got {config.num_attention_heads}"
    assert config.num_key_value_heads == 8, f"Expected 8, got {config.num_key_value_heads}"
    assert config.head_dim == 128, f"Expected 128, got {config.head_dim}"
    assert config.max_position_embeddings == 32768, f"Expected 32768 for Base model"
    # Verify projection dimensions (Qwen3 uses larger Q projection than hidden_size)
    assert config.num_attention_heads * config.head_dim == 2048, "Q proj should be 2048"
    assert config.num_key_value_heads * config.head_dim == 1024, "KV proj should be 1024"
    print("✓ LLM config matches Qwen3-0.6B-Base")


def test_vision_encoder_forward():
    """Test VisionEncoder forward pass."""
    from models.vision_encoder import VisionConfig, VisionEncoder
    
    print("\n=== Vision Encoder Forward Pass ===")
    config = VisionConfig()
    model = VisionEncoder(config)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.1f}M")
    
    # Test with random input
    batch_size = 2
    x = torch.randn(batch_size, 3, config.image_size, config.image_size)
    
    with torch.no_grad():
        output = model(x)
    
    expected_shape = (batch_size, config.num_patches, config.hidden_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print("✓ Vision encoder forward pass works")


def test_llm_forward():
    """Test LanguageModel forward pass."""
    from models.language_model import LLMConfig, LanguageModel
    
    print("\n=== Language Model Forward Pass ===")
    config = LLMConfig()
    model = LanguageModel(config)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.1f}M ({param_count / 1e9:.2f}B)")
    
    # Test with random input
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(x)
    
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print("✓ LLM forward pass works")


def test_vision_encoder_load_weights():
    """Test loading SigLIP2 weights (uses local cache if available)."""
    from models.vision_encoder import VisionEncoder
    
    print("\n=== Loading SigLIP2 Weights ===")
    
    try:
        model = VisionEncoder.from_pretrained()  # Uses default cache_dir="./checkpoints"
        print(f"  Model loaded successfully")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count / 1e6:.1f}M")
        
        # Test forward pass with loaded weights
        x = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = model(x)
        
        print(f"  Forward pass output: {output.shape}")
        assert output.shape == (1, 729, 1152)
        print("✓ SigLIP2 weights loaded and forward pass works")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_llm_load_weights():
    """Test loading Qwen3-0.6B-Base weights (uses local cache if available)."""
    from models.language_model import LanguageModel
    
    print("\n=== Loading Qwen3-0.6B-Base Weights ===")
    
    try:
        model = LanguageModel.from_pretrained()  # Uses default cache_dir="./checkpoints"
        print(f"  Model loaded successfully")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count / 1e6:.1f}M ({param_count / 1e9:.2f}B)")
        
        # Test forward pass with loaded weights
        x = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            output = model(x)
        
        print(f"  Forward pass output: {output.shape}")
        assert output.shape == (1, 32, 151936)
        print("✓ Qwen3-0.6B-Base weights loaded and forward pass works")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_with_hf_outputs():
    """Compare our model outputs with HuggingFace model outputs."""
    print("\n=== Comparing with HuggingFace Outputs ===")
    
    try:
        from transformers import AutoModelForCausalLM
        from models.language_model import LanguageModel
        
        print("Loading HuggingFace model (Qwen3-0.6B-Base)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        hf_model.eval()
        
        print("Loading our model...")
        our_model = LanguageModel.from_pretrained("Qwen/Qwen3-0.6B-Base")
        our_model.eval()
        our_model = our_model.float()
        
        # Test input
        input_ids = torch.randint(0, 1000, (1, 16))
        
        with torch.no_grad():
            hf_output = hf_model(input_ids).logits
            our_output = our_model(input_ids)
        
        print(f"  HF output shape: {hf_output.shape}")
        print(f"  Our output shape: {our_output.shape}")
        
        # Check if outputs are close
        max_diff = (hf_output - our_output).abs().max().item()
        mean_diff = (hf_output - our_output).abs().mean().item()
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✓ Outputs match HuggingFace model")
            return True
        else:
            print("⚠ Outputs differ from HuggingFace model")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NanoMolmo2 Model Implementations")
    print("=" * 60)
    
    # Test configs
    test_vision_encoder_config()
    test_llm_config()
    
    # Test forward passes (without loading weights)
    test_vision_encoder_forward()
    test_llm_forward()
    
    # Test weight loading
    print("\n" + "=" * 60)
    print("Testing Weight Loading (Local Cache → HuggingFace fallback)")
    print("=" * 60)
    
    vision_ok = test_vision_encoder_load_weights()
    llm_ok = test_llm_load_weights()
    
    if vision_ok and llm_ok:
        # Compare outputs
        test_compare_with_hf_outputs()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Vision Encoder: {'✓' if vision_ok else '✗'}")
    print(f"  Language Model: {'✓' if llm_ok else '✗'}")
