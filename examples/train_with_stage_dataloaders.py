"""
Example script for training Molmo2 with stage-specific dataloaders.

Demonstrates:
1. How to set up dataloaders for each training stage
2. How to configure tokenizer and image processor
3. How to run training with proper data mixing

Usage:
    # Stage 1: Pre-training
    python examples/train_with_stage_dataloaders.py --stage 1 --batch-size 32
    
    # Stage 2: SFT
    python examples/train_with_stage_dataloaders.py --stage 2 --batch-size 16
    
    # Stage 3: Long-context
    python examples/train_with_stage_dataloaders.py --stage 3 --batch-size 4
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from torchvision import transforms
import torch

from data.stage_dataloaders import get_stage_dataloader, get_all_stage_dataloaders


def setup_tokenizer_and_processor():
    """
    Set up tokenizer and image processor.
    
    Returns:
        Tuple of (tokenizer, image_processor)
    """
    # Load tokenizer (Qwen3-0.6B as base)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Add special tokens for multimodal inputs
    special_tokens = {
        "additional_special_tokens": [
            "<image>",
            "<video>",
            "<point>",
            "<timestamp>",
            "<pad>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Set up image processor (simple version, replace with actual processor)
    image_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return tokenizer, image_processor


def train_stage_1(
    data_root: str = "./data/molmo2_datasets",
    batch_size: int = 32,
    num_workers: int = 4,
    num_epochs: int = 1,
):
    """
    Train Stage 1: Pre-training.
    
    Configuration:
    - 60% Dense captioning
    - 30% Image pointing
    - 10% NLP data
    - Sequence length: 4,096 tokens
    """
    print("\n" + "="*60)
    print("STAGE 1: PRE-TRAINING")
    print("="*60)
    
    # Setup tokenizer and processor
    tokenizer, image_processor = setup_tokenizer_and_processor()
    
    # Create dataloader
    train_loader = get_stage_dataloader(
        stage=1,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="train",
    )
    
    # Training loop (simplified)
    print(f"\nStarting training for {num_epochs} epoch(s)...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # batch contains:
            # - input_ids: [batch_size, seq_length]
            # - attention_mask: [batch_size, seq_length]
            # - pixel_values: [batch_size, num_images, 3, H, W]
            # - loss_weights: [batch_size, seq_length]
            
            # Your training code here
            # outputs = model(batch)
            # loss = compute_loss(outputs, batch, loss_weights=batch["loss_weights"])
            # loss.backward()
            # optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")
            
            # For demo, only process a few batches
            if batch_idx >= 2:
                break
        
        # For demo, only run 1 epoch
        break
    
    print("\n✓ Stage 1 training complete!")


def train_stage_2(
    data_root: str = "./data/molmo2_datasets",
    batch_size: int = 16,
    num_workers: int = 4,
    num_epochs: int = 1,
):
    """
    Train Stage 2: Supervised Fine-Tuning.
    
    Configuration:
    - Sqrt-proportional sampling across 100+ datasets
    - Sequence length: 4,096 tokens
    - Frame limit: 128 frames (videos)
    - Frame rate: 2 fps
    """
    print("\n" + "="*60)
    print("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
    print("="*60)
    
    # Setup tokenizer and processor
    tokenizer, image_processor = setup_tokenizer_and_processor()
    
    # Create dataloader
    train_loader = get_stage_dataloader(
        stage=2,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="train",
    )
    
    # Training loop (simplified)
    print(f"\nStarting training for {num_epochs} epoch(s)...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Your training code here
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")
            
            # For demo, only process a few batches
            if batch_idx >= 2:
                break
        
        # For demo, only run 1 epoch
        break
    
    print("\n✓ Stage 2 training complete!")


def train_stage_3(
    data_root: str = "./data/molmo2_datasets",
    batch_size: int = 4,
    num_workers: int = 4,
    num_epochs: int = 1,
):
    """
    Train Stage 3: Long-context SFT.
    
    Configuration:
    - Sequence length: 36,864 tokens (9x longer)
    - Frame limit: 384 frames (3x more)
    - Frame rate: 2 fps
    - Requires: Context Parallelism (CP)
    """
    print("\n" + "="*60)
    print("STAGE 3: LONG-CONTEXT SFT")
    print("="*60)
    print("⚠ This stage requires Context Parallelism (CP) and Ulysses attention")
    print("⚠ Make sure you have sufficient GPU memory (A100 80GB recommended)")
    
    # Setup tokenizer and processor
    tokenizer, image_processor = setup_tokenizer_and_processor()
    
    # Create dataloader
    train_loader = get_stage_dataloader(
        stage=3,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="train",
    )
    
    # Training loop (simplified)
    print(f"\nStarting training for {num_epochs} epoch(s)...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Your training code here (with CP)
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")
            
            # For demo, only process a few batches
            if batch_idx >= 2:
                break
        
        # For demo, only run 1 epoch
        break
    
    print("\n✓ Stage 3 training complete!")


def train_all_stages(
    data_root: str = "./data/molmo2_datasets",
    num_workers: int = 4,
):
    """
    Train all 3 stages sequentially.
    
    This demonstrates the complete Molmo2 training pipeline.
    """
    print("\n" + "="*70)
    print("COMPLETE MOLMO2 TRAINING PIPELINE (ALL 3 STAGES)")
    print("="*70)
    
    # Stage 1: Pre-training
    train_stage_1(data_root=data_root, batch_size=32, num_workers=num_workers, num_epochs=1)
    
    # Stage 2: SFT
    train_stage_2(data_root=data_root, batch_size=16, num_workers=num_workers, num_epochs=1)
    
    # Stage 3: Long-context
    train_stage_3(data_root=data_root, batch_size=4, num_workers=num_workers, num_epochs=1)
    
    print("\n" + "="*70)
    print("✓ ALL 3 STAGES COMPLETE!")
    print("="*70)


def inspect_dataloader(stage: int, data_root: str = "./data/molmo2_datasets"):
    """
    Inspect a dataloader without training.
    
    Useful for debugging and understanding data format.
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING STAGE {stage} DATALOADER")
    print("="*60)
    
    # Setup tokenizer and processor
    tokenizer, image_processor = setup_tokenizer_and_processor()
    
    # Create dataloader
    loader = get_stage_dataloader(
        stage=stage,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        batch_size=2,
        num_workers=0,  # Single worker for debugging
        split="train",
    )
    
    print(f"\nDataloader info:")
    print(f"  Total batches: {len(loader)}")
    print(f"  Batch size: {loader.batch_size}")
    
    # Get first batch
    print("\nFetching first batch...")
    try:
        batch = next(iter(loader))
        print("\nBatch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
    except Exception as e:
        print(f"\n✗ Error fetching batch: {e}")
        print("  Make sure datasets are downloaded and available in data_root")


def main():
    parser = argparse.ArgumentParser(
        description="Train Molmo2 with stage-specific dataloaders"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Training stage (1=pre-training, 2=SFT, 3=long-context)"
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Train all 3 stages sequentially"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect dataloader without training"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/molmo2_datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (uses stage-specific defaults if not specified)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_stages and args.stage is None and not args.inspect:
        parser.error("Must specify --stage, --all-stages, or --inspect")
    
    # Run appropriate function
    if args.inspect:
        if args.stage is None:
            parser.error("--inspect requires --stage")
        inspect_dataloader(stage=args.stage, data_root=args.data_root)
    
    elif args.all_stages:
        train_all_stages(
            data_root=args.data_root,
            num_workers=args.num_workers,
        )
    
    elif args.stage == 1:
        train_stage_1(
            data_root=args.data_root,
            batch_size=args.batch_size or 32,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
        )
    
    elif args.stage == 2:
        train_stage_2(
            data_root=args.data_root,
            batch_size=args.batch_size or 16,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
        )
    
    elif args.stage == 3:
        train_stage_3(
            data_root=args.data_root,
            batch_size=args.batch_size or 4,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
        )


if __name__ == "__main__":
    main()
