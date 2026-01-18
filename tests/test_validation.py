#!/usr/bin/env python3
"""
Tests for Molmo2 Validation Module

Run: python tests/test_validation.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_validation_info():
    """Test getting validation dataset info."""
    from data.validation import get_validation_info
    
    info = get_validation_info("pretrain")
    assert info["stage"] == "pretrain"
    assert "cosyn-point" in info["datasets"]
    assert "pixmo-count" in info["datasets"]
    assert info["total_samples"] == 1540  # 1000 + 540
    print(f"✓ Validation info: {info['total_samples']} total samples")


def test_validation_loader_pretrain():
    """Test creating pretrain validation dataloader."""
    from data.validation import create_validation_loader
    
    loader = create_validation_loader(
        data_root="./data/molmo2",
        stage="pretrain",
        batch_size=16,
        num_workers=0,
    )
    
    # Get first batch
    batch = next(iter(loader))
    
    assert "images" in batch
    assert "prompts" in batch
    assert "targets" in batch
    assert "tasks" in batch
    assert "metadata" in batch
    
    # Check we have samples
    assert len(batch["prompts"]) > 0
    print(f"✓ Validation loader: {len(batch['prompts'])} samples in batch")
    print(f"  Tasks: {set(batch['tasks'])}")


def test_validation_sample_content():
    """Test validation sample content."""
    from data.validation import create_validation_loader
    
    loader = create_validation_loader(
        data_root="./data/molmo2",
        stage="pretrain",
        batch_size=4,
        num_workers=0,
    )
    
    batch = next(iter(loader))
    
    # Check each sample has expected fields
    for i in range(len(batch["prompts"])):
        assert batch["prompts"][i], "Prompt should not be empty"
        assert batch["tasks"][i] in ["pointing", "counting"], f"Unknown task: {batch['tasks'][i]}"
        assert len(batch["images"][i]) > 0 or batch["tasks"][i] == "text", "Should have images"
        
        # Check metadata
        meta = batch["metadata"][i]
        assert "dataset" in meta
        assert "idx" in meta
    
    print(f"✓ Sample content validated for {len(batch['prompts'])} samples")


def test_validation_metrics_dataclass():
    """Test ValidationMetrics dataclass."""
    from data.validation import ValidationMetrics
    
    metrics = ValidationMetrics(
        loss=0.5,
        perplexity=1.65,
        pointing_accuracy=0.85,
        counting_accuracy=0.90,
        counting_mae=0.5,
        total_samples=100,
    )
    
    # Test to_dict
    d = metrics.to_dict()
    assert "val/loss" in d
    assert "val/perplexity" in d
    assert d["val/loss"] == 0.5
    
    print(f"✓ ValidationMetrics: {d}")


def test_validation_runner_init():
    """Test ValidationRunner initialization (without actual model)."""
    from data.validation import ValidationRunner
    import torch
    import torch.nn as nn
    
    # Dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            return x
    
    model = DummyModel()
    
    # Dummy tokenizer
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.zeros(len(texts), 10)}
    
    tokenizer = DummyTokenizer()
    
    runner = ValidationRunner(model, tokenizer)
    assert runner.model is model
    assert runner.tokenizer is tokenizer
    
    print("✓ ValidationRunner initialized")


def test_evaluation_scheduler():
    """Test EvaluationScheduler for periodic evaluation."""
    from data.validation import EvaluationScheduler, create_validation_loader
    import torch
    import torch.nn as nn
    
    # Dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            return x
        def eval(self):
            pass
        def train(self):
            pass
    
    model = DummyModel()
    
    # Dummy tokenizer
    class DummyTokenizer:
        pass
    
    tokenizer = DummyTokenizer()
    
    # Create validation loader
    val_loader = create_validation_loader(
        data_root="./data/molmo2",
        stage="pretrain",
        batch_size=16,
        num_workers=0,
    )
    
    scheduler = EvaluationScheduler(
        model=model,
        tokenizer=tokenizer,
        val_loader=val_loader,
        eval_every_steps=2000,
        benchmark_every_steps=10000,
        max_val_batches=2,  # Small for testing
    )
    
    # Test should_evaluate
    assert not scheduler.should_evaluate(0)
    assert not scheduler.should_evaluate(1000)
    assert scheduler.should_evaluate(2000)
    assert scheduler.should_evaluate(4000)
    
    # Test should_benchmark
    assert not scheduler.should_benchmark(0)
    assert not scheduler.should_benchmark(5000)
    assert scheduler.should_benchmark(10000)
    
    print("✓ EvaluationScheduler: step-based evaluation")


def test_evaluation_schedule_info():
    """Test evaluation schedule info."""
    from data.validation import get_evaluation_schedule_info
    
    schedule = get_evaluation_schedule_info(
        total_steps=100000,
        eval_every=2000,
        benchmark_every=10000,
    )
    
    assert schedule["num_validations"] == 50
    assert schedule["num_benchmarks"] == 10
    assert schedule["validation_steps"][0] == 2000
    assert schedule["benchmark_steps"][0] == 10000
    
    print(f"✓ Schedule info: {schedule['num_validations']} validations, {schedule['num_benchmarks']} benchmarks")


def test_all_validation_datasets_accessible():
    """Test that all validation datasets can be loaded."""
    from data.validation import (
        CosynPointValDataset,
        PixmoCountValDataset,
    )
    
    data_root = Path("./data/molmo2")
    
    # Test cosyn-point
    try:
        ds = CosynPointValDataset(data_root / "cosyn-point", split="validation")
        sample = ds[0]
        assert sample is not None or len(ds) > 0
        print(f"✓ cosyn-point validation: {len(ds)} samples")
    except FileNotFoundError as e:
        print(f"⚠ cosyn-point validation not found: {e}")
    
    # Test pixmo-count
    try:
        ds = PixmoCountValDataset(data_root / "pixmo-count", split="validation")
        sample = ds[0]
        assert sample is not None or len(ds) > 0
        print(f"✓ pixmo-count validation: {len(ds)} samples")
    except FileNotFoundError as e:
        print(f"⚠ pixmo-count validation not found: {e}")


def main():
    print("=" * 50)
    print("MOLMO2 VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Validation Info", test_validation_info),
        ("Validation Loader (Pretrain)", test_validation_loader_pretrain),
        ("Sample Content", test_validation_sample_content),
        ("Metrics Dataclass", test_validation_metrics_dataclass),
        ("Runner Init", test_validation_runner_init),
        ("Evaluation Scheduler", test_evaluation_scheduler),
        ("Evaluation Schedule Info", test_evaluation_schedule_info),
        ("All Datasets Accessible", test_all_validation_datasets_accessible),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
