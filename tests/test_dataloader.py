"""
Test Molmo2 DataLoader

Run: python tests/test_dataloader.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from data import create_dataloader, Molmo2DataLoader, Sample


def test_sample():
    """Test Sample dataclass."""
    s = Sample()
    assert s.images == [] and s.text == "" and s.task == "caption"
    
    img = Image.new("RGB", (100, 100), "red")
    s2 = Sample([img], "test", "pointing", 2.0)
    assert len(s2.images) == 1 and s2.weight == 2.0
    print("✓ Sample")


def test_dataloader():
    """Test dataloader iteration."""
    loader = create_dataloader(batch_size=128, num_workers=0, max_samples=200)
    batch = next(iter(loader))
    
    assert all(k in batch for k in ["images", "texts", "tasks", "weights"])
    assert isinstance(batch["weights"], torch.Tensor)
    assert len(batch["texts"]) <= 128
    print(f"✓ DataLoader: {len(batch['texts'])} samples")


def test_batch_contents():
    """Test batch contents are valid."""
    loader = create_dataloader(batch_size=128, num_workers=0, max_samples=200)
    batch = next(iter(loader))
    
    valid_tasks = {"caption", "pointing", "counting", "text"}
    for text, task in zip(batch["texts"], batch["tasks"]):
        assert isinstance(text, str) and len(text) > 0
        assert task in valid_tasks
    
    assert (batch["weights"] > 0).all()
    print(f"✓ Batch contents: tasks={set(batch['tasks'])}")


def test_weighted_sampling():
    """Test sampling distribution."""
    loader = create_dataloader(batch_size=128, num_workers=0, max_samples=500)
    
    counts = {}
    for i, batch in enumerate(loader):
        if i >= 10: break
        for task in batch["tasks"]:
            counts[task] = counts.get(task, 0) + 1
    
    total = sum(counts.values())
    print(f"✓ Sampling: " + ", ".join(f"{k}={v/total:.0%}" for k, v in counts.items()))


def test_dataset_selection():
    """Test selecting specific datasets."""
    loader = Molmo2DataLoader(
        batch_size=128, num_workers=0, max_samples=200, datasets=["pixmo-cap"]
    )
    loader._setup()
    batch = next(iter(loader))
    
    assert all(t == "caption" for t in batch["tasks"])
    print("✓ Dataset selection")


if __name__ == "__main__":
    print("=" * 40)
    print("MOLMO2 DATALOADER TESTS")
    print("=" * 40)
    
    tests = [test_sample, test_dataloader, test_batch_contents, 
             test_weighted_sampling, test_dataset_selection]
    
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print("=" * 40)
    print(f"RESULTS: {passed} passed, {failed} failed")
