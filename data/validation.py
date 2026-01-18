"""
Molmo2 Validation Module - Pure PyTorch

Provides validation dataloaders and metrics for all training stages.

Available validation splits (Stage 1 Pre-training):
- cosyn-point: 1,000 samples
- pixmo-count: 540 samples (val) + 540 samples (test)

Usage:
    from data.validation import create_validation_loader, ValidationRunner
    
    val_loader = create_validation_loader(stage="pretrain")
    runner = ValidationRunner(model, tokenizer)
    metrics = runner.evaluate(val_loader)
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import pyarrow.parquet as pq


@dataclass
class ValidationSample:
    """Single validation sample with ground truth."""
    images: List[Image.Image] = field(default_factory=list)
    prompt: str = ""
    target: str = ""  # Ground truth answer
    task: str = "caption"
    metadata: Dict[str, Any] = field(default_factory=dict)


def read_parquet(path: str) -> List[Dict]:
    """Read parquet file to list of dicts."""
    return pq.read_table(path).to_pylist()


# =============================================================================
# Validation Datasets
# =============================================================================

class CosynPointValDataset(Dataset):
    """cosyn-point validation set: Synthetic pointing data."""
    
    def __init__(self, data_dir: Path, split: str = "validation"):
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Validation split not found: {path}")
        self.data = read_parquet(str(path))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Optional[ValidationSample]:
        row = self.data[idx]
        img_data = row.get("image")
        if not img_data:
            return None
        
        try:
            if isinstance(img_data, dict):
                img = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
            else:
                img = Image.open(BytesIO(img_data)).convert("RGB")
        except Exception:
            return None
        
        questions = row.get("questions", [])
        answer_points = row.get("answer_points", [])
        
        prompt = questions[0] if questions else ""
        target = answer_points[0] if answer_points else ""
        
        return ValidationSample(
            images=[img],
            prompt=f"Q: {prompt}",
            target=target,
            task="pointing",
            metadata={"idx": idx, "dataset": "cosyn-point"},
        )


class PixmoCountValDataset(Dataset):
    """pixmo-count validation/test set: Object counting."""
    
    def __init__(self, data_dir: Path, split: str = "validation", cache_dir: Path = None):
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Validation split not found: {path}")
        self.data = read_parquet(str(path))
        self.cache = cache_dir or data_dir / "cache"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image with caching."""
        import ssl
        from urllib.request import urlopen
        
        self.cache.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
        
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                pass
        
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(url, timeout=10, context=ctx) as r:
                img = Image.open(BytesIO(r.read())).convert("RGB")
                img.save(cache_path, "JPEG")
                return img
        except Exception:
            return None
    
    def __getitem__(self, idx: int) -> Optional[ValidationSample]:
        row = self.data[idx]
        img = self._download_image(row.get("image_url", ""))
        if not img:
            return None
        
        label = row.get("label", "objects")
        count = row.get("count", 0)
        
        return ValidationSample(
            images=[img],
            prompt=f"Q: How many {label}?",
            target=str(count),
            task="counting",
            metadata={"idx": idx, "dataset": "pixmo-count", "count": count},
        )


# =============================================================================
# Validation Registry
# =============================================================================

# Stage 1: Pre-training validation datasets
PRETRAIN_VAL_DATASETS = {
    "cosyn-point": {
        "class": CosynPointValDataset,
        "split": "validation",
        "samples": 1000,
        "task": "pointing",
    },
    "pixmo-count": {
        "class": PixmoCountValDataset,
        "split": "validation",
        "samples": 540,
        "task": "counting",
    },
}

# Stage 2/3: SFT validation datasets (to be expanded)
SFT_VAL_DATASETS = {
    # Same as pretrain for now, but can include more datasets
    "cosyn-point": PRETRAIN_VAL_DATASETS["cosyn-point"],
    "pixmo-count": PRETRAIN_VAL_DATASETS["pixmo-count"],
    # Future: add Molmo2-SynMultiImageQA val, Molmo2-Cap val, etc.
}

# Evaluation-only benchmarks (post-training)
EVAL_BENCHMARKS = {
    # These need to be downloaded separately
    "Molmo2-VideoPointEval": {"samples": 181, "task": "video_pointing"},
    "Molmo2-VideoCountEval": {"samples": 533, "task": "video_counting"},
    "Molmo2-VideoTrackEval": {"samples": 1000, "task": "video_tracking"},
}


def collate_validation(batch: List[Optional[ValidationSample]]) -> Dict[str, Any]:
    """Collate validation samples, filter None."""
    batch = [s for s in batch if s is not None]
    if not batch:
        return {
            "images": [],
            "prompts": [],
            "targets": [],
            "tasks": [],
            "metadata": [],
        }
    return {
        "images": [s.images for s in batch],
        "prompts": [s.prompt for s in batch],
        "targets": [s.target for s in batch],
        "tasks": [s.task for s in batch],
        "metadata": [s.metadata for s in batch],
    }


def create_validation_loader(
    data_root: str = "./data/molmo2",
    stage: str = "pretrain",
    batch_size: int = 32,
    num_workers: int = 4,
    datasets: List[str] = None,
) -> DataLoader:
    """
    Create validation dataloader for specified stage.
    
    Args:
        data_root: Root directory containing datasets
        stage: Training stage - "pretrain" or "sft"
        batch_size: Batch size for validation
        num_workers: Number of data loading workers
        datasets: Optional list of specific datasets to include
    
    Returns:
        DataLoader for validation
    """
    data_root = Path(data_root)
    
    # Select validation registry based on stage
    if stage == "pretrain":
        registry = PRETRAIN_VAL_DATASETS
    elif stage in ("sft", "longcontext"):
        registry = SFT_VAL_DATASETS
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'pretrain' or 'sft'.")
    
    # Filter by requested datasets
    if datasets:
        registry = {k: v for k, v in registry.items() if k in datasets}
    
    # Load validation datasets
    all_datasets = []
    for name, config in registry.items():
        path = data_root / name
        if not path.exists():
            print(f"Warning: {name} not found at {path}, skipping")
            continue
        
        try:
            ds = config["class"](path, split=config["split"])
            all_datasets.append(ds)
            print(f"Loaded {name} validation: {len(ds)} samples")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    if not all_datasets:
        raise RuntimeError(f"No validation datasets found in {data_root}")
    
    combined = ConcatDataset(all_datasets)
    
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        collate_fn=collate_validation,
        num_workers=num_workers,
        pin_memory=True,
    )


# =============================================================================
# Validation Metrics
# =============================================================================

@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    loss: float = 0.0
    perplexity: float = 0.0
    
    # Task-specific metrics
    pointing_accuracy: float = 0.0
    counting_accuracy: float = 0.0
    counting_mae: float = 0.0  # Mean Absolute Error
    
    # Counts
    total_samples: int = 0
    pointing_samples: int = 0
    counting_samples: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "val/loss": self.loss,
            "val/perplexity": self.perplexity,
            "val/pointing_accuracy": self.pointing_accuracy,
            "val/counting_accuracy": self.counting_accuracy,
            "val/counting_mae": self.counting_mae,
            "val/total_samples": self.total_samples,
        }


class ValidationRunner:
    """
    Run validation and compute metrics.
    
    Usage:
        runner = ValidationRunner(model, tokenizer, device)
        metrics = runner.evaluate(val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,  # Tokenizer with encode/decode
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: int = None,
    ) -> ValidationMetrics:
        """
        Run validation and compute metrics.
        
        Args:
            dataloader: Validation dataloader
            max_batches: Optional limit on number of batches to evaluate
        
        Returns:
            ValidationMetrics with computed metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        pointing_correct = 0
        pointing_total = 0
        
        counting_correct = 0
        counting_total = 0
        counting_mae_sum = 0.0
        
        num_batches = 0
        
        for batch in dataloader:
            if max_batches and num_batches >= max_batches:
                break
            
            if not batch["prompts"]:
                continue
            
            # Compute loss
            loss, num_tokens = self._compute_batch_loss(batch)
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
            # Compute task-specific metrics
            for i, task in enumerate(batch["tasks"]):
                if task == "pointing":
                    # For pointing, check if predicted points are close to target
                    # This is a simplified metric - real implementation would parse points
                    pointing_total += 1
                    # TODO: Implement proper pointing accuracy
                
                elif task == "counting":
                    target_count = batch["metadata"][i].get("count", 0)
                    # TODO: Generate prediction and compute accuracy
                    counting_total += 1
            
            num_batches += 1
        
        # Compute final metrics
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = ValidationMetrics(
            loss=avg_loss,
            perplexity=min(perplexity, 1e6),  # Clamp to avoid overflow
            pointing_accuracy=pointing_correct / max(pointing_total, 1),
            counting_accuracy=counting_correct / max(counting_total, 1),
            counting_mae=counting_mae_sum / max(counting_total, 1),
            total_samples=pointing_total + counting_total,
            pointing_samples=pointing_total,
            counting_samples=counting_total,
        )
        
        self.model.train()
        return metrics
    
    def _compute_batch_loss(self, batch: Dict[str, Any]) -> Tuple[float, int]:
        """
        Compute loss for a batch.
        
        Returns:
            (loss_value, num_tokens)
        """
        # This is a placeholder - actual implementation depends on model architecture
        # The model forward pass would look like:
        # 
        # inputs = self._prepare_inputs(batch)
        # outputs = self.model(**inputs)
        # loss = outputs.loss
        # return loss.item(), inputs["labels"].ne(-100).sum().item()
        
        # For now, return dummy values
        return 0.0, len(batch["prompts"])
    
    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from batch."""
        # Tokenize prompts and targets
        prompts = batch["prompts"]
        targets = batch["targets"]
        
        # Combine prompt + target for teacher forcing
        full_texts = [f"{p}\nA: {t}" for p, t in zip(prompts, targets)]
        
        # Tokenize
        # encodings = self.tokenizer(
        #     full_texts,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        
        # return {
        #     "input_ids": encodings["input_ids"].to(self.device),
        #     "attention_mask": encodings["attention_mask"].to(self.device),
        #     "images": batch["images"],
        #     "labels": encodings["input_ids"].to(self.device),  # For loss computation
        # }
        
        return {}


# =============================================================================
# Periodic Evaluation Manager
# =============================================================================

class EvaluationScheduler:
    """
    Manages periodic evaluation during training.
    
    Supports both in-training validation and benchmark evaluation.
    Optionally logs to wandb.
    
    Usage:
        scheduler = EvaluationScheduler(
            model=model,
            tokenizer=tokenizer,
            val_loader=val_loader,
            eval_every_steps=2000,
            benchmark_every_steps=10000,
            logger=wandb_logger,  # Optional
        )
        
        for step, batch in enumerate(train_loader):
            # ... training step ...
            
            # Check if we should evaluate
            metrics = scheduler.step(step)
            if metrics:
                print(f"Step {step}: {metrics}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        val_loader: DataLoader = None,
        eval_every_steps: int = 2000,
        benchmark_every_steps: int = 10000,
        benchmark_loaders: Dict[str, DataLoader] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        max_val_batches: int = 50,  # Limit validation batches for speed
        logger: Any = None,  # WandbLogger instance
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_loader = val_loader
        self.eval_every_steps = eval_every_steps
        self.benchmark_every_steps = benchmark_every_steps
        self.benchmark_loaders = benchmark_loaders or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_val_batches = max_val_batches
        self.logger = logger
        
        self.runner = ValidationRunner(model, tokenizer, device, dtype)
        
        # Track best metrics for checkpointing
        self.best_loss = float("inf")
        self.best_step = 0
        self.history: List[Dict[str, Any]] = []
    
    def should_evaluate(self, step: int) -> bool:
        """Check if we should run validation at this step."""
        return step > 0 and step % self.eval_every_steps == 0
    
    def should_benchmark(self, step: int) -> bool:
        """Check if we should run benchmarks at this step."""
        return step > 0 and step % self.benchmark_every_steps == 0
    
    def step(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Check and run evaluation if needed.
        
        Args:
            step: Current training step
        
        Returns:
            Dictionary with metrics if evaluation was run, None otherwise
        """
        results = {}
        
        # Run validation
        if self.should_evaluate(step) and self.val_loader is not None:
            val_metrics = self.runner.evaluate(
                self.val_loader,
                max_batches=self.max_val_batches,
            )
            results["validation"] = val_metrics.to_dict()
            results["step"] = step
            
            # Track best
            if val_metrics.loss < self.best_loss:
                self.best_loss = val_metrics.loss
                self.best_step = step
                results["is_best"] = True
            else:
                results["is_best"] = False
            
            # Log to wandb
            if self.logger is not None:
                self.logger.log_validation(step, val_metrics.to_dict())
        
        # Run benchmarks (less frequently)
        if self.should_benchmark(step) and self.benchmark_loaders:
            benchmark_results = {}
            for name, loader in self.benchmark_loaders.items():
                metrics = self.runner.evaluate(loader, max_batches=None)
                benchmark_results[name] = metrics.to_dict()
                
                # Log benchmark to wandb
                if self.logger is not None:
                    self.logger.log_benchmark(step, name, metrics.to_dict())
            
            results["benchmarks"] = benchmark_results
        
        if results:
            self.history.append(results)
            return results
        
        return None
    
    def get_best(self) -> Tuple[float, int]:
        """Get best validation loss and step."""
        return self.best_loss, self.best_step
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.history


def create_evaluation_scheduler(
    model: nn.Module,
    tokenizer: Any,
    data_root: str = "./data/molmo2",
    stage: str = "pretrain",
    eval_every_steps: int = 2000,
    benchmark_every_steps: int = 10000,
    val_batch_size: int = 32,
    device: torch.device = None,
    logger: Any = None,  # WandbLogger instance
) -> EvaluationScheduler:
    """
    Create an evaluation scheduler with validation and benchmark loaders.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        data_root: Root directory containing datasets
        stage: Training stage ("pretrain" or "sft")
        eval_every_steps: Run validation every N steps
        benchmark_every_steps: Run benchmarks every N steps
        val_batch_size: Batch size for validation
        device: Device to use
        logger: WandbLogger instance for logging metrics
    
    Returns:
        EvaluationScheduler instance
    """
    # Create validation loader
    val_loader = create_validation_loader(
        data_root=data_root,
        stage=stage,
        batch_size=val_batch_size,
        num_workers=4,
    )
    
    # TODO: Create benchmark loaders when benchmark datasets are downloaded
    benchmark_loaders = {}
    
    return EvaluationScheduler(
        model=model,
        tokenizer=tokenizer,
        val_loader=val_loader,
        eval_every_steps=eval_every_steps,
        benchmark_every_steps=benchmark_every_steps,
        benchmark_loaders=benchmark_loaders,
        device=device,
        logger=logger,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_validation_info(stage: str = "pretrain") -> Dict[str, Any]:
    """Get information about available validation datasets."""
    if stage == "pretrain":
        registry = PRETRAIN_VAL_DATASETS
    else:
        registry = SFT_VAL_DATASETS
    
    info = {
        "stage": stage,
        "datasets": {},
        "total_samples": 0,
    }
    
    for name, config in registry.items():
        info["datasets"][name] = {
            "samples": config["samples"],
            "task": config["task"],
            "split": config["split"],
        }
        info["total_samples"] += config["samples"]
    
    return info


def get_evaluation_schedule_info(
    total_steps: int,
    eval_every: int = 2000,
    benchmark_every: int = 10000,
) -> Dict[str, Any]:
    """Get information about evaluation schedule."""
    num_evals = total_steps // eval_every
    num_benchmarks = total_steps // benchmark_every
    
    return {
        "total_steps": total_steps,
        "eval_every_steps": eval_every,
        "benchmark_every_steps": benchmark_every,
        "num_validations": num_evals,
        "num_benchmarks": num_benchmarks,
        "validation_steps": list(range(eval_every, total_steps + 1, eval_every)),
        "benchmark_steps": list(range(benchmark_every, total_steps + 1, benchmark_every)),
    }


if __name__ == "__main__":
    print("=" * 50)
    print("MOLMO2 VALIDATION DATASETS")
    print("=" * 50)
    
    for stage in ["pretrain", "sft"]:
        info = get_validation_info(stage)
        print(f"\n{stage.upper()} Stage:")
        print(f"  Total samples: {info['total_samples']}")
        for name, ds_info in info["datasets"].items():
            print(f"  - {name}: {ds_info['samples']} ({ds_info['task']})")
    
    print("\n" + "=" * 50)
    print("EVALUATION BENCHMARKS (post-training)")
    print("=" * 50)
    for name, bench_info in EVAL_BENCHMARKS.items():
        print(f"  - {name}: {bench_info['samples']} ({bench_info['task']})")
    
    print("\n" + "=" * 50)
    print("EVALUATION SCHEDULE (Stage 1: ~100K steps)")
    print("=" * 50)
    schedule = get_evaluation_schedule_info(
        total_steps=100000,
        eval_every=2000,
        benchmark_every=10000,
    )
    print(f"  Validation every: {schedule['eval_every_steps']} steps")
    print(f"  Benchmarks every: {schedule['benchmark_every_steps']} steps")
    print(f"  Total validations: {schedule['num_validations']}")
    print(f"  Total benchmarks: {schedule['num_benchmarks']}")
    print(f"  First 5 val steps: {schedule['validation_steps'][:5]}")
    print(f"  First 5 bench steps: {schedule['benchmark_steps'][:5]}")
