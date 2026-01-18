"""
Molmo2 Training Logger - Weights & Biases Integration

Provides unified logging for training metrics, validation results, and benchmarks.

Usage:
    from training.logging import WandbLogger
    
    logger = WandbLogger(
        project="nanomolmo2",
        name="stage1-pretrain",
        config=train_config,
    )
    
    # Log training metrics
    logger.log_train(step=100, loss=2.5, lr=1e-4)
    
    # Log validation metrics
    logger.log_validation(step=2000, metrics=val_metrics)
    
    # Finish
    logger.finish()
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class TrainConfig:
    """Training configuration for logging."""
    # Model
    model_name: str = "nanomolmo2"
    vision_encoder: str = "siglip2-so400m-patch14-384"
    llm: str = "qwen3-0.6b-base"
    
    # Training
    stage: str = "pretrain"
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100000
    
    # Data
    seq_length: int = 4096
    max_frames: int = 128
    
    # Evaluation
    eval_every_steps: int = 2000
    benchmark_every_steps: int = 10000
    
    # Hardware
    num_gpus: int = 1
    mixed_precision: str = "bf16"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WandbLogger:
    """
    Weights & Biases logger for training.
    
    Falls back to console logging if wandb is not available.
    """
    
    def __init__(
        self,
        project: str = "nanomolmo2",
        name: str = None,
        config: TrainConfig = None,
        tags: List[str] = None,
        notes: str = None,
        save_dir: str = "./logs",
        enabled: bool = True,
        resume: bool = False,
        run_id: str = None,
    ):
        self.project = project
        self.name = name
        self.config = config or TrainConfig()
        self.tags = tags or []
        self.notes = notes
        self.save_dir = Path(save_dir)
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if self.enabled:
            self._init_wandb(resume, run_id)
        else:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not installed. Logging to console only.")
                print("Install with: uv pip install wandb")
            self._init_local()
    
    def _init_wandb(self, resume: bool, run_id: str):
        """Initialize wandb run."""
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config.to_dict(),
            tags=self.tags,
            notes=self.notes,
            dir=str(self.save_dir),
            resume="allow" if resume else None,
            id=run_id,
        )
        print(f"Wandb initialized: {self.run.url}")
    
    def _init_local(self):
        """Initialize local logging."""
        self.log_file = self.save_dir / f"{self.name or 'train'}.jsonl"
        print(f"Local logging to: {self.log_file}")
    
    def log(self, metrics: Dict[str, Any], step: int = None):
        """Log arbitrary metrics."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)
        else:
            self._log_local(metrics, step)
    
    def _log_local(self, metrics: Dict[str, Any], step: int = None):
        """Log to local file."""
        entry = {"step": step, **metrics}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_train(
        self,
        step: int,
        loss: float,
        lr: float = None,
        grad_norm: float = None,
        tokens_per_sec: float = None,
        **kwargs,
    ):
        """Log training metrics."""
        metrics = {
            "train/loss": loss,
            "train/step": step,
        }
        if lr is not None:
            metrics["train/lr"] = lr
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        if tokens_per_sec is not None:
            metrics["train/tokens_per_sec"] = tokens_per_sec
        metrics.update({f"train/{k}": v for k, v in kwargs.items()})
        
        self.log(metrics, step=step)
    
    def log_validation(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = "val",
    ):
        """Log validation metrics."""
        # Rename keys if needed
        log_metrics = {}
        for k, v in metrics.items():
            if k.startswith("val/"):
                log_metrics[k] = v
            else:
                log_metrics[f"{prefix}/{k}"] = v
        
        self.log(log_metrics, step=step)
    
    def log_benchmark(
        self,
        step: int,
        benchmark_name: str,
        metrics: Dict[str, float],
    ):
        """Log benchmark results."""
        log_metrics = {
            f"benchmark/{benchmark_name}/{k}": v
            for k, v in metrics.items()
        }
        self.log(log_metrics, step=step)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = None,
        **kwargs,
    ):
        """Log epoch-level metrics."""
        metrics = {
            "epoch/train_loss": train_loss,
            "epoch": epoch,
        }
        if val_loss is not None:
            metrics["epoch/val_loss"] = val_loss
        metrics.update({f"epoch/{k}": v for k, v in kwargs.items()})
        
        self.log(metrics)
    
    def log_model_info(
        self,
        total_params: int,
        trainable_params: int,
        model_size_mb: float = None,
    ):
        """Log model information."""
        info = {
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        }
        if model_size_mb:
            info["model/size_mb"] = model_size_mb
        
        if self.enabled and self.run:
            wandb.config.update(info)
        self.log(info, step=0)
    
    def log_image(
        self,
        key: str,
        image,  # PIL.Image or numpy array
        caption: str = None,
        step: int = None,
    ):
        """Log an image."""
        if self.enabled and self.run:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
    
    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: int = None,
    ):
        """Log a table."""
        if self.enabled and self.run:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({key: table}, step=step)
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters."""
        if self.enabled and self.run:
            wandb.watch(model, log="all", log_freq=log_freq)
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        metadata: Dict[str, Any] = None,
    ):
        """Save and log a checkpoint."""
        if self.enabled and self.run:
            artifact = wandb.Artifact(
                name=f"{self.name or 'model'}-checkpoint",
                type="model",
                metadata=metadata,
            )
            artifact.add_file(checkpoint_path)
            self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish the run."""
        if self.enabled and self.run:
            wandb.finish()
            print("Wandb run finished.")


class MetricsTracker:
    """
    Track and aggregate metrics over steps.
    
    Usage:
        tracker = MetricsTracker()
        
        for step, batch in enumerate(loader):
            loss = train_step(batch)
            tracker.update(loss=loss.item())
            
            if step % log_every == 0:
                avg_metrics = tracker.get_and_reset()
                logger.log_train(step, **avg_metrics)
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, **kwargs):
        """Add metric values."""
        for k, v in kwargs.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
    
    def get(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in self.metrics.items()
        }
    
    def get_and_reset(self) -> Dict[str, float]:
        """Get averages and reset."""
        result = self.get()
        self.reset()
        return result
    
    def reset(self):
        """Clear all metrics."""
        self.metrics = {}


def create_logger(
    project: str = "nanomolmo2",
    name: str = None,
    stage: str = "pretrain",
    config: Dict[str, Any] = None,
    use_wandb: bool = True,
) -> WandbLogger:
    """
    Create a logger with sensible defaults.
    
    Args:
        project: Wandb project name
        name: Run name (auto-generated if None)
        stage: Training stage for tagging
        config: Training configuration dict
        use_wandb: Whether to use wandb (falls back to local if unavailable)
    
    Returns:
        WandbLogger instance
    """
    import datetime
    
    if name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stage}_{timestamp}"
    
    train_config = TrainConfig(stage=stage)
    if config:
        for k, v in config.items():
            if hasattr(train_config, k):
                setattr(train_config, k, v)
    
    return WandbLogger(
        project=project,
        name=name,
        config=train_config,
        tags=[stage, "nanomolmo2"],
        enabled=use_wandb,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("WANDB LOGGER TEST")
    print("=" * 50)
    
    # Test without wandb (local logging)
    logger = WandbLogger(
        project="nanomolmo2-test",
        name="test-run",
        config=TrainConfig(stage="pretrain"),
        enabled=False,  # Force local logging for test
    )
    
    # Log some metrics
    logger.log_train(step=100, loss=2.5, lr=1e-4, grad_norm=1.2)
    logger.log_train(step=200, loss=2.3, lr=1e-4, grad_norm=1.1)
    
    # Log validation
    logger.log_validation(step=2000, metrics={
        "loss": 2.1,
        "perplexity": 8.2,
        "pointing_accuracy": 0.75,
    })
    
    print(f"\n✓ Logged to: {logger.log_file}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    for i in range(10):
        tracker.update(loss=2.5 - i * 0.1, grad_norm=1.0)
    
    avg = tracker.get_and_reset()
    print(f"✓ Metrics tracker: avg_loss={avg['loss']:.2f}")
    
    print("\n" + "=" * 50)
    print("WANDB STATUS")
    print("=" * 50)
    print(f"  wandb available: {WANDB_AVAILABLE}")
    if not WANDB_AVAILABLE:
        print("  Install with: uv pip install wandb")
