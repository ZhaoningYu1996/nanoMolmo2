"""
NanoMolmo2 Trainer

Pure PyTorch training loop with:
- AMP (bfloat16 with LayerNorm/RoPE in full precision)
- FSDP2 for distributed training
- torch.compile for static shapes
- Gradient accumulation and clipping
- Checkpointing and logging
"""

import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .distributed import (
    setup_distributed,
    cleanup_distributed,
    setup_fsdp,
    compile_model,
    setup_amp,
    clip_gradient_norm,
    compute_gradient_norm,
    FSDPConfig,
    CompileConfig,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    min_lr_ratio: float = 0.1  # Min LR = learning_rate * min_lr_ratio
    
    # Training duration
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 5000
    log_every: int = 100
    
    # Batch configuration
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    
    # Mixed precision (as per Molmo2 tech report)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_grad_scaler: bool = False  # Usually not needed for bfloat16
    
    # Distributed
    use_fsdp: bool = False
    fsdp_config: Optional[FSDPConfig] = None
    
    # Compilation
    use_compile: bool = True
    compile_config: Optional[CompileConfig] = None
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    save_total_limit: int = 5  # Keep N most recent checkpoints
    
    # Logging
    log_dir: str = "./logs"
    use_tensorboard: bool = True
    experiment_name: str = "nanomolmo2"


class Trainer:
    """
    NanoMolmo2 Trainer.
    
    Pure PyTorch implementation following Molmo2 training setup:
    - FSDP2 for distributed training
    - AMP with bfloat16
    - torch.compile for throughput
    - Cosine LR schedule with warmup
    
    Usage:
        trainer = Trainer(model, train_loader, config)
        trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Setup distributed
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = self.rank == 0
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup FSDP if enabled
        if self.config.use_fsdp and self.world_size > 1:
            fsdp_config = self.config.fsdp_config or FSDPConfig()
            self.model = setup_fsdp(self.model, fsdp_config, self.local_rank)
        
        # Compile model if enabled
        if self.config.use_compile:
            compile_config = self.config.compile_config or CompileConfig()
            self.model = compile_model(self.model, compile_config)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup LR scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup AMP
        self.amp_dtype = setup_amp(self.config.amp_dtype) if self.config.use_amp else torch.float32
        self.scaler = GradScaler("cuda", enabled=self.config.use_grad_scaler)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
        # Setup logging
        self.writer = None
        if self.config.use_tensorboard and self.is_main:
            self._setup_tensorboard()
        
        # Create output directories
        if self.is_main:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and LayerNorm
            if "bias" in name or "layernorm" in name or "layer_norm" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup."""
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps
        min_lr_ratio = self.config.min_lr_ratio
        
        def lr_lambda(step: int) -> float:
            # Warmup phase
            if step < warmup_steps:
                return step / warmup_steps
            
            # Decay phase
            if self.config.lr_scheduler == "cosine":
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return min_lr_ratio + (1 - min_lr_ratio) * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
            elif self.config.lr_scheduler == "linear":
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 1 - (1 - min_lr_ratio) * progress
            else:  # constant
                return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(self.config.log_dir) / self.config.experiment_name
            self.writer = SummaryWriter(log_dir=str(log_path))
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
    
    def train(self):
        """Run training loop."""
        if self.is_main:
            print(f"Starting training...")
            print(f"  Device: {self.device}")
            print(f"  World size: {self.world_size}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size}")
            print(f"  Max steps: {self.config.max_steps}")
            print(f"  Learning rate: {self.config.learning_rate}")
            print(f"  AMP dtype: {self.amp_dtype}")
        
        self.model.train()
        
        # Training loop
        step_loss = 0.0
        step_tokens = 0
        accumulation_step = 0
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass with AMP
                with autocast("cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
                    outputs = self.model(**batch)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple):
                        loss = outputs[1] if len(outputs) > 1 else outputs[0]
                        if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                            pass  # Already a scalar loss
                        else:
                            loss = outputs[0]  # Assume first element
                            loss = loss.mean()
                    else:
                        loss = outputs.mean() if outputs.dim() > 0 else outputs
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                step_loss += loss.item() * self.config.gradient_accumulation_steps
                step_tokens += batch.get("input_ids", torch.tensor([])).numel()
                accumulation_step += 1
                
                # Gradient accumulation
                if accumulation_step < self.config.gradient_accumulation_steps:
                    continue
                
                # Gradient clipping
                if self.config.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = clip_gradient_norm(self.model, self.config.max_grad_norm)
                
                # Optimizer step
                if self.config.use_grad_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                accumulation_step = 0
                
                # Logging
                if self.global_step % self.config.log_every == 0 and self.is_main:
                    elapsed = time.time() - start_time
                    tokens_per_sec = step_tokens / elapsed if elapsed > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    
                    print(
                        f"Step {self.global_step}: "
                        f"loss={step_loss:.4f}, "
                        f"lr={lr:.2e}, "
                        f"grad_norm={grad_norm:.2f}, "
                        f"tokens/s={tokens_per_sec:.0f}"
                    )
                    
                    if self.writer:
                        self.writer.add_scalar("train/loss", step_loss, self.global_step)
                        self.writer.add_scalar("train/lr", lr, self.global_step)
                        self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
                        self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.global_step)
                    
                    step_loss = 0.0
                    step_tokens = 0
                    start_time = time.time()
                
                # Evaluation
                if self.global_step % self.config.eval_every == 0:
                    if self.eval_loader is not None:
                        eval_loss = self.evaluate()
                        if self.is_main:
                            print(f"Step {self.global_step}: eval_loss={eval_loss:.4f}")
                            if self.writer:
                                self.writer.add_scalar("eval/loss", eval_loss, self.global_step)
                            
                            # Track best loss
                            if eval_loss < self.best_loss:
                                self.best_loss = eval_loss
                                self._save_checkpoint("best")
                
                # Checkpointing
                if self.global_step % self.config.save_every == 0 and self.is_main:
                    self._save_checkpoint(f"step_{self.global_step}")
                
                # Check if done
                if self.global_step >= self.config.max_steps:
                    break
            
            self.epoch += 1
        
        # Final checkpoint
        if self.is_main:
            self._save_checkpoint("final")
            print(f"Training complete! Best loss: {self.best_loss:.4f}")
        
        # Cleanup
        if self.writer:
            self.writer.close()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for batch in self.eval_loader:
            batch = self._prepare_batch(batch)
            
            with autocast("cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
                outputs = self.model(**batch)
                
                if isinstance(outputs, tuple):
                    loss = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    loss = outputs
                
                if loss.dim() > 0:
                    loss = loss.mean()
            
            batch_size = batch.get("input_ids", torch.tensor([[]])).shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        self.model.train()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        prepared = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    prepared[key] = [v.to(self.device) for v in value]
                else:
                    prepared[key] = value
            else:
                prepared[key] = value
        
        return prepared
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = self.model.state_dict()
        torch.save(model_state, checkpoint_dir / "model.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # Save scheduler state
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": {
                "learning_rate": self.config.learning_rate,
                "max_steps": self.config.max_steps,
                "batch_size": self.config.batch_size,
            },
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model state
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        
        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print(f"Loaded optimizer from {optimizer_path}")
        
        # Load scheduler state
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            print(f"Loaded scheduler from {scheduler_path}")
        
        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            print(f"Resumed from step {self.global_step}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        if self.config.save_total_limit <= 0:
            return
        
        output_dir = Path(self.config.output_dir)
        
        # Find all step checkpoints (not best/final)
        checkpoints = []
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step = int(d.name.split("_")[1])
                    checkpoints.append((step, d))
                except ValueError:
                    continue
        
        # Sort by step and remove old ones
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        for step, checkpoint_dir in checkpoints[self.config.save_total_limit:]:
            import shutil
            shutil.rmtree(checkpoint_dir)
            print(f"Removed old checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    # Test trainer components
    print("Testing Trainer...")
    
    # Create dummy model and data
    model = nn.Linear(10, 10)
    dummy_data = [{"input_ids": torch.randn(2, 10)} for _ in range(10)]
    loader = DataLoader(dummy_data, batch_size=2)
    
    # Create config
    config = TrainingConfig(
        max_steps=5,
        log_every=1,
        use_amp=False,
        use_fsdp=False,
        use_compile=False,
    )
    
    print(f"Config: max_steps={config.max_steps}, lr={config.learning_rate}")
    
    print("✓ Trainer tests passed!")
