"""
NanoMolmo2 Training Utilities

- FSDP2 wrapper for distributed training
- torch.compile utilities for static shapes
- AMP (Automatic Mixed Precision) with bfloat16
- Training utilities (optimizer, scheduler, gradient handling)
- Weights & Biases logging
"""

from .trainer import Trainer, TrainingConfig
from .distributed import (
    setup_fsdp,
    compile_model,
    setup_amp,
    get_gradient_scaler,
)
from .wandb_logging import (
    WandbLogger,
    TrainConfig,
    MetricsTracker,
    create_logger,
)

__all__ = [
    # Training
    "Trainer",
    "TrainingConfig",
    # Distributed
    "setup_fsdp",
    "compile_model", 
    "setup_amp",
    "get_gradient_scaler",
    # Logging
    "WandbLogger",
    "TrainConfig",
    "MetricsTracker",
    "create_logger",
]
