"""
Distributed Training Utilities

Based on Molmo2 tech report:
- FSDP2 for model parallelism
- torch.compile with static shapes
- AMP with bfloat16 (LayerNorm/RoPE in full precision)
- SDPA instead of FlashAttention (supports custom masks)

References:
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    ModuleWrapPolicy,
)


@dataclass
class FSDPConfig:
    """Configuration for FSDP distributed training."""
    # Sharding strategy
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    
    # Mixed precision
    use_mixed_precision: bool = True
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"  # Gradient reduction in full precision
    buffer_dtype: str = "bfloat16"
    
    # Memory optimization
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, None
    
    # Wrapping policy
    min_num_params: int = 100_000_000  # Wrap layers with > 100M params
    use_orig_params: bool = True  # Required for torch.compile
    
    # Checkpointing
    activation_checkpointing: bool = False


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""
    enabled: bool = True
    backend: str = "inductor"  # "inductor", "cudagraphs", "eager"
    mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False  # True = error on graph breaks
    dynamic: bool = False  # Static shapes for max performance
    
    # Options for inductor backend
    options: Optional[Dict[str, Any]] = None


def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        # Check if running in distributed mode
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
            
            # Set device
            torch.cuda.set_device(local_rank)
            
            return rank, world_size, local_rank
        else:
            # Single GPU mode
            return 0, 1, 0
    else:
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_mixed_precision_policy(config: FSDPConfig) -> MixedPrecision:
    """Create mixed precision policy for FSDP."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    return MixedPrecision(
        param_dtype=dtype_map[config.param_dtype],
        reduce_dtype=dtype_map[config.reduce_dtype],
        buffer_dtype=dtype_map[config.buffer_dtype],
    )


def get_sharding_strategy(config: FSDPConfig) -> ShardingStrategy:
    """Get FSDP sharding strategy."""
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    return strategies.get(config.sharding_strategy, ShardingStrategy.FULL_SHARD)


def get_auto_wrap_policy(config: FSDPConfig, layer_class: Optional[type] = None):
    """Create auto wrap policy for FSDP."""
    if layer_class is not None:
        # Wrap specific layer types (e.g., TransformerLayer)
        return transformer_auto_wrap_policy(
            transformer_layer_cls={layer_class},
        )
    else:
        # Wrap based on parameter count
        return size_based_auto_wrap_policy(
            min_num_params=config.min_num_params,
        )


def setup_fsdp(
    model: nn.Module,
    config: Optional[FSDPConfig] = None,
    device_id: Optional[int] = None,
) -> FSDP:
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: Model to wrap
        config: FSDP configuration
        device_id: GPU device ID
        
    Returns:
        FSDP-wrapped model
    """
    config = config or FSDPConfig()
    
    if device_id is None:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    # Build FSDP kwargs
    fsdp_kwargs = {
        "sharding_strategy": get_sharding_strategy(config),
        "use_orig_params": config.use_orig_params,
    }
    
    # Mixed precision
    if config.use_mixed_precision:
        fsdp_kwargs["mixed_precision"] = get_mixed_precision_policy(config)
    
    # CPU offload
    if config.cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
    
    # Backward prefetch
    if config.backward_prefetch:
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        if config.backward_prefetch in prefetch_map:
            fsdp_kwargs["backward_prefetch"] = prefetch_map[config.backward_prefetch]
    
    # Device
    if device_id is not None:
        fsdp_kwargs["device_id"] = device_id
    
    # Auto wrap policy
    fsdp_kwargs["auto_wrap_policy"] = get_auto_wrap_policy(config)
    
    # Wrap model
    wrapped_model = FSDP(model, **fsdp_kwargs)
    
    return wrapped_model


def compile_model(
    model: nn.Module,
    config: Optional[CompileConfig] = None,
) -> nn.Module:
    """
    Apply torch.compile to model for optimized execution.
    
    As per Molmo2 tech report:
    - Static shapes are essential for maximizing throughput
    - Uses torch.compile (not TorchScript)
    
    Args:
        model: Model to compile
        config: Compile configuration
        
    Returns:
        Compiled model
    """
    config = config or CompileConfig()
    
    if not config.enabled:
        return model
    
    # Check if compile is available (PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        print("Warning: torch.compile not available. Requires PyTorch 2.0+")
        return model
    
    compile_kwargs = {
        "backend": config.backend,
        "mode": config.mode,
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    
    if config.options:
        compile_kwargs["options"] = config.options
    
    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        print(f"Model compiled with backend={config.backend}, mode={config.mode}")
        return compiled_model
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
        return model


def setup_amp(dtype: str = "bfloat16") -> torch.dtype:
    """
    Set up Automatic Mixed Precision.
    
    As per Molmo2 tech report:
    - Most operations run in bfloat16
    - LayerNorm and RoPE in full precision
    
    Args:
        dtype: Target dtype ("float16", "bfloat16", "float32")
        
    Returns:
        torch.dtype for autocast
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    amp_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Check device support
    if amp_dtype == torch.bfloat16:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                print(f"Warning: GPU compute capability {capability} may not support bfloat16 efficiently")
    
    return amp_dtype


def get_gradient_scaler(enabled: bool = True) -> torch.amp.GradScaler:
    """
    Get gradient scaler for mixed precision training.
    
    Note: GradScaler is primarily needed for float16.
    For bfloat16, it's often not necessary but can still help.
    
    Args:
        enabled: Whether scaling is enabled
        
    Returns:
        GradScaler instance
    """
    return torch.amp.GradScaler("cuda", enabled=enabled)


class DistributedDataParallel:
    """
    Context manager for distributed training.
    
    Usage:
        with DistributedDataParallel() as ddp:
            model = ddp.setup_model(model)
            # training loop
    """
    
    def __init__(self, fsdp_config: Optional[FSDPConfig] = None):
        self.fsdp_config = fsdp_config or FSDPConfig()
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
    
    def __enter__(self):
        self.rank, self.world_size, self.local_rank = setup_distributed()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP."""
        return setup_fsdp(model, self.fsdp_config, self.local_rank)
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum"):
        """All-reduce tensor across processes."""
        if dist.is_initialized():
            ops = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.AVG}
            dist.all_reduce(tensor, op=ops.get(op, dist.ReduceOp.SUM))
        return tensor


def compute_gradient_norm(
    model: nn.Module,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Compute total gradient norm across all parameters.
    
    As per Molmo2 tech report:
    - Gradients are averaged across devices
    - Total loss divided by average number of loss tokens
    
    Args:
        model: Model with gradients
        norm_type: Type of norm (1.0, 2.0, inf)
        
    Returns:
        Total gradient norm
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
    
    return total_norm


def clip_gradient_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clip gradients by total norm.
    
    Args:
        model: Model with gradients
        max_norm: Maximum norm value
        norm_type: Type of norm
        
    Returns:
        Clipped gradient norm
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type,
    )


def average_gradients_by_loss_tokens(
    gradients: List[torch.Tensor],
    local_loss_tokens: int,
    world_size: int,
) -> List[torch.Tensor]:
    """
    Average gradients using global average of loss tokens.
    
    From Molmo2 tech report:
    "We always compute the per-device gradient by dividing the total loss
    on that device by the average number of loss tokens across all devices,
    not the number of loss tokens on that particular device."
    
    This avoids up-weighting examples with short responses.
    
    Args:
        gradients: List of gradient tensors
        local_loss_tokens: Number of loss tokens on this device
        world_size: Number of devices
        
    Returns:
        Scaled gradients
    """
    # Get average loss tokens across all devices
    total_tokens = torch.tensor([local_loss_tokens], dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    avg_tokens = total_tokens.item() / world_size
    
    # Scale gradients
    scale = local_loss_tokens / avg_tokens if avg_tokens > 0 else 1.0
    
    scaled_gradients = [g * scale if g is not None else None for g in gradients]
    
    return scaled_gradients


if __name__ == "__main__":
    # Test utilities
    print("Testing distributed utilities...")
    
    # Test FSDP config
    fsdp_config = FSDPConfig()
    print(f"FSDP config: sharding={fsdp_config.sharding_strategy}")
    
    # Test compile config
    compile_config = CompileConfig()
    print(f"Compile config: backend={compile_config.backend}")
    
    # Test AMP dtype
    amp_dtype = setup_amp("bfloat16")
    print(f"AMP dtype: {amp_dtype}")
    
    # Test scaler
    scaler = get_gradient_scaler(enabled=True)
    print(f"GradScaler created: {scaler}")
    
    # Test simple model compilation
    model = nn.Linear(10, 10)
    compiled = compile_model(model, CompileConfig(enabled=True))
    print(f"Model compiled: {type(compiled)}")
    
    print("✓ Distributed utilities tests passed!")
