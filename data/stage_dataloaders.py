"""
Stage-specific dataloaders for Molmo2's 3-stage training pipeline.

Implements:
- Stage 1: Pre-training (vision-language alignment)
- Stage 2: Supervised Fine-Tuning (instruction following)
- Stage 3: Long-context SFT (extended sequences)

Each stage has specific data mixing ratios and configurations.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader
import logging

from .dataloaders.base import MultimodalCollator
from .dataloaders.image_datasets import (
    CaptioningDataset,
    PointingDataset,
    VQADataset,
    CountingDataset,
)
from .dataloaders.video_datasets import (
    VideoCaptioningDataset,
    VideoPointingDataset,
    VideoTrackingDataset,
    VideoQADataset,
)
from .dataloaders.utils import DataMixingConfig, SequencePacker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Stage1PretrainingDataModule:
    """
    Data module for Stage 1: Pre-training.
    
    Configuration:
    - 60% Dense captioning (PixMo-Cap)
    - 30% Image pointing (PixMo-Points, PixMo-Count)
    - 10% NLP data (Tulu)
    
    Training details:
    - Sequence length: 4,096 tokens
    - Frame limit: N/A (image-only in pre-training)
    - Steps: ~100K (or ~32K as per paper)
    """
    
    def __init__(
        self,
        data_root: str = "./data/molmo2_datasets",
        max_seq_length: int = 4096,
        use_sequence_packing: bool = True,
    ):
        """
        Initialize pre-training data module.
        
        Args:
            data_root: Root directory containing downloaded datasets
            max_seq_length: Maximum sequence length
            use_sequence_packing: Whether to use sequence packing
        """
        self.data_root = Path(data_root)
        self.max_seq_length = max_seq_length
        self.use_sequence_packing = use_sequence_packing
        
        # Get mixing configuration
        self.mixing_config = DataMixingConfig.get_pretrain_config()
        
        logger.info("="*60)
        logger.info("Stage 1: Pre-training Data Module")
        logger.info("="*60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"Sequence packing: {self.use_sequence_packing}")
        logger.info(f"Mixing config: {self.mixing_config.dataset_weights}")
    
    def setup_datasets(self, split: str = "train") -> Dict[str, Dataset]:
        """
        Set up all datasets for pre-training stage.
        
        Args:
            split: Dataset split ("train" or "val")
            
        Returns:
            Dictionary mapping dataset names to Dataset objects
        """
        datasets = {}
        
        # 60% Dense captioning
        try:
            datasets["pixmo-cap"] = CaptioningDataset(
                data_path=self.data_root / "pixmo-cap",
                split=split,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded pixmo-cap: {len(datasets['pixmo-cap'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load pixmo-cap: {e}")
        
        # 30% Image pointing (PixMo-Points)
        try:
            datasets["pixmo-points"] = PointingDataset(
                data_path=self.data_root / "pixmo-points",
                split=split,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded pixmo-points: {len(datasets['pixmo-points'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load pixmo-points: {e}")
        
        # Additional pointing: PixMo-Count
        try:
            datasets["pixmo-count"] = CountingDataset(
                data_path=self.data_root / "pixmo-count",
                split=split,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded pixmo-count: {len(datasets['pixmo-count'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load pixmo-count: {e}")
        
        # 10% NLP data (Tulu) - Will be added when text-only dataset is implemented
        # datasets["tulu"] = TextDataset(...)
        
        if not datasets:
            raise ValueError("No datasets loaded! Check data_root and dataset availability.")
        
        return datasets
    
    def create_mixed_dataloader(
        self,
        tokenizer,
        image_processor,
        batch_size: int = 32,
        num_workers: int = 4,
        split: str = "train",
    ) -> DataLoader:
        """
        Create mixed dataloader with proper sampling weights.
        
        Args:
            tokenizer: Text tokenizer
            image_processor: Vision processor
            batch_size: Batch size
            num_workers: Number of data loading workers
            split: Dataset split
            
        Returns:
            DataLoader with mixed datasets
        """
        datasets = self.setup_datasets(split=split)
        
        # Create sampling weights based on mixing configuration
        weights = []
        dataset_list = []
        
        # Map dataset names to mixing weights
        weight_mapping = {
            "pixmo-cap": self.mixing_config.dataset_weights.get("dense_captioning", 0.6),
            "pixmo-points": self.mixing_config.dataset_weights.get("image_pointing", 0.3) * 0.7,  # 70% of pointing
            "pixmo-count": self.mixing_config.dataset_weights.get("image_pointing", 0.3) * 0.3,   # 30% of pointing
        }
        
        for name, dataset in datasets.items():
            dataset_list.append(dataset)
            weight = weight_mapping.get(name, 1.0)
            weights.extend([weight] * len(dataset))
        
        # Combine datasets
        combined_dataset = ConcatDataset(dataset_list)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined_dataset),
            replacement=True,
        )
        
        # Create collator
        collator = MultimodalCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_seq_length=self.max_seq_length,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"✓ Created pre-training dataloader: {len(combined_dataset)} samples, batch_size={batch_size}")
        return dataloader


class Stage2SFTDataModule:
    """
    Data module for Stage 2: Supervised Fine-Tuning.
    
    Configuration:
    - Sqrt-proportional sampling across 100+ datasets
    - Manual rebalancing based on validation performance
    
    Training details:
    - Sequence length: 4,096 tokens
    - Frame limit: 128 frames (videos)
    - Frame rate: 2 fps
    - Steps: ~50K
    """
    
    def __init__(
        self,
        data_root: str = "./data/molmo2_datasets",
        max_seq_length: int = 4096,
        max_frames: int = 128,
        fps: float = 2.0,
        use_sequence_packing: bool = True,
    ):
        """
        Initialize SFT data module.
        
        Args:
            data_root: Root directory containing downloaded datasets
            max_seq_length: Maximum sequence length
            max_frames: Maximum video frames
            fps: Video frame sampling rate
            use_sequence_packing: Whether to use sequence packing
        """
        self.data_root = Path(data_root)
        self.max_seq_length = max_seq_length
        self.max_frames = max_frames
        self.fps = fps
        self.use_sequence_packing = use_sequence_packing
        
        logger.info("="*60)
        logger.info("Stage 2: SFT Data Module")
        logger.info("="*60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"Max frames: {self.max_frames}, FPS: {self.fps}")
        logger.info(f"Sequence packing: {self.use_sequence_packing}")
    
    def setup_datasets(self, split: str = "train") -> Dict[str, Dataset]:
        """
        Set up all datasets for SFT stage.
        
        Args:
            split: Dataset split ("train" or "val")
            
        Returns:
            Dictionary mapping dataset names to Dataset objects
        """
        datasets = {}
        
        # === Molmo2 Original Datasets ===
        
        # Video captioning
        try:
            datasets["molmo2-cap"] = VideoCaptioningDataset(
                data_path=self.data_root / "molmo2-cap",
                split=split,
                max_frames=self.max_frames,
                fps=self.fps,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded molmo2-cap: {len(datasets['molmo2-cap'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load molmo2-cap: {e}")
        
        # Video QA datasets
        for dataset_name in ["molmo2-capqa", "molmo2-subtitleqa", "molmo2-askmodelanything"]:
            try:
                datasets[dataset_name] = VideoQADataset(
                    data_path=self.data_root / dataset_name,
                    split=split,
                    max_frames=self.max_frames,
                    fps=self.fps,
                    max_seq_length=self.max_seq_length,
                )
                logger.info(f"✓ Loaded {dataset_name}: {len(datasets[dataset_name])} samples")
            except Exception as e:
                logger.warning(f"✗ Failed to load {dataset_name}: {e}")
        
        # Video pointing (novel)
        try:
            datasets["molmo2-videopoint"] = VideoPointingDataset(
                data_path=self.data_root / "molmo2-videopoint",
                split=split,
                max_frames=self.max_frames,
                fps=self.fps,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded molmo2-videopoint: {len(datasets['molmo2-videopoint'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load molmo2-videopoint: {e}")
        
        # Video tracking (novel)
        try:
            datasets["molmo2-videotrack"] = VideoTrackingDataset(
                data_path=self.data_root / "molmo2-videotrack",
                split=split,
                max_frames=self.max_frames,
                fps=self.fps,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded molmo2-videotrack: {len(datasets['molmo2-videotrack'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load molmo2-videotrack: {e}")
        
        # === PixMo Datasets ===
        
        try:
            datasets["pixmo-cap"] = CaptioningDataset(
                data_path=self.data_root / "pixmo-cap",
                split=split,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded pixmo-cap: {len(datasets['pixmo-cap'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load pixmo-cap: {e}")
        
        try:
            datasets["pixmo-askmodelanything"] = VQADataset(
                data_path=self.data_root / "pixmo-askmodelanything",
                split=split,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded pixmo-askmodelanything: {len(datasets['pixmo-askmodelanything'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load pixmo-askmodelanything: {e}")
        
        # === Academic VQA Datasets ===
        
        for dataset_name in ["vqa-v2", "docvqa", "textvqa", "chartqa"]:
            try:
                datasets[dataset_name] = VQADataset(
                    data_path=self.data_root / dataset_name,
                    split=split,
                    max_seq_length=self.max_seq_length,
                )
                logger.info(f"✓ Loaded {dataset_name}: {len(datasets[dataset_name])} samples")
            except Exception as e:
                logger.warning(f"✗ Failed to load {dataset_name}: {e}")
        
        if not datasets:
            raise ValueError("No datasets loaded! Check data_root and dataset availability.")
        
        return datasets
    
    def create_mixed_dataloader(
        self,
        tokenizer,
        image_processor,
        batch_size: int = 16,
        num_workers: int = 4,
        split: str = "train",
    ) -> DataLoader:
        """
        Create mixed dataloader with sqrt-proportional sampling.
        
        Args:
            tokenizer: Text tokenizer
            image_processor: Vision processor
            batch_size: Batch size
            num_workers: Number of data loading workers
            split: Dataset split
            
        Returns:
            DataLoader with mixed datasets
        """
        datasets = self.setup_datasets(split=split)
        
        # Compute dataset sizes for sqrt-proportional weighting
        dataset_sizes = {name: len(dataset) for name, dataset in datasets.items()}
        
        # Get SFT mixing config with sqrt-proportional sampling
        mixing_config = DataMixingConfig.get_sft_config(dataset_sizes)
        
        # Create sampling weights
        weights = []
        dataset_list = []
        
        for name, dataset in datasets.items():
            dataset_list.append(dataset)
            weight = mixing_config.dataset_weights.get(name, 1.0)
            weights.extend([weight] * len(dataset))
        
        # Combine datasets
        combined_dataset = ConcatDataset(dataset_list)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined_dataset),
            replacement=True,
        )
        
        # Create collator
        collator = MultimodalCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_seq_length=self.max_seq_length,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"✓ Created SFT dataloader: {len(combined_dataset)} samples, batch_size={batch_size}")
        logger.info(f"Dataset weights (top 5): {dict(list(mixing_config.dataset_weights.items())[:5])}")
        return dataloader


class Stage3LongContextDataModule:
    """
    Data module for Stage 3: Long-context SFT.
    
    Configuration:
    - Same datasets as SFT stage
    - Extended sequences and frames
    
    Training details:
    - Sequence length: 36,864 tokens (9x longer)
    - Frame limit: 384 frames (3x more)
    - Frame rate: 2 fps
    - Steps: 2K (short fine-tuning)
    - Requires: Context Parallelism (CP), Ulysses attention
    """
    
    def __init__(
        self,
        data_root: str = "./data/molmo2_datasets",
        max_seq_length: int = 36864,  # 9x longer
        max_frames: int = 384,         # 3x more frames
        fps: float = 2.0,
        use_sequence_packing: bool = False,  # Usually disabled for long-context
    ):
        """
        Initialize long-context data module.
        
        Args:
            data_root: Root directory containing downloaded datasets
            max_seq_length: Maximum sequence length (36,864 tokens)
            max_frames: Maximum video frames (384 frames)
            fps: Video frame sampling rate
            use_sequence_packing: Whether to use sequence packing (usually disabled)
        """
        self.data_root = Path(data_root)
        self.max_seq_length = max_seq_length
        self.max_frames = max_frames
        self.fps = fps
        self.use_sequence_packing = use_sequence_packing
        
        logger.info("="*60)
        logger.info("Stage 3: Long-Context SFT Data Module")
        logger.info("="*60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Max sequence length: {self.max_seq_length} (9x longer)")
        logger.info(f"Max frames: {self.max_frames} (3x more), FPS: {self.fps}")
        logger.info(f"Sequence packing: {self.use_sequence_packing}")
        logger.info("⚠ Requires Context Parallelism (CP) and Ulysses attention")
    
    def setup_datasets(self, split: str = "train") -> Dict[str, Dataset]:
        """
        Set up datasets for long-context stage.
        
        Uses same datasets as SFT but with extended configurations.
        
        Args:
            split: Dataset split ("train" or "val")
            
        Returns:
            Dictionary mapping dataset names to Dataset objects
        """
        datasets = {}
        
        # Focus on video datasets that benefit from long context
        
        # Video captioning
        try:
            datasets["molmo2-cap"] = VideoCaptioningDataset(
                data_path=self.data_root / "molmo2-cap",
                split=split,
                max_frames=self.max_frames,  # 384 frames
                fps=self.fps,
                max_seq_length=self.max_seq_length,  # 36,864 tokens
            )
            logger.info(f"✓ Loaded molmo2-cap: {len(datasets['molmo2-cap'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load molmo2-cap: {e}")
        
        # Video QA (benefits from long context)
        for dataset_name in ["molmo2-capqa", "molmo2-subtitleqa"]:
            try:
                datasets[dataset_name] = VideoQADataset(
                    data_path=self.data_root / dataset_name,
                    split=split,
                    max_frames=self.max_frames,
                    fps=self.fps,
                    max_seq_length=self.max_seq_length,
                )
                logger.info(f"✓ Loaded {dataset_name}: {len(datasets[dataset_name])} samples")
            except Exception as e:
                logger.warning(f"✗ Failed to load {dataset_name}: {e}")
        
        # Video tracking (benefits from long context)
        try:
            datasets["molmo2-videotrack"] = VideoTrackingDataset(
                data_path=self.data_root / "molmo2-videotrack",
                split=split,
                max_frames=self.max_frames,
                fps=self.fps,
                max_seq_length=self.max_seq_length,
            )
            logger.info(f"✓ Loaded molmo2-videotrack: {len(datasets['molmo2-videotrack'])} samples")
        except Exception as e:
            logger.warning(f"✗ Failed to load molmo2-videotrack: {e}")
        
        if not datasets:
            raise ValueError("No datasets loaded! Check data_root and dataset availability.")
        
        return datasets
    
    def create_mixed_dataloader(
        self,
        tokenizer,
        image_processor,
        batch_size: int = 4,  # Smaller batch size due to long sequences
        num_workers: int = 4,
        split: str = "train",
    ) -> DataLoader:
        """
        Create mixed dataloader for long-context training.
        
        Args:
            tokenizer: Text tokenizer
            image_processor: Vision processor
            batch_size: Batch size (smaller due to memory)
            num_workers: Number of data loading workers
            split: Dataset split
            
        Returns:
            DataLoader with mixed datasets
        """
        datasets = self.setup_datasets(split=split)
        
        # Compute dataset sizes for sqrt-proportional weighting
        dataset_sizes = {name: len(dataset) for name, dataset in datasets.items()}
        
        # Use same SFT mixing strategy
        mixing_config = DataMixingConfig.get_sft_config(dataset_sizes)
        
        # Create sampling weights
        weights = []
        dataset_list = []
        
        for name, dataset in datasets.items():
            dataset_list.append(dataset)
            weight = mixing_config.dataset_weights.get(name, 1.0)
            weights.extend([weight] * len(dataset))
        
        # Combine datasets
        combined_dataset = ConcatDataset(dataset_list)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined_dataset),
            replacement=True,
        )
        
        # Create collator
        collator = MultimodalCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_seq_length=self.max_seq_length,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"✓ Created long-context dataloader: {len(combined_dataset)} samples, batch_size={batch_size}")
        logger.info(f"⚠ Memory warning: Each sample can be up to {self.max_seq_length} tokens with {self.max_frames} frames")
        return dataloader


# ============================================================================
# Convenience Functions
# ============================================================================

def get_stage_dataloader(
    stage: int,
    tokenizer,
    image_processor,
    data_root: str = "./data/molmo2_datasets",
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    split: str = "train",
    **kwargs
) -> DataLoader:
    """
    Get dataloader for a specific training stage.
    
    Args:
        stage: Training stage (1, 2, or 3)
        tokenizer: Text tokenizer
        image_processor: Vision processor
        data_root: Root directory containing datasets
        batch_size: Batch size (uses stage-specific defaults if None)
        num_workers: Number of data loading workers
        split: Dataset split ("train" or "val")
        **kwargs: Additional arguments for data module
        
    Returns:
        DataLoader for the specified stage
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>> image_processor = ...
        >>> 
        >>> # Stage 1: Pre-training
        >>> stage1_loader = get_stage_dataloader(
        >>>     stage=1,
        >>>     tokenizer=tokenizer,
        >>>     image_processor=image_processor,
        >>>     batch_size=32,
        >>> )
        >>> 
        >>> # Stage 2: SFT
        >>> stage2_loader = get_stage_dataloader(
        >>>     stage=2,
        >>>     tokenizer=tokenizer,
        >>>     image_processor=image_processor,
        >>>     batch_size=16,
        >>> )
        >>> 
        >>> # Stage 3: Long-context
        >>> stage3_loader = get_stage_dataloader(
        >>>     stage=3,
        >>>     tokenizer=tokenizer,
        >>>     image_processor=image_processor,
        >>>     batch_size=4,
        >>> )
    """
    if stage == 1:
        # Stage 1: Pre-training
        if batch_size is None:
            batch_size = 32
        
        data_module = Stage1PretrainingDataModule(
            data_root=data_root,
            **kwargs
        )
        return data_module.create_mixed_dataloader(
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
        )
    
    elif stage == 2:
        # Stage 2: SFT
        if batch_size is None:
            batch_size = 16
        
        data_module = Stage2SFTDataModule(
            data_root=data_root,
            **kwargs
        )
        return data_module.create_mixed_dataloader(
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
        )
    
    elif stage == 3:
        # Stage 3: Long-context
        if batch_size is None:
            batch_size = 4
        
        data_module = Stage3LongContextDataModule(
            data_root=data_root,
            **kwargs
        )
        return data_module.create_mixed_dataloader(
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
        )
    
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")


def get_all_stage_dataloaders(
    tokenizer,
    image_processor,
    data_root: str = "./data/molmo2_datasets",
    batch_sizes: Optional[Dict[int, int]] = None,
    num_workers: int = 4,
    split: str = "train",
) -> Dict[int, DataLoader]:
    """
    Get dataloaders for all 3 training stages.
    
    Args:
        tokenizer: Text tokenizer
        image_processor: Vision processor
        data_root: Root directory containing datasets
        batch_sizes: Dict mapping stage number to batch size
        num_workers: Number of data loading workers
        split: Dataset split ("train" or "val")
        
    Returns:
        Dictionary mapping stage number to DataLoader
        
    Example:
        >>> dataloaders = get_all_stage_dataloaders(
        >>>     tokenizer=tokenizer,
        >>>     image_processor=image_processor,
        >>>     batch_sizes={1: 32, 2: 16, 3: 4},
        >>> )
        >>> 
        >>> # Train stage 1
        >>> for batch in dataloaders[1]:
        >>>     ...
        >>> 
        >>> # Train stage 2
        >>> for batch in dataloaders[2]:
        >>>     ...
        >>> 
        >>> # Train stage 3
        >>> for batch in dataloaders[3]:
        >>>     ...
    """
    if batch_sizes is None:
        batch_sizes = {1: 32, 2: 16, 3: 4}
    
    dataloaders = {}
    for stage in [1, 2, 3]:
        dataloaders[stage] = get_stage_dataloader(
            stage=stage,
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_root=data_root,
            batch_size=batch_sizes.get(stage),
            num_workers=num_workers,
            split=split,
        )
    
    return dataloaders
