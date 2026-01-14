"""
Utility classes for efficient data loading in Molmo2.

Implements:
- Sequence packing algorithm for merging short examples
- Message-tree encoding for videos with multiple annotations
- Token weighting strategies
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch


class SequencePacker:
    """
    Implements sequence packing algorithm from Molmo2 paper.
    
    Merges short examples into single long sequences to improve
    training throughput while respecting max sequence length.
    """
    
    def __init__(self, max_seq_length: int = 4096):
        """
        Initialize sequence packer.
        
        Args:
            max_seq_length: Maximum sequence length for packing
        """
        self.max_seq_length = max_seq_length
    
    def pack_sequences(
        self,
        samples: List[Dict],
        sep_token_id: int,
    ) -> List[Dict]:
        """
        Pack multiple short sequences into longer ones.
        
        Algorithm:
        1. Sort samples by length
        2. Greedily pack short samples together
        3. Insert separator tokens between packed samples
        4. Track loss mask to avoid cross-sample attention
        
        Args:
            samples: List of tokenized samples with input_ids, attention_mask, etc.
            sep_token_id: Token ID for separator between packed samples
            
        Returns:
            List of packed samples
        """
        packed_samples = []
        current_pack = {
            "input_ids": [],
            "attention_mask": [],
            "loss_weights": [],
            "pack_boundaries": [],  # Track where samples are packed
        }
        current_length = 0
        
        # Sort by length for efficient packing
        sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]))
        
        for sample in sorted_samples:
            sample_length = len(sample["input_ids"])
            
            # Check if sample fits in current pack
            if current_length + sample_length + 1 <= self.max_seq_length:  # +1 for separator
                if current_length > 0:
                    # Add separator
                    current_pack["input_ids"].append(sep_token_id)
                    current_pack["attention_mask"].append(1)
                    current_pack["loss_weights"].append(0.0)  # No loss on separator
                    current_length += 1
                
                # Add sample
                current_pack["input_ids"].extend(sample["input_ids"])
                current_pack["attention_mask"].extend(sample["attention_mask"])
                current_pack["loss_weights"].extend(sample.get("loss_weights", [1.0] * sample_length))
                current_pack["pack_boundaries"].append((current_length, current_length + sample_length))
                current_length += sample_length
            else:
                # Current pack is full, save it and start new pack
                if current_length > 0:
                    packed_samples.append(self._finalize_pack(current_pack))
                
                # Start new pack with current sample
                current_pack = {
                    "input_ids": list(sample["input_ids"]),
                    "attention_mask": list(sample["attention_mask"]),
                    "loss_weights": list(sample.get("loss_weights", [1.0] * sample_length)),
                    "pack_boundaries": [(0, sample_length)],
                }
                current_length = sample_length
        
        # Add final pack
        if current_length > 0:
            packed_samples.append(self._finalize_pack(current_pack))
        
        return packed_samples
    
    def _finalize_pack(self, pack: Dict) -> Dict:
        """Convert packed dict to final format with proper masking."""
        return {
            "input_ids": torch.tensor(pack["input_ids"]),
            "attention_mask": torch.tensor(pack["attention_mask"]),
            "loss_weights": torch.tensor(pack["loss_weights"]),
            "pack_boundaries": pack["pack_boundaries"],
        }


class MessageTreeEncoder:
    """
    Implements message-tree encoding for videos with multiple annotations.
    
    Handles videos with multiple QA pairs or annotations by encoding
    them in a tree structure to maximize training efficiency.
    """
    
    def __init__(self):
        """Initialize message tree encoder."""
        pass
    
    def encode_multi_annotation_video(
        self,
        video_frames: List,
        annotations: List[Dict],
        tokenizer,
    ) -> List[Dict]:
        """
        Encode a video with multiple annotations in tree format.
        
        For a video with N annotations, creates N training samples that
        share the same visual encoding but have different text branches.
        
        Args:
            video_frames: List of video frames (PIL Images)
            annotations: List of annotation dicts with "question" and "answer"
            tokenizer: Text tokenizer
            
        Returns:
            List of encoded samples with shared visual context
        """
        samples = []
        
        # Encode visual frames once (shared across all annotations)
        # In actual model forward pass, this will be cached
        
        for ann_idx, annotation in enumerate(annotations):
            sample = {
                "visual_frames": video_frames,  # Shared reference
                "visual_frame_idx": 0,  # All start with same frames
                "question": annotation["question"],
                "answer": annotation["answer"],
                "annotation_id": ann_idx,
            }
            samples.append(sample)
        
        return samples


class TokenWeightingStrategy:
    """
    Implements token weighting scheme from Molmo2 paper.
    
    Balances learning across diverse tasks by applying per-token weights
    during loss computation. Pointing/tracking tasks get higher weights.
    """
    
    # Task weights from paper (approximate based on description)
    TASK_WEIGHTS = {
        "video_caption": 1.0,
        "image_caption": 1.0,
        "video_pointing": 2.0,  # Higher weight for pointing
        "image_pointing": 2.0,
        "video_tracking": 2.0,  # Higher weight for tracking
        "video_qa": 1.0,
        "image_qa": 1.0,
        "counting": 1.5,  # Moderate increase for counting
    }
    
    @classmethod
    def get_task_weight(cls, task_type: str) -> float:
        """
        Get weighting factor for a given task type.
        
        Args:
            task_type: Task type string
            
        Returns:
            Weight factor (default 1.0 if task not found)
        """
        return cls.TASK_WEIGHTS.get(task_type, 1.0)
    
    @staticmethod
    def apply_token_weights(
        loss: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-token weights to loss.
        
        Args:
            loss: Unweighted loss tensor [batch_size, seq_length]
            loss_weights: Weight tensor [batch_size, seq_length]
            
        Returns:
            Weighted loss
        """
        return (loss * loss_weights).sum() / loss_weights.sum().clamp(min=1.0)


@dataclass
class DataMixingConfig:
    """
    Configuration for dataset mixing during training.
    
    Follows Molmo2's strategy:
    - Pre-training: 60% captioning, 30% pointing, 10% NLP
    - SFT: Manual assignment with sqrt-proportional sampling + rebalancing
    """
    stage: str  # "pretrain", "sft", "long_context"
    dataset_weights: Dict[str, float]
    
    @classmethod
    def get_pretrain_config(cls) -> "DataMixingConfig":
        """Get pre-training mixing configuration."""
        return cls(
            stage="pretrain",
            dataset_weights={
                "dense_captioning": 0.60,
                "image_pointing": 0.30,
                "nlp_data": 0.10,
            }
        )
    
    @classmethod
    def get_sft_config(cls, dataset_sizes: Dict[str, int]) -> "DataMixingConfig":
        """
        Get SFT mixing configuration with sqrt-proportional sampling.
        
        Args:
            dataset_sizes: Dict mapping dataset names to their sizes
            
        Returns:
            DataMixingConfig with computed weights
        """
        # Compute sqrt-proportional weights
        sqrt_sizes = {name: size ** 0.5 for name, size in dataset_sizes.items()}
        total = sum(sqrt_sizes.values())
        weights = {name: sqrt_size / total for name, sqrt_size in sqrt_sizes.items()}
        
        # Manual rebalancing can be applied here based on validation performance
        # (In practice, this would be tuned during training)
        
        return cls(
            stage="sft",
            dataset_weights=weights
        )
