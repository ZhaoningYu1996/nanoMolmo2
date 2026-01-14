"""
Image-based dataset implementations for Molmo2.

Implements datasets for pre-training and SFT stages:
- Dense captioning (PixMo style)
- Image pointing (PixMo-Points, PixMo-Count)
- Visual Question Answering
"""

from typing import Dict, List, Optional
from pathlib import Path
import json
from PIL import Image

from .base import MultimodalDataset, MultimodalSample
from .utils import TokenWeightingStrategy


class CaptioningDataset(MultimodalDataset):
    """
    Dense image captioning dataset.
    
    Used in pre-training stage (60% of mixture).
    Follows PixMo dense captioning style from Molmo2.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize captioning dataset.
        
        Expected data format (JSONL):
        {"image_path": "path/to/image.jpg", "caption": "detailed caption..."}
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("image_caption"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load captioning data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "image_path": item["image_path"],
                    "caption": item["caption"],
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single captioning sample."""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        return MultimodalSample(
            visual_inputs=[image],
            text=sample["caption"],
            input_type="image",
            task_weight=self.task_weight,
        )


class PointingDataset(MultimodalDataset):
    """
    Image pointing dataset (PixMo-Points, PixMo-Count style).
    
    Used in pre-training stage (30% of mixture).
    Teaches model to output coordinates in HTML-like format.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize pointing dataset.
        
        Expected data format (JSONL):
        {
            "image_path": "path/to/image.jpg",
            "query": "Point to all cats",
            "points": [
                {"x": 0.5, "y": 0.3, "object_id": 0},
                {"x": 0.7, "y": 0.6, "object_id": 1}
            ]
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("image_pointing"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load pointing data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "image_path": item["image_path"],
                    "query": item["query"],
                    "points": item["points"],
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single pointing sample."""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Format output with points
        points_text = self.format_pointing_output(
            points=sample["points"],
            input_type="image"
        )
        
        # Construct full text: query + answer
        full_text = f"Q: {sample['query']}\nA: {points_text}"
        
        return MultimodalSample(
            visual_inputs=[image],
            text=full_text,
            input_type="image",
            points=sample["points"],
            task_weight=self.task_weight,
        )


class VQADataset(MultimodalDataset):
    """
    Visual Question Answering dataset.
    
    Used in SFT stage. Supports various VQA formats including:
    - VQA v2.0
    - RealWorldQA
    - Custom VQA datasets
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize VQA dataset.
        
        Expected data format (JSONL):
        {
            "image_path": "path/to/image.jpg",
            "question": "What color is the car?",
            "answer": "Red"
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("image_qa"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load VQA data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "image_path": item["image_path"],
                    "question": item["question"],
                    "answer": item["answer"],
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single VQA sample."""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Format as Q&A
        full_text = f"Q: {sample['question']}\nA: {sample['answer']}"
        
        return MultimodalSample(
            visual_inputs=[image],
            text=full_text,
            input_type="image",
            task_weight=self.task_weight,
        )


class CountingDataset(MultimodalDataset):
    """
    Object counting dataset (PixMo-Count style).
    
    Used in SFT stage with moderate task weight (1.5x).
    Focuses on accurate object counting in images.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize counting dataset.
        
        Expected data format (JSONL):
        {
            "image_path": "path/to/image.jpg",
            "query": "How many apples are there?",
            "count": 5,
            "points": [{"x": ..., "y": ...}, ...]  # Optional
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("counting"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load counting data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "image_path": item["image_path"],
                    "query": item["query"],
                    "count": item["count"],
                    "points": item.get("points", None),
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single counting sample."""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Format answer (optionally include points)
        answer = str(sample["count"])
        if sample["points"]:
            points_text = self.format_pointing_output(
                points=sample["points"],
                input_type="image"
            )
            answer = f"{answer}\n{points_text}"
        
        full_text = f"Q: {sample['query']}\nA: {answer}"
        
        return MultimodalSample(
            visual_inputs=[image],
            text=full_text,
            input_type="image",
            points=sample["points"],
            task_weight=self.task_weight,
        )
