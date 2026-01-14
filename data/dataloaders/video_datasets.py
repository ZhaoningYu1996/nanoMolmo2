"""
Video-based dataset implementations for Molmo2.

Implements datasets for video understanding tasks:
- Dense video captioning
- Video pointing and tracking (novel contribution)
- Long-form video QA
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

from .base import MultimodalDataset, MultimodalSample
from .utils import TokenWeightingStrategy


class VideoCaptioningDataset(MultimodalDataset):
    """
    Dense video captioning dataset.
    
    Part of Molmo2 Data release. Combines:
    - Human narration
    - Transcription
    - LLM enrichment with frame-level details
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_frames: int = 128,  # F=128 for standard, F=384 for long-context
        fps: float = 2.0,  # S=2 fps
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize video captioning dataset.
        
        Expected data format (JSONL):
        {
            "video_path": "path/to/video.mp4",
            "caption": "detailed frame-by-frame caption...",
            "subtitles": "optional subtitle text"
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_frames: Maximum number of frames to sample (F)
            fps: Frame sampling rate (S)
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_frames=max_frames,
            fps=fps,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("video_caption"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load video captioning data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "video_path": item["video_path"],
                    "caption": item["caption"],
                    "subtitles": item.get("subtitles", None),
                })
    
    def sample_video_frames(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
    ) -> Tuple[List[Image.Image], List[float]]:
        """
        Sample video frames at specified FPS with max frame limit.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS (uses self.fps if None)
            
        Returns:
            Tuple of (frames, timestamps)
        """
        target_fps = target_fps or self.fps
        
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        # Calculate frame indices to sample
        frame_interval = int(original_fps / target_fps)
        frame_indices = list(range(0, total_frames, frame_interval))[:self.max_frames]
        
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            timestamp = idx / original_fps
            frames.append(pil_image)
            timestamps.append(timestamp)
        
        cap.release()
        return frames, timestamps
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single video captioning sample."""
        sample = self.samples[idx]
        
        # Sample video frames
        frames, timestamps = self.sample_video_frames(sample["video_path"])
        
        return MultimodalSample(
            visual_inputs=frames,
            text=sample["caption"],
            input_type="video",
            timestamps=timestamps,
            subtitles=sample["subtitles"],
            task_weight=self.task_weight,
        )


class VideoPointingDataset(MultimodalDataset):
    """
    Video pointing dataset - novel contribution of Molmo2.
    
    Extends 2D pointing to temporal domain (space + time).
    Points are represented in HTML-like format with timestamps.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_frames: int = 128,
        fps: float = 2.0,
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize video pointing dataset.
        
        Expected data format (JSONL):
        {
            "video_path": "path/to/video.mp4",
            "query": "Point to the person throwing the ball",
            "points": [
                {"x": 0.5, "y": 0.3, "timestamp": 1.2, "object_id": 0},
                {"x": 0.52, "y": 0.31, "timestamp": 1.4, "object_id": 0},
                ...
            ]
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_frames: Maximum number of frames to sample (F)
            fps: Frame sampling rate (S)
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_frames=max_frames,
            fps=fps,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("video_pointing"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load video pointing data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "video_path": item["video_path"],
                    "query": item["query"],
                    "points": item["points"],
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single video pointing sample."""
        sample = self.samples[idx]
        
        # Sample video frames
        frames, timestamps = self.sample_video_frames(sample["video_path"])
        
        # Format output with temporal points
        points_text = self.format_pointing_output(
            points=sample["points"],
            input_type="video"
        )
        
        full_text = f"Q: {sample['query']}\nA: {points_text}"
        
        return MultimodalSample(
            visual_inputs=frames,
            text=full_text,
            input_type="video",
            timestamps=timestamps,
            points=sample["points"],
            task_weight=self.task_weight,
        )


class VideoTrackingDataset(MultimodalDataset):
    """
    Video object tracking dataset - another novel contribution.
    
    Teaches model to track objects across video frames.
    Significantly outperforms prior models (56.2 vs 41.1 J&F on benchmarks).
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_frames: int = 128,
        fps: float = 2.0,
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize video tracking dataset.
        
        Expected data format (JSONL):
        {
            "video_path": "path/to/video.mp4",
            "query": "Track the red car throughout the video",
            "tracks": [
                {"x": 0.5, "y": 0.3, "timestamp": 0.0, "object_id": 0},
                {"x": 0.51, "y": 0.31, "timestamp": 0.5, "object_id": 0},
                {"x": 0.53, "y": 0.33, "timestamp": 1.0, "object_id": 0},
                ...
            ]
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_frames: Maximum number of frames to sample (F)
            fps: Frame sampling rate (S)
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_frames=max_frames,
            fps=fps,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("video_tracking"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load video tracking data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "video_path": item["video_path"],
                    "query": item["query"],
                    "tracks": item["tracks"],
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single video tracking sample."""
        sample = self.samples[idx]
        
        # Sample video frames
        frames, timestamps = self.sample_video_frames(sample["video_path"])
        
        # Format tracking output (temporal point sequence with same object_id)
        tracks_text = self.format_pointing_output(
            points=sample["tracks"],
            input_type="video"
        )
        
        full_text = f"Q: {sample['query']}\nA: {tracks_text}"
        
        return MultimodalSample(
            visual_inputs=frames,
            text=full_text,
            input_type="video",
            timestamps=timestamps,
            points=sample["tracks"],
            task_weight=self.task_weight,
        )


class VideoQADataset(MultimodalDataset):
    """
    Long-form video Question Answering dataset.
    
    Part of Molmo2 Data release. Created using human-LLM collaboration
    pipeline without distillation from proprietary models.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_frames: int = 128,
        fps: float = 2.0,
        max_seq_length: int = 4096,
        **kwargs
    ):
        """
        Initialize video QA dataset.
        
        Expected data format (JSONL):
        {
            "video_path": "path/to/video.mp4",
            "question": "What happens in the video?",
            "answer": "Long-form detailed answer...",
            "subtitles": "optional subtitle text"
        }
        
        Args:
            data_path: Path to JSONL file or directory
            split: Dataset split ("train" or "val")
            max_frames: Maximum number of frames to sample (F)
            fps: Frame sampling rate (S)
            max_seq_length: Maximum sequence length
        """
        super().__init__(
            data_path=data_path,
            max_frames=max_frames,
            fps=fps,
            max_seq_length=max_seq_length,
            task_weight=TokenWeightingStrategy.get_task_weight("video_qa"),
            **kwargs
        )
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load video QA data from JSONL."""
        data_file = Path(self.data_path)
        if data_file.is_dir():
            data_file = data_file / f"{self.split}.jsonl"
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "video_path": item["video_path"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "subtitles": item.get("subtitles", None),
                })
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a single video QA sample."""
        sample = self.samples[idx]
        
        # Sample video frames
        frames, timestamps = self.sample_video_frames(sample["video_path"])
        
        # Format as Q&A
        full_text = f"Q: {sample['question']}\nA: {sample['answer']}"
        
        return MultimodalSample(
            visual_inputs=frames,
            text=full_text,
            input_type="video",
            timestamps=timestamps,
            subtitles=sample["subtitles"],
            task_weight=self.task_weight,
        )
