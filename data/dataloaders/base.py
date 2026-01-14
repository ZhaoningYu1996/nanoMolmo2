"""
Base classes for Molmo2 data loading.

Following the Molmo2 paper's architecture:
- Interleaved visual and text tokens
- Support for images, videos, and multi-image inputs
- Frame/image timestamps and indices
- Column tokens for multi-crop images
- Subtitle integration
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class MultimodalSample:
    """
    A single training sample with multimodal inputs.
    
    Attributes:
        visual_inputs: List of PIL Images or video frames
        text: Text input/output
        input_type: "image", "video", or "multi_image"
        timestamps: Frame timestamps for videos (optional)
        image_indices: Image indices for multi-image inputs (optional)
        subtitles: Video subtitles as text (optional)
        points: Ground-truth points for pointing/tracking tasks (optional)
        task_weight: Token weighting factor for this sample's task
    """
    visual_inputs: List[Image.Image]
    text: str
    input_type: str  # "image", "video", "multi_image"
    timestamps: Optional[List[float]] = None
    image_indices: Optional[List[int]] = None
    subtitles: Optional[str] = None
    points: Optional[List[Dict]] = None  # Format: [{"x": float, "y": float, "timestamp": float, "object_id": int}]
    task_weight: float = 1.0


class MultimodalDataset(Dataset):
    """
    Base dataset class for Molmo2 multimodal data.
    
    Implements core functionality:
    - Loading and preprocessing images/videos
    - Handling different input types (image, video, multi-image)
    - Task-specific token weighting
    - Message-tree encoding support
    """
    
    def __init__(
        self,
        data_path: str,
        max_frames: int = 128,  # F=128 for standard training, F=384 for long-context
        fps: float = 2.0,  # S=2 fps as per paper
        max_seq_length: int = 4096,  # Sequence length before long-context stage
        task_weight: float = 1.0,
    ):
        """
        Initialize base multimodal dataset.
        
        Args:
            data_path: Path to dataset
            max_frames: Maximum number of video frames (F)
            fps: Frame sampling rate (S)
            max_seq_length: Maximum sequence length
            task_weight: Token weighting factor for this dataset's task
        """
        self.data_path = data_path
        self.max_frames = max_frames
        self.fps = fps
        self.max_seq_length = max_seq_length
        self.task_weight = task_weight
        self.samples = []
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> MultimodalSample:
        raise NotImplementedError("Subclasses must implement __getitem__")
    
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
        # Implementation will use cv2 or decord for efficient video loading
        raise NotImplementedError("Video sampling to be implemented")
    
    def format_pointing_output(
        self,
        points: List[Dict],
        input_type: str,
    ) -> str:
        """
        Format points in HTML-like compressed format from paper:
        <point timestamp/image_index, x, y, object_id>
        
        Args:
            points: List of point dictionaries
            input_type: Type of input ("image", "video", "multi_image")
            
        Returns:
            Formatted pointing string
        """
        formatted_points = []
        for point in points:
            x, y = point["x"], point["y"]
            obj_id = point.get("object_id", 0)
            
            if input_type == "video":
                timestamp = point.get("timestamp", 0.0)
                formatted_points.append(f"<point {timestamp:.2f}, {x:.3f}, {y:.3f}, {obj_id}>")
            elif input_type == "multi_image":
                img_idx = point.get("image_index", 0)
                formatted_points.append(f"<point {img_idx}, {x:.3f}, {y:.3f}, {obj_id}>")
            else:  # single image
                formatted_points.append(f"<point {x:.3f}, {y:.3f}, {obj_id}>")
        
        return " ".join(formatted_points)


class MultimodalCollator:
    """
    Collate function for batching multimodal samples.
    
    Implements:
    - Proper padding for variable-length sequences
    - Visual token interleaving
    - Timestamp and index token insertion
    - Task-based token weighting
    """
    
    def __init__(
        self,
        tokenizer,
        image_processor,
        max_seq_length: int = 4096,
    ):
        """
        Initialize collator.
        
        Args:
            tokenizer: Text tokenizer
            image_processor: Vision processor
            max_seq_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch: List[MultimodalSample]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multimodal samples.
        
        Returns:
            Dictionary with:
            - input_ids: Tokenized text
            - attention_mask: Attention mask
            - pixel_values: Processed visual inputs
            - visual_token_mask: Mask for visual token positions
            - timestamps: Frame timestamps (for videos)
            - image_indices: Image indices (for multi-image)
            - loss_weights: Per-token loss weights based on task
        """
        # Implementation will handle:
        # 1. Process visual inputs (images/videos)
        # 2. Tokenize text with special tokens
        # 3. Insert visual tokens, timestamps, indices
        # 4. Apply task-based token weighting
        # 5. Pad to max sequence length
        raise NotImplementedError("Collator to be implemented")
