"""
Molmo2 Data Loaders

Implements data loading following the Molmo2 paper's specifications:
- Multi-stage training pipeline support
- Image and video data handling
- Efficient sequence packing
- Message-tree encoding
- Token weighting for task balancing
"""

from .base import MultimodalDataset, MultimodalCollator
from .image_datasets import (
    CaptioningDataset,
    PointingDataset,
    VQADataset,
)
from .video_datasets import (
    VideoCaptioningDataset,
    VideoPointingDataset,
    VideoTrackingDataset,
    VideoQADataset,
)
from .utils import SequencePacker, MessageTreeEncoder

__all__ = [
    "MultimodalDataset",
    "MultimodalCollator",
    "CaptioningDataset",
    "PointingDataset",
    "VQADataset",
    "VideoCaptioningDataset",
    "VideoPointingDataset",
    "VideoTrackingDataset",
    "VideoQADataset",
    "SequencePacker",
    "MessageTreeEncoder",
]
