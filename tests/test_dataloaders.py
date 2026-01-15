"""
Tests for dataloader implementations.

Tests image and video datasets to ensure they:
- Load data correctly from JSONL files
- Return proper MultimodalSample objects
- Handle video frame sampling
- Format pointing outputs correctly
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from data.dataloaders.image_datasets import (
    CaptioningDataset,
    PointingDataset,
    VQADataset,
    CountingDataset,
)
from data.dataloaders.video_datasets import (
    VideoCaptioningDataset,
    VideoPointingDataset,
    VideoTrackingDataset,
    VideoQADataset,
)
from data.dataloaders.base import MultimodalSample


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    img_path = Path(temp_dir) / "test_image.jpg"
    img = Image.new('RGB', (224, 224), color='red')
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample test video."""
    video_path = Path(temp_dir) / "test_video.mp4"
    
    # Create a short video with 30 frames at 10 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (224, 224))
    
    for i in range(30):
        # Create frames with changing colors
        frame = np.full((224, 224, 3), (i * 8, 128, 255 - i * 8), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    return str(video_path)


class TestImageDatasets:
    """Test suite for image-based datasets."""
    
    def test_captioning_dataset(self, temp_dir, sample_image):
        """Test CaptioningDataset loads and returns correct format."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "image_path": sample_image,
                "caption": "A red test image"
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = CaptioningDataset(
            data_path=str(temp_dir),
            split="train",
            max_seq_length=512
        )
        
        # Test dataset properties
        assert len(dataset) == 1
        assert dataset.task_weight == 1.0
        
        # Test sample retrieval
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert len(sample.visual_inputs) == 1
        assert isinstance(sample.visual_inputs[0], Image.Image)
        assert sample.text == "A red test image"
        assert sample.input_type == "image"
        assert sample.task_weight == 1.0
    
    def test_pointing_dataset(self, temp_dir, sample_image):
        """Test PointingDataset formats points correctly."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "image_path": sample_image,
                "query": "Point to all cats",
                "points": [
                    {"x": 0.5, "y": 0.3, "object_id": 0},
                    {"x": 0.7, "y": 0.6, "object_id": 1}
                ]
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = PointingDataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        # Test sample
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert sample.input_type == "image"
        assert "Q: Point to all cats" in sample.text
        assert "<point 0.500, 0.300, 0>" in sample.text
        assert "<point 0.700, 0.600, 1>" in sample.text
        assert len(sample.points) == 2
        assert sample.task_weight == 2.0  # Higher weight for pointing
    
    def test_vqa_dataset(self, temp_dir, sample_image):
        """Test VQADataset formats Q&A correctly."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "image_path": sample_image,
                "question": "What color is the image?",
                "answer": "Red"
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = VQADataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        # Test sample
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert "Q: What color is the image?" in sample.text
        assert "A: Red" in sample.text
        assert sample.input_type == "image"
    
    def test_counting_dataset(self, temp_dir, sample_image):
        """Test CountingDataset with and without points."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            # Sample with points
            sample_data = {
                "image_path": sample_image,
                "query": "How many apples?",
                "count": 3,
                "points": [
                    {"x": 0.2, "y": 0.3, "object_id": 0},
                    {"x": 0.5, "y": 0.5, "object_id": 1},
                    {"x": 0.8, "y": 0.7, "object_id": 2}
                ]
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = CountingDataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        # Test sample with points
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert "Q: How many apples?" in sample.text
        assert "A: 3" in sample.text
        assert "<point" in sample.text  # Points should be included
        assert len(sample.points) == 3
        assert sample.task_weight == 1.5  # Moderate weight for counting


class TestVideoDatasets:
    """Test suite for video-based datasets."""
    
    def test_video_captioning_dataset(self, temp_dir, sample_video):
        """Test VideoCaptioningDataset loads video and samples frames."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "caption": "A video with changing colors",
                "subtitles": "Optional subtitle text"
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = VideoCaptioningDataset(
            data_path=str(temp_dir),
            split="train",
            max_frames=10,
            fps=2.0
        )
        
        # Test dataset properties
        assert len(dataset) == 1
        assert dataset.max_frames == 10
        assert dataset.fps == 2.0
        
        # Test sample retrieval
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert sample.input_type == "video"
        assert len(sample.visual_inputs) > 0
        assert len(sample.visual_inputs) <= 10  # Respects max_frames
        assert len(sample.timestamps) == len(sample.visual_inputs)
        assert sample.text == "A video with changing colors"
        assert sample.subtitles == "Optional subtitle text"
        
        # Verify timestamps are increasing
        assert all(sample.timestamps[i] < sample.timestamps[i+1] 
                  for i in range(len(sample.timestamps)-1))
    
    def test_video_frame_sampling(self, temp_dir, sample_video):
        """Test video frame sampling respects FPS and max_frames."""
        # Create test data file for dataset initialization
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "caption": "Test video"
            }
            f.write(json.dumps(sample_data) + '\n')
        
        dataset = VideoCaptioningDataset(
            data_path=temp_dir,
            split="train",
            max_frames=5,
            fps=1.0
        )
        
        # Sample frames
        frames, timestamps = dataset.sample_video_frames(sample_video)
        
        # Verify frame count respects max_frames
        assert len(frames) <= 5
        assert len(timestamps) == len(frames)
        
        # Verify all frames are PIL Images
        assert all(isinstance(frame, Image.Image) for frame in frames)
        
        # Verify timestamps are at expected intervals (roughly 1 second apart)
        if len(timestamps) > 1:
            intervals = [timestamps[i+1] - timestamps[i] 
                        for i in range(len(timestamps)-1)]
            # Should be roughly 1 second apart (allow some tolerance)
            assert all(0.8 <= interval <= 1.2 for interval in intervals)
    
    def test_video_pointing_dataset(self, temp_dir, sample_video):
        """Test VideoPointingDataset formats temporal points correctly."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "query": "Point to the person",
                "points": [
                    {"x": 0.5, "y": 0.3, "timestamp": 0.5, "object_id": 0},
                    {"x": 0.52, "y": 0.31, "timestamp": 1.0, "object_id": 0},
                    {"x": 0.54, "y": 0.32, "timestamp": 1.5, "object_id": 0}
                ]
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = VideoPointingDataset(
            data_path=str(temp_dir),
            split="train",
            max_frames=10,
            fps=2.0
        )
        
        # Test sample
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert sample.input_type == "video"
        assert "Q: Point to the person" in sample.text
        # Check temporal point format: <point timestamp, x, y, object_id>
        assert "<point 0.50, 0.500, 0.300, 0>" in sample.text
        assert "<point 1.00, 0.520, 0.310, 0>" in sample.text
        assert len(sample.points) == 3
        assert sample.task_weight == 2.0  # Higher weight for pointing
    
    def test_video_tracking_dataset(self, temp_dir, sample_video):
        """Test VideoTrackingDataset for object tracking."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "query": "Track the red car",
                "tracks": [
                    {"x": 0.3, "y": 0.4, "timestamp": 0.0, "object_id": 0},
                    {"x": 0.35, "y": 0.45, "timestamp": 0.5, "object_id": 0},
                    {"x": 0.4, "y": 0.5, "timestamp": 1.0, "object_id": 0}
                ]
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = VideoTrackingDataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        # Test sample
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert sample.input_type == "video"
        assert "Q: Track the red car" in sample.text
        # All tracks should have same object_id
        assert all(point["object_id"] == 0 for point in sample.points)
        assert sample.task_weight == 2.0  # Higher weight for tracking
    
    def test_video_qa_dataset(self, temp_dir, sample_video):
        """Test VideoQADataset for long-form QA."""
        # Create test data
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "question": "What happens in the video?",
                "answer": "The colors change gradually from blue to orange.",
                "subtitles": "No audio"
            }
            f.write(json.dumps(sample_data) + '\n')
        
        # Initialize dataset
        dataset = VideoQADataset(
            data_path=str(temp_dir),
            split="train",
            max_frames=128
        )
        
        # Test sample
        sample = dataset[0]
        assert isinstance(sample, MultimodalSample)
        assert sample.input_type == "video"
        assert "Q: What happens in the video?" in sample.text
        assert "A: The colors change gradually" in sample.text
        assert sample.subtitles == "No audio"


class TestPointingFormatting:
    """Test suite for pointing output formatting."""
    
    def test_image_point_format(self, temp_dir, sample_image):
        """Test single image point formatting."""
        # Create minimal test data file for dataset initialization
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "image_path": sample_image,
                "query": "test",
                "points": []
            }
            f.write(json.dumps(sample_data) + '\n')
        
        dataset = PointingDataset(data_path=temp_dir, split="train")
        
        points = [
            {"x": 0.123, "y": 0.456, "object_id": 0},
            {"x": 0.789, "y": 0.012, "object_id": 1}
        ]
        
        formatted = dataset.format_pointing_output(points, "image")
        
        # Check format: <point x, y, object_id>
        assert "<point 0.123, 0.456, 0>" in formatted
        assert "<point 0.789, 0.012, 1>" in formatted
    
    def test_video_point_format(self, temp_dir, sample_video):
        """Test video point formatting with timestamps."""
        # Create minimal test data file for dataset initialization
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "query": "test",
                "points": []
            }
            f.write(json.dumps(sample_data) + '\n')
        
        dataset = VideoPointingDataset(data_path=temp_dir, split="train")
        
        points = [
            {"x": 0.5, "y": 0.5, "timestamp": 1.23, "object_id": 0},
            {"x": 0.6, "y": 0.6, "timestamp": 2.45, "object_id": 0}
        ]
        
        formatted = dataset.format_pointing_output(points, "video")
        
        # Check format: <point timestamp, x, y, object_id>
        assert "<point 1.23, 0.500, 0.500, 0>" in formatted
        assert "<point 2.45, 0.600, 0.600, 0>" in formatted


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self, temp_dir):
        """Test dataset with no samples."""
        data_file = Path(temp_dir) / "train.jsonl"
        data_file.touch()  # Create empty file
        
        dataset = CaptioningDataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        assert len(dataset) == 0
    
    def test_missing_optional_fields(self, temp_dir, sample_video):
        """Test video dataset without optional subtitles."""
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "video_path": sample_video,
                "caption": "Test caption"
                # No subtitles field
            }
            f.write(json.dumps(sample_data) + '\n')
        
        dataset = VideoCaptioningDataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        sample = dataset[0]
        assert sample.subtitles is None
    
    def test_counting_without_points(self, temp_dir, sample_image):
        """Test CountingDataset when points are not provided."""
        data_file = Path(temp_dir) / "train.jsonl"
        with open(data_file, 'w') as f:
            sample_data = {
                "image_path": sample_image,
                "query": "How many objects?",
                "count": 5
                # No points field
            }
            f.write(json.dumps(sample_data) + '\n')
        
        dataset = CountingDataset(
            data_path=str(temp_dir),
            split="train"
        )
        
        sample = dataset[0]
        assert "A: 5" in sample.text
        assert "<point" not in sample.text  # No points in output
        assert sample.points is None
