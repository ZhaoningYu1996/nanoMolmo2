# Molmo2 DataLoaders Documentation

**Educational implementation of Molmo2's data loading pipeline**

This directory contains the data loading infrastructure for nanoMolmo2, following the [Molmo2 technical report](https://molmo.allenai.org/)'s specifications. All components are designed to handle multimodal inputs (images and videos) with proper task weighting, efficient packing, and temporal grounding.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Dataset Implementations](#dataset-implementations)
- [Training Utilities](#training-utilities)
- [Usage Examples](#usage-examples)
- [Data Format Specifications](#data-format-specifications)

---

## Architecture Overview

The dataloader system implements Molmo2's three-stage training pipeline:

1. **Pre-training Stage**: 60% captioning, 30% pointing, 10% NLP data
2. **Supervised Fine-tuning (SFT)**: Mixed multimodal datasets with sqrt-proportional sampling
3. **Long-context SFT**: Extended sequences (36,864 tokens) for long videos (384 frames)

### Key Design Principles

- **Unified multimodal handling**: Single data structure for images, videos, and multi-image inputs
- **Task-specific weighting**: Higher weights for challenging tasks (pointing/tracking: 2.0x)
- **Efficient packing**: Merge short examples to maximize GPU utilization
- **Temporal grounding**: HTML-like format for points in space and time

---

## Core Components

### 1. `base.py` - Foundation Classes

#### `MultimodalSample` (Dataclass)

The fundamental unit of data. Every sample contains:

```python
@dataclass
class MultimodalSample:
    visual_inputs: List[Image.Image]      # PIL Images (frames for video)
    text: str                             # Text input/output
    input_type: str                       # "image", "video", or "multi_image"
    timestamps: Optional[List[float]]     # Frame timestamps (videos only)
    image_indices: Optional[List[int]]    # Image indices (multi-image only)
    subtitles: Optional[str]              # Video subtitles as text
    points: Optional[List[Dict]]          # Ground-truth coordinates
    task_weight: float                    # Token weighting factor (default: 1.0)
```

**Example**: Image pointing sample
```python
sample = MultimodalSample(
    visual_inputs=[pil_image],
    text="Q: Point to all cats\nA: <point 0.5, 0.3, 0> <point 0.7, 0.6, 1>",
    input_type="image",
    points=[{"x": 0.5, "y": 0.3, "object_id": 0}, ...],
    task_weight=2.0  # Higher weight for pointing
)
```

#### `MultimodalDataset` (Abstract Base)

Base class for all datasets. Provides:

- **`format_pointing_output()`**: Converts point dicts to HTML-like strings
  - Images: `<point x, y, object_id>`
  - Videos: `<point timestamp, x, y, object_id>`
  - Multi-image: `<point image_index, x, y, object_id>`

- **`sample_video_frames()`**: Extract frames at target FPS with max frame limit
  - Standard: 2 fps, max 128 frames
  - Long-context: 2 fps, max 384 frames

**Subclasses must implement**:
- `_load_data()`: Load from JSONL files
- `__getitem__()`: Return `MultimodalSample`

#### `MultimodalCollator` (Batch Processing)

Converts `MultimodalSample` list into model-ready batches:

**Inputs**: `List[MultimodalSample]`

**Outputs**: Dictionary with
- `input_ids`: Tokenized text with visual placeholders
- `attention_mask`: Attention mask
- `pixel_values`: Processed image/video frames
- `visual_token_mask`: Marks visual token positions
- `timestamps`: Frame timestamps (videos)
- `image_indices`: Image indices (multi-image)
- `loss_weights`: Per-token weights based on task

**Process**:
1. Process visual inputs through image processor (ViT encoding)
2. Tokenize text with special tokens (`<image>`, `<timestamp>`, etc.)
3. Insert visual tokens at placeholder positions
4. Apply task-based token weighting
5. Pad sequences to max length

---

## Dataset Implementations

### Image Datasets (`image_datasets.py`)

#### 1. `CaptioningDataset`
**Purpose**: Dense image captioning (PixMo style)  
**Training stage**: Pre-training (60% of mixture)  
**Task weight**: 1.0x

**Data format** (JSONL):
```json
{"image_path": "images/cat.jpg", "caption": "A fluffy orange cat sleeping on a couch..."}
```

**Output**: Caption text only (no Q&A format)

#### 2. `PointingDataset`
**Purpose**: Spatial pointing with coordinates  
**Training stage**: Pre-training (30% of mixture)  
**Task weight**: 2.0x (higher priority)

**Data format** (JSONL):
```json
{
  "image_path": "images/scene.jpg",
  "query": "Point to all people",
  "points": [
    {"x": 0.3, "y": 0.5, "object_id": 0},
    {"x": 0.7, "y": 0.6, "object_id": 1}
  ]
}
```

**Output format**:
```
Q: Point to all people
A: <point 0.3, 0.5, 0> <point 0.7, 0.6, 1>
```

#### 3. `VQADataset`
**Purpose**: Visual Question Answering  
**Training stage**: SFT  
**Task weight**: 1.0x

**Data format** (JSONL):
```json
{
  "image_path": "images/car.jpg",
  "question": "What color is the car?",
  "answer": "Red"
}
```

**Output format**:
```
Q: What color is the car?
A: Red
```

#### 4. `CountingDataset`
**Purpose**: Object counting (PixMo-Count style)  
**Training stage**: SFT  
**Task weight**: 1.5x (moderate priority)

**Data format** (JSONL):
```json
{
  "image_path": "images/apples.jpg",
  "query": "How many apples?",
  "count": 5,
  "points": [...]  // Optional point annotations
}
```

**Output**: Count number, optionally followed by point coordinates

---

### Video Datasets (`video_datasets.py`)

All video datasets share common features:
- **Frame sampling**: 2 fps (configurable)
- **Max frames**: 128 (standard) or 384 (long-context)
- **Timestamp tracking**: Frame timestamps in seconds
- **Subtitle support**: Optional subtitle text

#### 1. `VideoCaptioningDataset`
**Purpose**: Dense video captioning with frame-level details  
**Task weight**: 1.0x

**Data creation pipeline** (from paper):
1. Human narration of video
2. Speech-to-text transcription
3. LLM enrichment with frame-level visual details

**Data format** (JSONL):
```json
{
  "video_path": "videos/cooking.mp4",
  "caption": "At 0:00, a chef enters the kitchen. At 0:05, they pick up a knife...",
  "subtitles": "Today we're making pasta..."
}
```

#### 2. `VideoPointingDataset`
**Purpose**: Temporal pointing (space + time)  
**Task weight**: 2.0x (novel contribution)

**Innovation**: Extends 2D pointing to 4D (x, y, time, object_id)

**Data format** (JSONL):
```json
{
  "video_path": "videos/sports.mp4",
  "query": "Point to the person throwing the ball",
  "points": [
    {"x": 0.5, "y": 0.3, "timestamp": 1.2, "object_id": 0},
    {"x": 0.52, "y": 0.31, "timestamp": 1.4, "object_id": 0}
  ]
}
```

**Output format**:
```
Q: Point to the person throwing the ball
A: <point 1.2, 0.5, 0.3, 0> <point 1.4, 0.52, 0.31, 0>
```

#### 3. `VideoTrackingDataset`
**Purpose**: Object tracking across frames  
**Task weight**: 2.0x (novel contribution)

**Performance**: Molmo2-8B achieves 56.2 J&F vs Gemini 3 Pro's 41.1

**Data format** (JSONL):
```json
{
  "video_path": "videos/traffic.mp4",
  "query": "Track the red car",
  "tracks": [
    {"x": 0.5, "y": 0.3, "timestamp": 0.0, "object_id": 0},
    {"x": 0.51, "y": 0.31, "timestamp": 0.5, "object_id": 0},
    {"x": 0.53, "y": 0.33, "timestamp": 1.0, "object_id": 0}
  ]
}
```

**Key difference from pointing**: Same `object_id` across all points (temporal consistency)

#### 4. `VideoQADataset`
**Purpose**: Long-form video question answering  
**Task weight**: 1.0x

**Data creation**: Human-LLM collaboration (no proprietary model distillation)

**Data format** (JSONL):
```json
{
  "video_path": "videos/documentary.mp4",
  "question": "What is the main topic discussed?",
  "answer": "The video discusses climate change impacts...",
  "subtitles": "Scientists have observed..."
}
```

---

## Training Utilities

### 1. `SequencePacker` (`utils.py`)

**Problem**: Short examples waste GPU memory (padding overhead)  
**Solution**: Pack multiple short sequences into single long sequence

**Algorithm**:
1. Sort samples by length (shortest first)
2. Greedily pack samples together
3. Insert separator tokens between packed samples
4. Track pack boundaries to prevent cross-sample attention

**Example**:
```python
# Before packing: 3 samples with lengths [100, 150, 200]
# Max sequence length: 512

# After packing:
# Pack 1: Sample1 (100) + SEP (1) + Sample2 (150) = 251 tokens
# Pack 2: Sample3 (200) = 200 tokens

packer = SequencePacker(max_seq_length=4096)
packed = packer.pack_sequences(samples, sep_token_id=tokenizer.sep_token_id)
```

**Benefits**:
- ~30% improvement in training throughput
- Reduced padding overhead
- Better GPU utilization

### 2. `MessageTreeEncoder` (`utils.py`)

**Problem**: Videos with multiple Q&A pairs need efficient encoding  
**Solution**: Share visual encoding across all annotations

**Example**:
```python
# Video with 3 Q&A pairs
annotations = [
    {"question": "What color is the car?", "answer": "Red"},
    {"question": "Where is it parked?", "answer": "In a garage"},
    {"question": "How many doors?", "answer": "Four"}
]

encoder = MessageTreeEncoder()
samples = encoder.encode_multi_annotation_video(
    video_frames=frames,
    annotations=annotations,
    tokenizer=tokenizer
)
# Returns 3 samples with shared visual encoding
```

**Benefits**:
- Visual features computed once
- Memory efficient for multi-annotation videos
- Faster training on video datasets

### 3. `TokenWeightingStrategy` (`utils.py`)

**Purpose**: Balance learning across diverse tasks

**Task weights** (from paper):
```python
TASK_WEIGHTS = {
    "video_caption": 1.0,
    "image_caption": 1.0,
    "video_pointing": 2.0,    # Higher priority
    "image_pointing": 2.0,
    "video_tracking": 2.0,     # Higher priority
    "video_qa": 1.0,
    "image_qa": 1.0,
    "counting": 1.5,           # Moderate priority
}
```

**Usage in loss computation**:
```python
# Unweighted loss: [batch_size, seq_length]
loss = criterion(predictions, targets)

# Apply per-token weights
weighted_loss = TokenWeightingStrategy.apply_token_weights(
    loss=loss,
    loss_weights=batch["loss_weights"]
)
```

**Why weights matter**:
- Pointing/tracking are harder tasks → need more signal
- Prevents model from ignoring difficult tasks
- Balances learning across diverse data

### 4. `DataMixingConfig` (`utils.py`)

**Purpose**: Control dataset mixing ratios per training stage

**Pre-training mixture** (fixed ratios):
```python
config = DataMixingConfig.get_pretrain_config()
# Returns:
# {
#   "dense_captioning": 0.60,  # 60%
#   "image_pointing": 0.30,    # 30%
#   "nlp_data": 0.10           # 10%
# }
```

**SFT mixture** (sqrt-proportional sampling):
```python
dataset_sizes = {
    "vqa": 100000,
    "video_qa": 50000,
    "pointing": 30000
}
config = DataMixingConfig.get_sft_config(dataset_sizes)
# Computes weights proportional to sqrt(size)
# Larger datasets don't dominate, smaller datasets get fair representation
```

**Math behind sqrt-proportional**:
```
Weight_i = sqrt(Size_i) / sum(sqrt(Size_j) for all j)
```

**Benefits**:
- Prevents large datasets from dominating
- Smaller high-quality datasets get sufficient sampling
- Manual rebalancing possible based on validation performance

---

## Usage Examples

### Basic Dataset Loading

```python
from data.dataloaders import CaptioningDataset, MultimodalCollator
from transformers import AutoTokenizer
from torchvision import transforms

# Initialize tokenizer and image processor
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
image_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset
dataset = CaptioningDataset(
    data_path="data/train.jsonl",
    split="train",
    max_seq_length=4096
)

# Create collator
collator = MultimodalCollator(
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_seq_length=4096
)

# Create DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collator,
    shuffle=True
)

# Iterate
for batch in dataloader:
    # batch contains:
    # - input_ids: [batch_size, seq_length]
    # - attention_mask: [batch_size, seq_length]
    # - pixel_values: [batch_size, num_images, 3, H, W]
    # - loss_weights: [batch_size, seq_length]
    pass
```

### Multi-Dataset Training with Mixing

```python
from data.dataloaders import (
    CaptioningDataset,
    PointingDataset,
    DataMixingConfig
)
from torch.utils.data import ConcatDataset, WeightedRandomSampler

# Load datasets
datasets = {
    "captioning": CaptioningDataset("data/caption.jsonl"),
    "pointing": PointingDataset("data/pointing.jsonl"),
}

# Get mixing weights (pre-training stage)
mixing_config = DataMixingConfig.get_pretrain_config()

# Create weighted sampler
weights = []
for name, dataset in datasets.items():
    weight = mixing_config.dataset_weights.get(name, 1.0)
    weights.extend([weight] * len(dataset))

combined_dataset = ConcatDataset(list(datasets.values()))
sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(combined_dataset),
    replacement=True
)

dataloader = DataLoader(
    combined_dataset,
    batch_size=4,
    sampler=sampler,
    collate_fn=collator
)
```

### Video Dataset with Frame Sampling

```python
from data.dataloaders import VideoQADataset

# Standard training: 2 fps, max 128 frames
dataset = VideoQADataset(
    data_path="data/video_qa.jsonl",
    max_frames=128,
    fps=2.0,
    max_seq_length=4096
)

# Long-context training: 2 fps, max 384 frames
long_dataset = VideoQADataset(
    data_path="data/video_qa.jsonl",
    max_frames=384,
    fps=2.0,
    max_seq_length=36864  # Extended for long-context stage
)

# Get a sample
sample = dataset[0]
print(f"Video frames: {len(sample.visual_inputs)}")
print(f"Timestamps: {sample.timestamps}")
print(f"Subtitles: {sample.subtitles}")
print(f"Q&A: {sample.text}")
```

### Sequence Packing

```python
from data.dataloaders.utils import SequencePacker

# Create packer
packer = SequencePacker(max_seq_length=4096)

# Get batch of tokenized samples
batch = [collator([sample]) for sample in dataset[:10]]

# Pack sequences
packed_batch = packer.pack_sequences(
    samples=batch,
    sep_token_id=tokenizer.sep_token_id
)

# Each packed sample contains:
# - input_ids: Combined sequence
# - pack_boundaries: [(start1, end1), (start2, end2), ...]
# - loss_weights: Weights for each token
```

---

## Data Format Specifications

### JSONL File Structure

All datasets use **JSONL** (JSON Lines) format:
- One JSON object per line
- Each line is a complete, valid JSON object
- Newline-separated for streaming and parallel processing

**Example file** (`train.jsonl`):
```jsonl
{"image_path": "img1.jpg", "caption": "A cat sleeping"}
{"image_path": "img2.jpg", "caption": "A dog running"}
{"image_path": "img3.jpg", "caption": "A bird flying"}
```

### Coordinate Format

All spatial coordinates are **normalized** [0, 1]:
- `x`: Horizontal position (0 = left edge, 1 = right edge)
- `y`: Vertical position (0 = top edge, 1 = bottom edge)

**Example**:
```json
{"x": 0.5, "y": 0.3}  // Center horizontally, 30% from top
```

### Timestamp Format

Video timestamps are in **seconds** (float):
- `timestamp: 0.0` = First frame
- `timestamp: 1.5` = 1.5 seconds into video

**Example**:
```json
{"timestamp": 1.5, "x": 0.5, "y": 0.3}
```

### HTML-like Pointing Format

Output format for model predictions:

**Single image**:
```
<point x, y, object_id>
```

**Video** (temporal):
```
<point timestamp, x, y, object_id>
```

**Multi-image**:
```
<point image_index, x, y, object_id>
```

**Multiple points**:
```
<point 0.5, 0.3, 0> <point 0.7, 0.6, 1> <point 0.2, 0.8, 2>
```

---

## Training Stage Configurations

### Stage 1: Pre-training

**Data mixture**:
- 60% Dense captioning (images)
- 30% Image pointing (PixMo-Points, PixMo-Count)
- 10% NLP data (Tulu)

**Sequence length**: 4,096 tokens  
**Frame limit**: 128 frames (videos)  
**Training steps**: ~100K

**Datasets to use**:
- `CaptioningDataset`
- `PointingDataset`
- NLP dataset (from Tulu)

### Stage 2: Supervised Fine-tuning (SFT)

**Data mixture**: Sqrt-proportional sampling across:
- PixMo datasets
- Molmo2 video datasets (captioning, pointing, tracking, QA)
- Tulu NLP
- Open-source VQA datasets

**Sequence length**: 4,096 tokens  
**Frame limit**: 128 frames (videos)  
**Training steps**: ~50K

**Datasets to use**:
- `CaptioningDataset`
- `PointingDataset`
- `VQADataset`
- `CountingDataset`
- `VideoCaptioningDataset`
- `VideoPointingDataset`
- `VideoTrackingDataset`
- `VideoQADataset`

### Stage 3: Long-context SFT

**Data mixture**: Same as SFT stage  
**Sequence length**: 36,864 tokens  
**Frame limit**: 384 frames (videos)  
**Training steps**: 2K  
**Special requirements**: Context Parallelism (CP), Ulysses attention

**Configuration changes**:
```python
# Use max_frames=384 for all video datasets
dataset = VideoQADataset(
    data_path="data/video_qa.jsonl",
    max_frames=384,
    fps=2.0,
    max_seq_length=36864
)
```

---

## Performance Considerations

### Memory Optimization

**Video frame limit**: 128 frames (standard) keeps memory manageable
```
Memory per video ≈ 128 frames × 3 channels × 224 × 224 × 4 bytes
                 ≈ 77 MB (uncompressed)
```

**Sequence packing**: ~30% throughput improvement by reducing padding

**Efficient frame sampling**: 2 fps captures sufficient temporal information while keeping token count low

### Training Throughput

**Batch size guidelines**:
- A100 80GB: Batch size 4-8 (standard training)
- A100 80GB: Batch size 1-2 (long-context training)

**Gradient accumulation**: Use to simulate larger batches
```python
effective_batch_size = batch_size × num_gpus × grad_accum_steps
# Example: 4 × 2 GPUs × 4 accum = 32 effective batch size
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce `max_frames` (384 → 128 → 64)
2. Reduce `max_seq_length` (4096 → 2048)
3. Use gradient checkpointing
4. Reduce batch size
5. Enable sequence packing

### Issue: Slow data loading

**Solutions**:
1. Use `num_workers > 0` in DataLoader
2. Pre-process videos to lower resolution
3. Cache processed frames to disk
4. Use faster video codecs (H.264 instead of H.265)

### Issue: Imbalanced task learning

**Solutions**:
1. Adjust task weights in `TokenWeightingStrategy.TASK_WEIGHTS`
2. Modify data mixing ratios in `DataMixingConfig`
3. Monitor per-task validation losses
4. Use curriculum learning (start with easier tasks)

---

## Future Enhancements

### Planned Features

- [ ] Implement `MultimodalCollator.__call__()` (visual processing + tokenization)
- [ ] Add support for multi-crop images (column tokens)
- [ ] Implement bi-directional attention masking for visual tokens
- [ ] Add data augmentation (random crops, color jitter for images)
- [ ] Implement caching for processed video frames
- [ ] Add validation dataset class with metrics computation

### Research Extensions

- [ ] Test SlowFast encoding for efficient video processing
- [ ] Explore test-time scaling strategies
- [ ] Implement adaptive frame sampling based on video complexity
- [ ] Add support for audio-visual multimodal learning

---

## References

- [Molmo2 Technical Report](https://molmo.allenai.org/)
- [PixMo Datasets](https://github.com/allenai/pixmo)
- [Tulu NLP Data](https://github.com/allenai/tulu)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

---

**Last Updated**: 2026-01-14  
**Version**: 1.0.0  
**Contributors**: Zhaoning Yu & Claude Sonnet 4.5
