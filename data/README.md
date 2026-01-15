# Molmo2 Data Pipeline

Complete data preparation and loading system for training nanoMolmo2.

## Quick Start

### 1. Download Datasets

Download all Molmo2 training datasets:

```bash
# Download all datasets (pre-training + SFT)
python scripts/download_molmo2_data.py --stage all

# Or download specific stages
python scripts/download_molmo2_data.py --stage pretraining
python scripts/download_molmo2_data.py --stage sft

# Download with evaluation datasets
python scripts/download_molmo2_data.py --stage all --include-eval
```

**Storage Requirements:**
- Pre-training datasets: ~100GB
- SFT datasets: ~50GB
- Evaluation datasets: ~10GB

### 2. Inspect Downloaded Data

Examine dataset structure before writing loaders:

```bash
# List all downloaded datasets
python scripts/inspect_molmo2_data.py

# Inspect specific dataset
python scripts/inspect_molmo2_data.py pixmo-cap
python scripts/inspect_molmo2_data.py molmo2-videocapqa

# Show more samples
python scripts/inspect_molmo2_data.py pixmo-cap --num-samples 10
```

### 3. Use Dataloaders

Once datasets are downloaded, use the dataloaders for training:

```python
from data.loaders import get_dataloader

# Pre-training stage
train_loader = get_dataloader(
    stage="pretraining",
    batch_size=32,
    num_workers=4
)

# SFT stage
sft_loader = get_dataloader(
    stage="sft",
    batch_size=16,
    num_workers=4
)
```

## Directory Structure

After download, data is organized as:

```
data/
├── molmo2_datasets/          # Downloaded datasets
│   ├── pixmo-cap/           # Dense image captioning
│   │   ├── train.parquet
│   │   └── validation.parquet
│   ├── pixmo-points/        # Image pointing
│   ├── molmo2-videocapqa/   # Video QA
│   └── ...
├── dataloaders/             # Dataset implementations
│   ├── base.py             # Base classes
│   ├── image_datasets.py   # Image dataset loaders
│   ├── video_datasets.py   # Video dataset loaders
│   └── utils.py            # Utilities
└── loaders.py              # Main dataloader interface
```

## Training Stages

### Pre-training Stage

**Goal:** Learn general visual understanding

**Datasets:**
- `pixmo-cap` (60%): Dense image captions (~200 words)
- `pixmo-points` (25%): Image pointing with referring expressions
- `pixmo-count` (5%): Object counting QA
- `pixmo-docs` (5%): Chart/table understanding
- `pixmo-capqa` (5%): Synthetic QA from captions

**Configuration:**
- Sequence length: 4096 tokens
- Frames per video: N/A (image only)
- Batch size: 256 (distributed)

### SFT Stage

**Goal:** Instruction following and video understanding

**Datasets:**
- `molmo2-cap`: Image dense captioning
- `molmo2-videocapqa`: Video captioning and QA
- `molmo2-videosubtitleqa`: Video QA with subtitles
- `molmo2-askmodelanything`: Human-authored image QA
- `molmo2-videopoint`: Video temporal pointing (novel)
- `molmo2-videotrack`: Video object tracking (novel)
- `molmo2-multiimageqa`: Multi-image QA
- `molmo2-synmultiimageqa`: Synthetic multi-image QA
- `molmo2-multiimagepoint`: Multi-image pointing

**Configuration:**
- Sequence length: 4096 tokens
- Frames per video: F=128 (standard), F=384 (long-context)
- Frame rate: S=2 fps
- Batch size: 128 (distributed)

### Long-context Stage

**Goal:** Handle extended video sequences

**Configuration:**
- Sequence length: Up to 16K tokens
- Frames per video: F=384
- Frame rate: S=2 fps
- Uses same datasets as SFT with longer sequences

## Data Formats

### Image Datasets (PixMo)

**pixmo-cap format:**
```json
{
  "image": <PIL.Image>,
  "caption": "Detailed dense caption describing the image...",
  "metadata": {...}
}
```

**pixmo-points format:**
```json
{
  "image": <PIL.Image>,
  "query": "Point to all cats in the image",
  "points": [
    {"x": 0.5, "y": 0.3, "object_id": 0},
    {"x": 0.7, "y": 0.6, "object_id": 1}
  ]
}
```

### Video Datasets (Molmo2)

**molmo2-videocapqa format:**
```json
{
  "video_path": "path/to/video.mp4",
  "question": "What happens in the video?",
  "answer": "Detailed answer...",
  "subtitles": "Optional subtitle text",
  "metadata": {...}
}
```

**molmo2-videopoint format:**
```json
{
  "video_path": "path/to/video.mp4",
  "query": "Point to the person throwing the ball",
  "points": [
    {"x": 0.5, "y": 0.3, "timestamp": 1.2, "object_id": 0},
    {"x": 0.52, "y": 0.31, "timestamp": 1.4, "object_id": 0}
  ]
}
```

**molmo2-videotrack format:**
```json
{
  "video_path": "path/to/video.mp4",
  "query": "Track the red car",
  "tracks": [
    {"x": 0.3, "y": 0.4, "timestamp": 0.0, "object_id": 0},
    {"x": 0.35, "y": 0.45, "timestamp": 0.5, "object_id": 0}
  ]
}
```

## Implementation Status

- ✅ Data download script (`scripts/download_molmo2_data.py`)
- ✅ Data inspection tool (`scripts/inspect_molmo2_data.py`)
- 🚧 Dataset implementations (to be rewritten after inspecting actual data)
- 🚧 Main dataloader interface (`data/loaders.py`)
- 🚧 Data preprocessing utilities
- 🚧 Integration tests with real data

## Next Steps

1. **Download a sample dataset** to understand format:
   ```bash
   python scripts/download_molmo2_data.py --stage pretraining --datasets pixmo-cap
   ```

2. **Inspect the data structure**:
   ```bash
   python scripts/inspect_molmo2_data.py pixmo-cap
   ```

3. **Rewrite dataloaders** based on actual data format

4. **Write integration tests** using downloaded datasets

5. **Implement data preprocessing** (resizing, normalization, etc.)

## References

- [Molmo2 Technical Report](https://allenai.org/blog/molmo2)
- [PixMo Data](https://allenai.org/open-data)
- [Molmo2 Data Collection](https://huggingface.co/collections/allenai/molmo2-data)
