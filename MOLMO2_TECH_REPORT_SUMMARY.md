# Molmo2 Technical Report Summary

**Source**: Allen Institute for AI (Ai2)  
**Release Date**: December 2025  
**Project**: State-of-the-art Open Multimodal Family for Video and Multi-Image Understanding

---

## Overview

Molmo2 is a family of open-source multimodal AI models designed for **video and multi-image understanding**. It represents a significant advancement in multimodal AI by:

1. **Achieving state-of-the-art performance** on video understanding benchmarks
2. **Introducing novel capabilities** like video pointing and tracking
3. **Releasing 9 new open datasets** with 9+ million multimodal examples
4. **Using human-LLM collaboration** without proprietary model distillation

---

## Key Innovations

### 1. Video Temporal Pointing (Novel)
- **Innovation**: Extends 2D image pointing to 4D (x, y, time, object_id)
- **Format**: `<point timestamp, x, y, object_id>`
- **Use case**: Point to objects at specific moments in videos
- **Dataset**: Molmo2-VideoPoint (330k examples)

### 2. Video Object Tracking (Novel)
- **Innovation**: Track objects across video frames using natural language
- **Performance**: 56.2 J&F vs Gemini 2.0 Pro's 41.1 on benchmarks
- **Dataset**: Molmo2-VideoTrack (220k examples)

### 3. Multi-Image Reasoning
- **Capability**: Reason across multiple images (charts, tables, documents)
- **Datasets**: 
  - Molmo2-MultiImageQA (45k examples)
  - Molmo2-SynMultiImageQA (188k examples)
  - Molmo2-MultiImagePoint (470k examples)

### 4. Dense Video Captioning
- **Creation Pipeline**:
  1. Human narration of video
  2. Speech-to-text transcription
  3. LLM enrichment with frame-level visual details
- **Dataset**: Molmo2-Cap (104k video-level, 431k clip-level)

---

## Architecture

### Base Model Variants
- **Molmo2-1B**: 1B parameters
- **Molmo2-7B**: 7B parameters  
- **Molmo2-8B**: 8B parameters (best performance)

### Components
1. **Vision Encoder**: Pre-trained ViT (Vision Transformer)
2. **Language Model**: OLMo (Ai2's open LLM)
3. **Multimodal Connector**: Projection layer between vision and language
4. **Special Tokens**: `<image>`, `<point>`, `<timestamp>`, etc.

### Token Format
- **Visual tokens**: Interleaved with text tokens
- **Timestamps**: Embedded as special tokens for temporal grounding
- **Points**: HTML-like compressed format for efficiency

---

## Training Pipeline (3 Stages)

### Stage 1: Pre-training
**Goal**: Learn general visual-language alignment

**Configuration**:
- **Steps**: ~100K training steps (~32K in actual training)
- **Sequence length**: 4,096 tokens
- **Frame limit**: 128 frames (videos)
- **Frame rate**: 2 fps

**Data Mixture** (Fixed Ratios):
- 60% Dense captioning (PixMo-Cap)
- 30% Image pointing (PixMo-Points, PixMo-Count)
- 10% NLP data (Tulu v2 SFT mixture)

**Datasets Used**:
1. `pixmo-cap` - Dense image captioning (~200 words avg)
2. `pixmo-points` - Image pointing with referring expressions
3. `pixmo-count` - Object counting QA
4. `cosyn-point` - Synthetic pointing data
5. `tulu-v2-sft-mixture` - Text-only instruction data

---

### Stage 2: Supervised Fine-Tuning (SFT)
**Goal**: Instruction following and multimodal task learning

**Configuration**:
- **Steps**: ~50K training steps
- **Sequence length**: 4,096 tokens
- **Frame limit**: 128 frames (videos)
- **Frame rate**: 2 fps

**Data Mixture** (Sqrt-proportional Sampling):
- Formula: `Weight_i = sqrt(Size_i) / sum(sqrt(Size_j))`
- Prevents large datasets from dominating
- Manual rebalancing based on validation performance

**Datasets Used** (100+ datasets total):

#### Molmo2 Original Datasets (9 new releases):
1. **Molmo2-Cap** (100k) - Video dense captioning
2. **Molmo2-AskModelAnything** (43k) - Human-authored video QA
3. **Molmo2-VideoCapQA** (1M) - Synthetic QA from video captions
4. **Molmo2-VideoSubtitleQA** (300k) - Video QA with subtitle context
5. **Molmo2-VideoPoint** (330k) - Video temporal pointing (NOVEL)
6. **Molmo2-VideoTrack** (220k) - Video object tracking (NOVEL)
7. **Molmo2-MultiImageQA** (45k) - Multi-image QA
8. **Molmo2-SynMultiImageQA** (188k) - Synthetic multi-image QA
9. **Molmo2-MultiImagePoint** (470k) - Multi-image pointing

#### PixMo Datasets:
- `pixmo-cap` (710k) - Dense image captioning
- `pixmo-ask-model-anything` (71k) - Human-authored image QA
- `pixmo-cap-qa` (190k) - Synthetic QA from captions
- `pixmo-clocks` (800k) - Clock reading

#### Academic Image Datasets:
- `llava-665k-multi` (2.5M) - LLaVA instruction-following
- `tallyqa` (250k) - Counting QA
- `vqa-v2` (440k) - Visual Question Answering v2
- `docvqa` (39k) - Document VQA
- `textvqa` (35k) - Text-based VQA
- `chartqa` (28k) - Chart understanding
- `st-vqa` (25k) - Scene text VQA
- `infographicvqa` (24k) - Infographic VQA
- `ai2d` (15k) - Diagram understanding
- `nlvr2` (86k) - Natural language visual reasoning
- `a-okvqa` (34k) - Knowledge-based VQA
- `ok-vqa` (9k) - Outside knowledge VQA
- `scienceqa` (6.2k) - Science QA

#### Video Understanding Datasets (Academic):
- `tgif` (63k) - Animated GIF QA
- `tvqa` (120k) - TV show QA
- `next-qa` (34k) - Video QA with reasoning
- `sports-qa` (56k) - Sports video QA
- `funqa` (200k) - Creative video understanding
- And 30+ more video datasets...

#### Tracking/Segmentation Datasets:
- `mevis` (20k) - Referring video object segmentation
- `refvos` (11k) - Referring video object segmentation
- `vicas` (130k) - Video caption and segmentation
- `trackingnet` (29k) - Large-scale object tracking
- And 10+ more tracking datasets...

#### Text Data:
- `tulu-v2-sft-mixture` (980k) - Text-only instruction data

---

### Stage 3: Long-Context SFT
**Goal**: Handle extended video sequences

**Configuration**:
- **Steps**: 2K training steps (short fine-tuning)
- **Sequence length**: 36,864 tokens (9x longer than SFT)
- **Frame limit**: 384 frames (3x more than SFT)
- **Frame rate**: 2 fps
- **Special requirements**: Context Parallelism (CP), Ulysses attention

**Data Mixture**: Same as SFT stage (same datasets, longer sequences)

**Technical Details**:
- Uses Context Parallelism to distribute long sequences across GPUs
- Ulysses attention for efficient long-sequence processing
- Handles videos up to ~3 minutes at 2 fps

---

## Data Creation Methodology

### Human-LLM Collaboration (No Distillation)
- **Key principle**: No proprietary model distillation (e.g., no GPT-4, Gemini data)
- **Pipeline**:
  1. Human annotation/narration
  2. LLM enrichment for scale
  3. Human validation/filtering

### Video Captioning Pipeline
1. **Human narration**: Describe what's happening in the video
2. **Speech-to-text**: Transcribe narration
3. **LLM enrichment**: Add frame-level visual details (colors, objects, actions)
4. **Quality control**: Human review

### Synthetic QA Generation
1. **Source**: Dense captions (human or LLM-generated)
2. **QA generation**: LLM creates questions answerable from caption
3. **Filtering**: Remove low-quality or ambiguous QAs
4. **Validation**: Sample-based human review

---

## Dataset Statistics

### Total Data Volume
- **9 new Molmo2 datasets**: 9+ million multimodal examples
- **Pre-training**: ~5 datasets, ~1.5M examples
- **SFT**: 100+ datasets, ~10M+ examples
- **Storage**: ~150GB total (pre-training + SFT)

### Data Distribution by Task
- **Video captioning**: ~500k examples
- **Video QA**: ~1.5M examples
- **Video pointing**: ~330k examples
- **Video tracking**: ~220k examples
- **Multi-image QA**: ~233k examples
- **Multi-image pointing**: ~470k examples
- **Image captioning**: ~710k examples
- **Image pointing/counting**: ~1M+ examples
- **Academic VQA**: ~3M+ examples
- **Text-only**: ~980k examples

---

## Task-Specific Token Weighting

To balance learning across diverse tasks, Molmo2 applies **per-token loss weighting**:

### Weighting Strategy
```python
TASK_WEIGHTS = {
    "video_caption": 1.0,        # Standard weight
    "image_caption": 1.0,
    "video_pointing": 2.0,       # Higher priority (novel task)
    "image_pointing": 2.0,
    "video_tracking": 2.0,       # Higher priority (novel task)
    "video_qa": 1.0,
    "image_qa": 1.0,
    "counting": 1.5,             # Moderate priority
}
```

### Rationale
- **Pointing/tracking are harder** → Need more learning signal
- **Prevents model from ignoring difficult tasks**
- **Improves overall task balance**

---

## Data Formats

### Point Format (HTML-like Compressed)

**Single Image**:
```
<point x, y, object_id>
Example: <point 0.5, 0.3, 0>
```

**Video (Temporal)**:
```
<point timestamp, x, y, object_id>
Example: <point 1.2, 0.5, 0.3, 0> <point 1.4, 0.52, 0.31, 0>
```

**Multi-Image**:
```
<point image_index, x, y, object_id>
Example: <point 0, 0.5, 0.3, 0> <point 1, 0.7, 0.6, 1>
```

### Coordinate System
- **Normalized** [0, 1] range
- `x`: Horizontal (0 = left, 1 = right)
- `y`: Vertical (0 = top, 1 = bottom)
- `timestamp`: Seconds from video start

### JSONL File Structure
All datasets use **JSONL** (JSON Lines) format:
- One JSON object per line
- Each line is a complete, valid JSON object
- Enables streaming and parallel processing

---

## Performance Benchmarks

### Video Understanding
- **Video QA**: Competitive with Gemini 2.0 Pro
- **Video Tracking**: 56.2 J&F (Molmo2-8B) vs 41.1 (Gemini 2.0 Pro)
- **Video Captioning**: Dense, detailed captions with temporal grounding

### Image Understanding
- **VQA**: Strong performance across multiple benchmarks
- **Pointing**: Accurate spatial grounding
- **Multi-image**: Reasoning across charts, tables, documents

### Text Understanding
- **Instruction following**: Competitive with text-only models
- **Long-form generation**: Coherent, detailed responses

---

## Training Infrastructure

### Hardware Requirements
- **Pre-training**: 16-64 GPUs (A100 80GB recommended)
- **SFT**: 8-32 GPUs
- **Long-context**: 16-64 GPUs (requires CP)

### Batch Sizes
- **Pre-training**: Global batch size 256
- **SFT**: Global batch size 128
- **Long-context**: Global batch size 32-64

### Training Time (Estimated)
- **Pre-training**: ~1-2 weeks on 32 A100s
- **SFT**: ~3-5 days on 16 A100s
- **Long-context**: ~1 day on 32 A100s

---

## Evaluation Datasets

Molmo2 also released **4 evaluation datasets**:

1. **Molmo2-CapEval** - Caption evaluation benchmark
2. **Molmo2-VideoPointEval** - Video pointing evaluation
3. **Molmo2-VideoCountEval** - Video counting evaluation
4. **Molmo2-VideoTrackEval** - Video tracking evaluation

---

## Key Takeaways

### For Researchers
1. **Open data**: All 9 datasets publicly available on HuggingFace
2. **No distillation**: Human-LLM collaboration without proprietary models
3. **Novel tasks**: Video pointing and tracking extend capabilities
4. **Reproducible**: Complete training pipeline documented

### For Practitioners
1. **3-stage training**: Pre-training → SFT → Long-context
2. **Data mixing**: Sqrt-proportional sampling + manual rebalancing
3. **Token weighting**: Higher weights for harder tasks (pointing, tracking)
4. **Efficient training**: Sequence packing, message-tree encoding

### For Dataset Users
1. **Diverse tasks**: Captioning, QA, pointing, tracking, multi-image
2. **Large scale**: 9M+ examples across 100+ datasets
3. **High quality**: Human-LLM collaboration with validation
4. **Accessible**: All on HuggingFace with permissive licenses

---

## References

- **Molmo2 Blog**: https://allenai.org/blog/molmo2
- **HuggingFace Collection**: https://huggingface.co/collections/allenai/molmo2-data
- **Individual Datasets**:
  - https://huggingface.co/datasets/allenai/Molmo2-VideoSubtitleQA
  - https://huggingface.co/datasets/allenai/Molmo2-VideoCapQA
  - https://huggingface.co/datasets/allenai/Molmo2-SynMultiImageQA
  - And 6 more on HuggingFace

---

## Citation

```bibtex
@article{molmo2_2025,
  title={Molmo2: Open State-of-the-Art Multimodal Models for Video and Multi-Image Understanding},
  author={Allen Institute for AI},
  year={2025},
  url={https://allenai.org/blog/molmo2}
}
```

---

**Document Created**: 2026-01-15  
**Author**: nanoMolmo2 Team  
**Version**: 1.0
