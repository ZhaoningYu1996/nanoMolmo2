# Molmo2 Training Pipeline Visualization

Complete visual guide to the 3-stage training pipeline for nanoMolmo2.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MOLMO2 TRAINING PIPELINE                      │
│                       (3-Stage Approach)                         │
└─────────────────────────────────────────────────────────────────┘

Stage 1: Pre-training          Stage 2: SFT              Stage 3: Long-Context
┌──────────────┐              ┌──────────────┐          ┌──────────────┐
│ Image-Text   │              │ Multimodal   │          │ Extended     │
│ Alignment    │─────────────>│ Instruction  │─────────>│ Sequences    │
│              │              │ Following    │          │              │
└──────────────┘              └──────────────┘          └──────────────┘
  ~100K steps                   ~50K steps                 ~2K steps
  4,096 tokens                  4,096 tokens               36,864 tokens
  Image only                    128 frames                 384 frames
```

---

## Stage 1: Pre-training (Vision-Language Alignment)

### Goal
Learn general visual understanding and align visual features with language model.

### Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                        INPUT DATA (60-30-10 Mix)                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  60% Dense Captioning          30% Pointing          10% NLP       │
│  ┌───────────────┐            ┌──────────┐          ┌─────────┐  │
│  │ PixMo-Cap     │            │PixMo-Pts │          │ Tulu    │  │
│  │ ~710k samples │            │~1M sample│          │~980k    │  │
│  │               │            │          │          │samples  │  │
│  │ "A fluffy     │            │"Point to │          │"Q: What │  │
│  │  orange cat   │            │ all cats"│          │ is...?" │  │
│  │  sleeping..." │            │<pt 0.5,  │          │"A: ..." │  │
│  └───────────────┘            │   0.3>   │          └─────────┘  │
│                                └──────────┘                        │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Load images (PIL)                                              │
│  2. Resize to 224×224                                              │
│  3. Tokenize text                                                  │
│  4. Format points in HTML-like syntax: <point x, y, obj_id>       │
│  5. Apply task weights (caption: 1.0x, pointing: 2.0x)            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                     WEIGHTED SAMPLING                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Sample probability:                                               │
│  ┌────────────────┬──────────┬──────────────┐                     │
│  │ Dataset        │ Weight   │ Sample Rate  │                     │
│  ├────────────────┼──────────┼──────────────┤                     │
│  │ Dense caption  │ 0.60     │ ████████████ │                     │
│  │ Pointing       │ 0.30     │ ██████       │                     │
│  │ NLP            │ 0.10     │ ██           │                     │
│  └────────────────┴──────────┴──────────────┘                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                          BATCHING                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Batch size: 32 (per GPU)                                          │
│  Sequence length: 4,096 tokens                                     │
│  Gradient accumulation: 4 steps                                    │
│  Effective batch: 32 × 4 GPUs × 4 accum = 512                     │
│                                                                     │
│  ┌──────────────────────────────────────────┐                     │
│  │ Batch Structure:                          │                     │
│  │ ┌──────────────────────────────────────┐ │                     │
│  │ │ input_ids:       [32, 4096]          │ │                     │
│  │ │ attention_mask:  [32, 4096]          │ │                     │
│  │ │ pixel_values:    [32, 1, 3, 224, 224]│ │                     │
│  │ │ loss_weights:    [32, 4096]          │ │                     │
│  │ └──────────────────────────────────────┘ │                     │
│  └──────────────────────────────────────────┘                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Steps: ~100,000                                                   │
│  Learning rate: 1e-4 with warmup (2K steps)                       │
│  Optimizer: AdamW                                                  │
│  Hardware: 16-32 A100 GPUs                                         │
│  Time: 1-2 weeks                                                   │
│                                                                     │
│  Output: Stage1 checkpoint → feeds into Stage 2                   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Stage 2: Supervised Fine-Tuning (Instruction Following)

### Goal
Learn to follow instructions across diverse multimodal tasks including video understanding.

### Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                  INPUT DATA (Sqrt-Proportional Mix)                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Molmo2 Datasets (9 new)     PixMo Datasets        Academic        │
│  ┌─────────────────────┐    ┌──────────────┐    ┌──────────┐     │
│  │ Video Captioning    │    │ Image Cap    │    │ VQA v2   │     │
│  │ Video QA            │    │ Image QA     │    │ DocVQA   │     │
│  │ Video Pointing ⭐   │    │ Pointing     │    │ TextVQA  │     │
│  │ Video Tracking ⭐   │    │ Counting     │    │ ChartQA  │     │
│  │ Multi-Image QA      │    │ Clocks       │    │ AI2D     │     │
│  │ Multi-Image Point   │    └──────────────┘    │ NLVR2    │     │
│  └─────────────────────┘                        │ ...100+  │     │
│   ⭐ = Novel contribution                        └──────────┘     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For Images:                   For Videos:                         │
│  1. Load & resize              1. Sample frames at 2 fps           │
│  2. Tokenize text              2. Max 128 frames                   │
│  3. Format points              3. Extract timestamps               │
│                                4. Process subtitles (if any)       │
│                                5. Format temporal points:           │
│                                   <point t, x, y, obj_id>          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                 SQRT-PROPORTIONAL SAMPLING                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Weight_i = sqrt(Size_i) / Σ sqrt(Size_j)                         │
│                                                                     │
│  Example:                                                          │
│  ┌──────────────────┬──────────┬────────┬──────────┐             │
│  │ Dataset          │ Size     │ √Size  │ Weight   │             │
│  ├──────────────────┼──────────┼────────┼──────────┤             │
│  │ Molmo2-CapQA     │ 1,000k   │ 1000   │ 0.25     │             │
│  │ VQA-v2           │  440k    │  663   │ 0.16     │             │
│  │ PixMo-Cap        │  710k    │  843   │ 0.21     │             │
│  │ Molmo2-Track     │  220k    │  469   │ 0.12     │             │
│  │ ...              │  ...     │  ...   │ ...      │             │
│  └──────────────────┴──────────┴────────┴──────────┘             │
│                                                                     │
│  This prevents large datasets from dominating!                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                          BATCHING                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Batch size: 16 (per GPU)                                          │
│  Sequence length: 4,096 tokens                                     │
│  Max frames: 128 (videos)                                          │
│  Frame rate: 2 fps                                                 │
│  Gradient accumulation: 8 steps                                    │
│  Effective batch: 16 × 4 GPUs × 8 accum = 512                     │
│                                                                     │
│  ┌──────────────────────────────────────────┐                     │
│  │ Batch Structure (Videos):                 │                     │
│  │ ┌──────────────────────────────────────┐ │                     │
│  │ │ input_ids:       [16, 4096]          │ │                     │
│  │ │ attention_mask:  [16, 4096]          │ │                     │
│  │ │ pixel_values:    [16, 128, 3, 224,   │ │                     │
│  │ │                        224]           │ │                     │
│  │ │ timestamps:      [16, 128]           │ │                     │
│  │ │ loss_weights:    [16, 4096]          │ │                     │
│  │ └──────────────────────────────────────┘ │                     │
│  └──────────────────────────────────────────┘                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Steps: ~50,000                                                    │
│  Learning rate: 5e-5 with warmup (1K steps)                       │
│  Optimizer: AdamW                                                  │
│  Hardware: 8-16 A100 GPUs                                          │
│  Time: 3-5 days                                                    │
│                                                                     │
│  Output: Stage2 checkpoint → feeds into Stage 3                   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Stage 3: Long-Context SFT (Extended Sequences)

### Goal
Enable the model to handle very long video sequences (up to ~3 minutes).

### Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                  INPUT DATA (Same as Stage 2)                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Focus on video datasets that benefit from long context:          │
│                                                                     │
│  ┌──────────────────────────────────────────────────┐             │
│  │ • Molmo2-Cap (dense video captioning)            │             │
│  │ • Molmo2-VideoCapQA (long-form video QA)         │             │
│  │ • Molmo2-VideoSubtitleQA (QA with subtitles)     │             │
│  │ • Molmo2-VideoTrack (object tracking over time)  │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                  EXTENDED PREPROCESSING                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Video Processing:                                                 │
│  1. Sample frames at 2 fps                                         │
│  2. Max 384 frames (3× more than Stage 2!)                        │
│  3. Extract timestamps for all frames                              │
│  4. Process long-form subtitles                                    │
│  5. Format temporal sequences:                                     │
│     <point t₁, x₁, y₁, id> ... <point t₃₈₄, x₃₈₄, y₃₈₄, id>     │
│                                                                     │
│  Memory requirement per video:                                     │
│  384 frames × 3 channels × 224 × 224 × 4 bytes ≈ 231 MB          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│                          BATCHING                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Batch size: 4 (per GPU) ← Smaller due to memory!                 │
│  Sequence length: 36,864 tokens (9× longer!)                      │
│  Max frames: 384 (3× more!)                                        │
│  Frame rate: 2 fps                                                 │
│  Gradient accumulation: 16 steps                                   │
│  Effective batch: 4 × 8 GPUs × 16 accum = 512                     │
│                                                                     │
│  ┌──────────────────────────────────────────┐                     │
│  │ Batch Structure:                          │                     │
│  │ ┌──────────────────────────────────────┐ │                     │
│  │ │ input_ids:       [4, 36864]          │ │                     │
│  │ │ attention_mask:  [4, 36864]          │ │                     │
│  │ │ pixel_values:    [4, 384, 3, 224,    │ │                     │
│  │ │                       224]            │ │                     │
│  │ │ timestamps:      [4, 384]            │ │                     │
│  │ │ loss_weights:    [4, 36864]          │ │                     │
│  │ └──────────────────────────────────────┘ │                     │
│  └──────────────────────────────────────────┘                     │
│                                                                     │
│  ⚠️ Requires Context Parallelism (CP):                            │
│  Split long sequence across multiple GPUs                          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                   ↓
┌────────────────────────────────────────────────────────────────────┐
│               MODEL TRAINING (with CP)                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Steps: ~2,000 (short fine-tuning)                                │
│  Learning rate: 3e-5 with warmup (100 steps)                      │
│  Optimizer: AdamW                                                  │
│  Hardware: 16-32 A100 GPUs (80GB recommended)                      │
│  Time: ~1 day                                                      │
│                                                                     │
│  Special Techniques:                                               │
│  • Context Parallelism: Distribute sequence across GPUs            │
│  • Ulysses Attention: Efficient attention for long sequences       │
│  • Ring Attention: Reduce memory footprint                         │
│                                                                     │
│  Output: Final Molmo2 model!                                       │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Mixing Strategies

### Stage 1: Fixed Ratios

```
┌────────────────────────────────────────────────────────────────┐
│                     PRE-TRAINING MIX                            │
└────────────────────────────────────────────────────────────────┘

  Dense Captioning: 60%  ████████████████████████████████████
  Image Pointing:   30%  ██████████████████
  NLP Data:         10%  ██████
```

### Stage 2 & 3: Sqrt-Proportional

```
┌────────────────────────────────────────────────────────────────┐
│               SFT MIX (Sqrt-Proportional)                       │
└────────────────────────────────────────────────────────────────┘

Without sqrt-proportional:           With sqrt-proportional:
(Linear to size)                     (Balanced)

Large dataset (1M):  ███████████    Large dataset:  ████████
Medium dataset (100k): ██            Medium dataset: █████
Small dataset (10k):   ▌             Small dataset:  ███

Problem: Large datasets dominate    Solution: Fair representation!
```

---

## Task-Specific Weighting

Loss is weighted per-token based on task difficulty:

```
┌────────────────────────────────────────────────────────────────┐
│                   TASK WEIGHTS                                  │
└────────────────────────────────────────────────────────────────┘

Captioning:        1.0×  ████████
QA:                1.0×  ████████
Counting:          1.5×  ████████████
Pointing:          2.0×  ████████████████
Tracking:          2.0×  ████████████████

Rationale:
• Pointing and tracking are harder → need more learning signal
• Prevents model from avoiding difficult tasks
• Balances learning across diverse capabilities
```

---

## Hardware Scaling

### Recommended GPU Configuration by Stage

```
┌─────────────┬──────────┬──────────┬──────────────────┬─────────┐
│ Stage       │ Min GPUs │ Rec GPUs │ GPU Type         │ Time    │
├─────────────┼──────────┼──────────┼──────────────────┼─────────┤
│ Stage 1     │    8     │   16-32  │ A100 40GB/80GB   │ 1-2 wk  │
│ Stage 2     │    4     │   8-16   │ A100 40GB/80GB   │ 3-5 day │
│ Stage 3     │    8     │  16-32   │ A100 80GB (req)  │ ~1 day  │
└─────────────┴──────────┴──────────┴──────────────────┴─────────┘

* Stage 3 requires high-bandwidth interconnect (NVLink) for CP
```

---

## Memory Requirements

### Per-GPU Memory Usage

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY BREAKDOWN                              │
└─────────────────────────────────────────────────────────────────┘

Stage 1 (Image, batch=32, seq=4096):
  Model params (0.6B):           ~2.4 GB
  Optimizer states:              ~7.2 GB
  Gradients:                     ~2.4 GB
  Activations (batch):           ~15 GB
  Visual features:               ~8 GB
  ─────────────────────────────────────
  Total:                         ~35 GB  ✓ Fits in A100 40GB

Stage 2 (Video 128 frames, batch=16, seq=4096):
  Model params:                  ~2.4 GB
  Optimizer states:              ~7.2 GB
  Gradients:                     ~2.4 GB
  Activations:                   ~12 GB
  Visual features (128 frames):  ~20 GB
  ─────────────────────────────────────
  Total:                         ~44 GB  ✓ Fits in A100 80GB

Stage 3 (Video 384 frames, batch=4, seq=36864):
  Model params:                  ~2.4 GB
  Optimizer states:              ~7.2 GB
  Gradients:                     ~2.4 GB
  Activations (long seq):        ~25 GB
  Visual features (384 frames):  ~45 GB
  ─────────────────────────────────────
  Total:                         ~84 GB  ⚠️ Requires CP across GPUs!
```

---

## Complete Training Timeline

```
Week 1-2: Stage 1 Pre-training
├─ Day 1-3:   Data download & verification
├─ Day 4-14:  Pre-training (100K steps)
└─ Day 15:    Checkpoint & validation

Week 3: Stage 2 SFT
├─ Day 1:     Data verification (videos)
├─ Day 2-6:   SFT training (50K steps)
└─ Day 7:     Checkpoint & validation

Week 4: Stage 3 Long-Context
├─ Day 1-2:   Setup Context Parallelism
├─ Day 3:     Long-context training (2K steps)
└─ Day 4-7:   Final evaluation & benchmarking

Total: ~4 weeks with 16-32 A100 GPUs
```

---

## Key Innovations Summary

### Novel Contributions from Molmo2

1. **Video Temporal Pointing**
   - Format: `<point timestamp, x, y, object_id>`
   - Extends 2D pointing to 4D (space + time)
   - Enables time-aware object localization

2. **Video Object Tracking**
   - Tracks objects across frames using natural language
   - Outperforms prior SOTA (56.2 vs 41.1 J&F)
   - Same point format but with temporal consistency

3. **Human-LLM Collaboration**
   - No proprietary model distillation
   - Human narration + LLM enrichment
   - Maintains data quality and openness

4. **Sqrt-Proportional Sampling**
   - Balances large and small datasets
   - Prevents large datasets from dominating
   - Simple but effective: `Weight ∝ √Size`

---

## References

- **Full Tech Report**: [MOLMO2_TECH_REPORT_SUMMARY.md](./MOLMO2_TECH_REPORT_SUMMARY.md)
- **Quick Start**: [QUICKSTART.md](./QUICKSTART.md)
- **Dataloader Docs**: [data/dataloaders/README.md](./data/dataloaders/README.md)
- **Molmo2 Blog**: https://allenai.org/blog/molmo2

---

**Last Updated**: 2026-01-15  
**Version**: 1.0  
**Contributors**: nanoMolmo2 Team
