#!/bin/bash

# Download all Molmo2 datasets
# This script downloads HuggingFace datasets and provides instructions for manual ones

set -e  # Exit on error

# Configuration
DATA_DIR="./data/molmo2_datasets"
CACHE_DIR=""  # Leave empty to use default HF cache

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Molmo2 Complete Dataset Downloader"
echo "=============================================="
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$DATA_DIR"/{pretraining,sft,eval,academic}

# Function to download HuggingFace datasets
download_hf() {
    local stage=$1
    local dataset=$2
    
    echo -e "${GREEN}[HF]${NC} Downloading $dataset..."
    python scripts/download_molmo2_data_v2.py \
        --stage "$stage" \
        --datasets "$dataset" \
        --data-dir "$DATA_DIR" \
        ${CACHE_DIR:+--cache-dir "$CACHE_DIR"} \
        --skip-manual
}

# Function to show manual download instruction
show_manual() {
    local name=$1
    local url=$2
    local target_dir=$3
    
    echo -e "${YELLOW}[MANUAL]${NC} $name"
    echo "  URL: $url"
    echo "  Save to: $target_dir"
    echo ""
}

echo ""
echo "=============================================="
echo "Phase 1: Pre-training Datasets (HuggingFace)"
echo "=============================================="
echo ""

# Pre-training HuggingFace datasets
download_hf "pretraining" "pixmo-cap"
download_hf "pretraining" "pixmo-points"
download_hf "pretraining" "pixmo-count"
download_hf "pretraining" "tulu"

echo ""
echo "=============================================="
echo "Phase 2: SFT Datasets - Molmo2 Originals"
echo "=============================================="
echo ""

# Molmo2 original datasets
download_hf "sft" "molmo2-cap"
download_hf "sft" "molmo2-askmodelanything"
download_hf "sft" "molmo2-capqa"
download_hf "sft" "molmo2-subtitleqa"
download_hf "sft" "molmo2-videopoint"
download_hf "sft" "molmo2-videotrack"
download_hf "sft" "molmo2-multiimageqa"
download_hf "sft" "molmo2-synmultiimageqa"
download_hf "sft" "molmo2-multiimagepoint"

echo ""
echo "=============================================="
echo "Phase 3: SFT Datasets - PixMo"
echo "=============================================="
echo ""

download_hf "sft" "pixmo-cap"
download_hf "sft" "pixmo-askmodelanything"
download_hf "sft" "pixmo-capqa"
download_hf "sft" "pixmo-clocks"

echo ""
echo "=============================================="
echo "Phase 4: SFT Datasets - Academic (HuggingFace)"
echo "=============================================="
echo ""

# Academic datasets available on HuggingFace
download_hf "sft" "llava-665k-multi"
download_hf "sft" "tallyqa"
download_hf "sft" "vqa-v2"
download_hf "sft" "docvqa"
download_hf "sft" "textvqa"
download_hf "sft" "chartqa"
download_hf "sft" "st-vqa"
download_hf "sft" "infographicvqa"
download_hf "sft" "ai2d"
download_hf "sft" "nlvr2"
download_hf "sft" "a-okvqa"
download_hf "sft" "ok-vqa"
download_hf "sft" "scienceqa"

echo ""
echo "=============================================="
echo "Phase 5: Evaluation Datasets"
echo "=============================================="
echo ""

# Evaluation datasets
download_hf "eval" "molmo2-capeval"
download_hf "eval" "molmo2-videopointeval"
download_hf "eval" "molmo2-videocounteval"
download_hf "eval" "molmo2-videotrackeval"

echo ""
echo "=============================================="
echo "Download Summary"
echo "=============================================="
echo ""

# Count downloaded datasets
hf_count=$(find "$DATA_DIR" -name "*.parquet" | wc -l)
echo -e "${GREEN}✓ Downloaded $hf_count HuggingFace dataset splits${NC}"
echo ""

echo "=============================================="
echo "Manual Downloads Required (70+ datasets)"
echo "=============================================="
echo ""
echo "The following datasets require manual download."
echo "See detailed instructions with:"
echo "  python scripts/download_molmo2_data_v2.py --show-manual-instructions"
echo ""

# List manual download categories
echo "Categories requiring manual download:"
echo ""
echo "1. CoSyn Datasets (8 datasets)"
show_manual "CoSyn-Point/Chart/Doc/Table/etc" \
    "https://github.com/allenai/CoSyn" \
    "$DATA_DIR/academic/cosyn/"

echo "2. Chart/Table Datasets (4 datasets)"
show_manual "PlotQA, DVQA, FigureQA, TabWMP" \
    "See --show-manual-instructions for URLs" \
    "$DATA_DIR/academic/charts/"

echo "3. Video Understanding (30+ datasets)"
show_manual "TVQA, Ego4D, Kinetics, ActivityNet, etc" \
    "See --show-manual-instructions for URLs" \
    "$DATA_DIR/academic/video/"

echo "4. Video Pointing (6 datasets)"
show_manual "MeViS, RefVOS, LV-VIS, OVIS, BURST, Ref-DAVIS17" \
    "See --show-manual-instructions for URLs" \
    "$DATA_DIR/academic/video_pointing/"

echo "5. Video Tracking (14 datasets)"
show_manual "ViCaS, ReVOS, TrackingNet, VastTrack, GOT-10k, LaSOT, etc" \
    "See --show-manual-instructions for URLs" \
    "$DATA_DIR/academic/video_tracking/"

echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo ""
echo "1. Inspect downloaded datasets:"
echo "   python scripts/inspect_molmo2_data.py"
echo ""
echo "2. Download manual datasets as needed:"
echo "   python scripts/download_molmo2_data_v2.py --show-manual-instructions --stage sft"
echo ""
echo "3. Start with pre-training datasets (all downloaded):"
echo "   - pixmo-cap (dense captioning)"
echo "   - pixmo-points (pointing)"
echo "   - pixmo-count (counting)"
echo ""
echo "4. For full SFT training, download manual datasets based on your needs"
echo ""
echo -e "${GREEN}✓ HuggingFace dataset download complete!${NC}"
