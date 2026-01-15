#!/bin/bash
# Simple script to download all Molmo2 datasets

set -e  # Exit on error

echo "=============================================="
echo "Molmo2 Dataset Download Script"
echo "=============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if requirements are installed
echo "Step 1: Installing Python dependencies..."
pip install -q datasets transformers tqdm pyarrow pillow || {
    echo "Error: Failed to install dependencies"
    echo "Please run: pip install -r requirements.txt"
    exit 1
}

echo "✓ Dependencies installed"
echo ""

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR:-./data/molmo2_datasets}"
mkdir -p "$OUTPUT_DIR"

echo "Step 2: Downloading datasets to $OUTPUT_DIR"
echo ""

# Download datasets
python3 scripts/download_datasets.py --all --output-dir "$OUTPUT_DIR" --skip-errors

echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo "Datasets saved to: $OUTPUT_DIR"
echo ""
echo "To verify downloads:"
echo "  ls -lh $OUTPUT_DIR"
echo ""
