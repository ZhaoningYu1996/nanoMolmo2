#!/usr/bin/env python3
"""
Molmo2 Dataset Downloader

Download datasets for different training stages based on the Molmo2 Technical Report.

Usage:
    python download_datasets.py --stage pretrain   # Stage 1: 5 datasets (~80GB)
    python download_datasets.py --stage sft        # Stage 2&3: 100+ datasets (~500GB)
    python download_datasets.py --stage all        # All datasets
    python download_datasets.py --list             # List all datasets
    python download_datasets.py --check            # Check download status

Options:
    --stage {pretrain,sft,all}  Which stage to download
    --data-dir PATH             Directory to save data (default: ./data/molmo2)
    --list                      List all datasets and exit
    --check                     Check which datasets are already downloaded
    --force                     Re-download even if already exists
    --dry-run                   Show what would be downloaded without downloading
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET DEFINITIONS (Based on Molmo2 Technical Report)
# ============================================================================

@dataclass
class DatasetInfo:
    """Information about a single dataset."""
    name: str
    hf_path: str
    description: str
    stage: str  # "pretrain" or "sft"
    examples: str
    size_gb: float = 0.0  # Estimated size in GB
    ratio: Optional[str] = None  # For pretrain stage
    category: Optional[str] = None  # For sft stage


# Stage 1: Pre-training datasets (5 total)
# From Molmo2 tech report: "Stage 1: Pre-training"
PRETRAIN_DATASETS = [
    DatasetInfo(
        name="pixmo-cap",
        hf_path="allenai/pixmo-cap",
        description="Dense image captioning (~200 words avg)",
        stage="pretrain",
        examples="717k",
        ratio="60%",
    ),
    DatasetInfo(
        name="pixmo-points",
        hf_path="allenai/pixmo-points",
        description="Image pointing with referring expressions",
        stage="pretrain",
        examples="2.38M",
        ratio="15%",
    ),
    DatasetInfo(
        name="pixmo-count",
        hf_path="allenai/pixmo-count",
        description="Object counting QA",
        stage="pretrain",
        examples="36.9k",
        ratio="10%",
    ),
    DatasetInfo(
        name="cosyn-point",
        hf_path="allenai/cosyn-point",
        description="Synthetic pointing data",
        stage="pretrain",
        examples="68.1k",
        ratio="5%",
    ),
    DatasetInfo(
        name="tulu-3-sft-mixture",
        hf_path="allenai/tulu-3-sft-mixture",
        description="Text-only instruction data (NLP)",
        stage="pretrain",
        examples="940k",
        ratio="10%",
    ),
]

# Stage 2 & 3: SFT datasets (100+ total, Stage 3 uses same data)
# From Molmo2 tech report: "Stage 2: Supervised Fine-Tuning"
SFT_DATASETS = [
    # Molmo2 Original Datasets (9 new releases)
    DatasetInfo(
        name="Molmo2-Cap",
        hf_path="allenai/Molmo2-Cap",
        description="Video dense captioning",
        stage="sft",
        examples="104k video + 431k clip",
        size_gb=50,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-AskModelAnything",
        hf_path="allenai/Molmo2-AskModelAnything",
        description="Human-authored video QA",
        stage="sft",
        examples="43k",
        size_gb=20,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-VideoCapQA",
        hf_path="allenai/Molmo2-VideoCapQA",
        description="Synthetic QA from video captions",
        stage="sft",
        examples="1M",
        size_gb=40,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-VideoSubtitleQA",
        hf_path="allenai/Molmo2-VideoSubtitleQA",
        description="Video QA with subtitles",
        stage="sft",
        examples="300k",
        size_gb=30,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-VideoPoint",
        hf_path="allenai/Molmo2-VideoPoint",
        description="Video temporal pointing (NOVEL)",
        stage="sft",
        examples="330k",
        size_gb=25,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-VideoTrack",
        hf_path="allenai/Molmo2-VideoTrack",
        description="Video object tracking (NOVEL)",
        stage="sft",
        examples="220k",
        size_gb=25,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-MultiImageQA",
        hf_path="allenai/Molmo2-MultiImageQA",
        description="Multi-image QA",
        stage="sft",
        examples="45k",
        size_gb=15,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-SynMultiImageQA",
        hf_path="allenai/Molmo2-SynMultiImageQA",
        description="Synthetic multi-image QA",
        stage="sft",
        examples="188k",
        size_gb=20,
        category="molmo2",
    ),
    DatasetInfo(
        name="Molmo2-MultiImagePoint",
        hf_path="allenai/Molmo2-MultiImagePoint",
        description="Multi-image pointing",
        stage="sft",
        examples="470k",
        size_gb=25,
        category="molmo2",
    ),
    
    # PixMo Datasets
    DatasetInfo(
        name="pixmo-cap",
        hf_path="allenai/pixmo-cap",
        description="Dense image captioning",
        stage="sft",
        examples="710k",
        size_gb=30,
        category="pixmo",
    ),
    DatasetInfo(
        name="pixmo-ask-model-anything",
        hf_path="allenai/pixmo-ask-model-anything",
        description="Human-authored image QA",
        stage="sft",
        examples="71k",
        size_gb=10,
        category="pixmo",
    ),
    DatasetInfo(
        name="pixmo-cap-qa",
        hf_path="allenai/pixmo-cap-qa",
        description="Synthetic QA from captions",
        stage="sft",
        examples="190k",
        size_gb=15,
        category="pixmo",
    ),
    DatasetInfo(
        name="pixmo-clocks",
        hf_path="allenai/pixmo-clocks",
        description="Clock reading",
        stage="sft",
        examples="800k",
        size_gb=10,
        category="pixmo",
    ),
    
    # Academic Image Datasets
    DatasetInfo(
        name="ai2d",
        hf_path="allenai/ai2d",
        description="Diagram understanding",
        stage="sft",
        examples="15k",
        size_gb=5,
        category="academic",
    ),
    DatasetInfo(
        name="chartqa",
        hf_path="ahmed-masry/ChartQA",
        description="Chart understanding",
        stage="sft",
        examples="28k",
        size_gb=10,
        category="academic",
    ),
    DatasetInfo(
        name="docvqa",
        hf_path="lmms-lab/DocVQA",
        description="Document VQA",
        stage="sft",
        examples="39k",
        size_gb=15,
        category="academic",
    ),
    DatasetInfo(
        name="textvqa",
        hf_path="lmms-lab/textvqa",
        description="Text-based VQA",
        stage="sft",
        examples="35k",
        size_gb=10,
        category="academic",
    ),
    DatasetInfo(
        name="scienceqa",
        hf_path="derek-thomas/ScienceQA",
        description="Science QA",
        stage="sft",
        examples="6.2k",
        size_gb=3,
        category="academic",
    ),
    DatasetInfo(
        name="aokvqa",
        hf_path="HuggingFaceM4/A-OKVQA",
        description="Knowledge-based VQA",
        stage="sft",
        examples="34k",
        size_gb=8,
        category="academic",
    ),
    DatasetInfo(
        name="okvqa",
        hf_path="Multimodal-Fatima/OK-VQA_train",
        description="Outside knowledge VQA",
        stage="sft",
        examples="9k",
        size_gb=3,
        category="academic",
    ),
    DatasetInfo(
        name="infographicvqa",
        hf_path="lmms-lab/InfographicsVQA",
        description="Infographic VQA",
        stage="sft",
        examples="24k",
        size_gb=8,
        category="academic",
    ),
    DatasetInfo(
        name="stvqa",
        hf_path="lmms-lab/ST-VQA",
        description="Scene text VQA",
        stage="sft",
        examples="25k",
        size_gb=8,
        category="academic",
    ),
    DatasetInfo(
        name="tallyqa",
        hf_path="lmms-lab/TallyQA",
        description="Counting QA",
        stage="sft",
        examples="250k",
        size_gb=12,
        category="academic",
    ),
    DatasetInfo(
        name="gqa",
        hf_path="lmms-lab/GQA",
        description="Visual reasoning",
        stage="sft",
        examples="1M+",
        size_gb=25,
        category="academic",
    ),
    DatasetInfo(
        name="nlvr2",
        hf_path="lmms-lab/NLVR2",
        description="Visual reasoning",
        stage="sft",
        examples="86k",
        size_gb=8,
        category="academic",
    ),
    
    # Video Datasets
    DatasetInfo(
        name="nextqa",
        hf_path="lmms-lab/NExT-QA",
        description="Video QA with reasoning",
        stage="sft",
        examples="34k",
        size_gb=50,
        category="video",
    ),
    DatasetInfo(
        name="perception-test",
        hf_path="lmms-lab/Perception-Test",
        description="Video perception",
        stage="sft",
        examples="12k",
        size_gb=40,
        category="video",
    ),
    DatasetInfo(
        name="activitynet-qa",
        hf_path="lmms-lab/ActivityNet-QA",
        description="Activity video QA",
        stage="sft",
        examples="58k",
        size_gb=100,
        category="video",
    ),
    DatasetInfo(
        name="videochatgpt",
        hf_path="lmms-lab/VideoChatGPT",
        description="Video instruction following",
        stage="sft",
        examples="100k",
        size_gb=80,
        category="video",
    ),
]

ALL_DATASETS = PRETRAIN_DATASETS + SFT_DATASETS


# ============================================================================
# DOWNLOADER CLASS
# ============================================================================

class DatasetDownloader:
    """Downloads and manages Molmo2 datasets."""
    
    def __init__(self, data_dir: Path, force: bool = False):
        self.data_dir = data_dir
        self.force = force
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.data_dir / ".download_status.json"
        self.status = self._load_status()
    
    def _load_status(self) -> Dict:
        """Load download status from file."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {"downloaded": {}, "failed": {}}
    
    def _save_status(self):
        """Save download status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def is_downloaded(self, dataset: DatasetInfo) -> bool:
        """Check if a dataset has already been downloaded."""
        # Check status file
        if dataset.name in self.status.get("downloaded", {}):
            # Verify the directory exists
            dataset_path = self.data_dir / dataset.name
            if dataset_path.exists() and any(dataset_path.iterdir()):
                return True
        
        # Check if directory exists with data
        dataset_path = self.data_dir / dataset.name
        if dataset_path.exists():
            files = list(dataset_path.glob("*.parquet")) + list(dataset_path.glob("*.json*"))
            if files:
                # Update status
                self.status.setdefault("downloaded", {})[dataset.name] = {
                    "hf_path": dataset.hf_path,
                    "files": [f.name for f in files[:5]],  # Store first 5 files
                }
                self._save_status()
                return True
        
        return False
    
    def download(self, dataset: DatasetInfo, dry_run: bool = False) -> bool:
        """Download a single dataset."""
        
        # Check if already downloaded
        if not self.force and self.is_downloaded(dataset):
            logger.info(f"✓ {dataset.name} - Already downloaded, skipping")
            return True
        
        if dry_run:
            logger.info(f"○ {dataset.name} - Would download from {dataset.hf_path}")
            return True
        
        logger.info(f"↓ {dataset.name} - Downloading from {dataset.hf_path}...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("Please install datasets: pip install datasets")
            return False
        
        try:
            # Download from HuggingFace
            try:
                dataset_obj = load_dataset(
                    dataset.hf_path,
                    cache_dir=str(self.data_dir / ".cache"),
                )
            except Exception:
                # Fallback: try with trust_remote_code for older datasets
                dataset_obj = load_dataset(
                    dataset.hf_path,
                    cache_dir=str(self.data_dir / ".cache"),
                    trust_remote_code=True,
                )
            
            # Save to local directory
            save_path = self.data_dir / dataset.name
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save each split
            if hasattr(dataset_obj, 'keys'):
                for split_name in dataset_obj.keys():
                    output_file = save_path / f"{split_name}.parquet"
                    dataset_obj[split_name].to_parquet(str(output_file))
                    logger.info(f"  Saved {split_name} ({len(dataset_obj[split_name])} examples)")
            else:
                output_file = save_path / "data.parquet"
                dataset_obj.to_parquet(str(output_file))
                logger.info(f"  Saved ({len(dataset_obj)} examples)")
            
            # Update status
            self.status.setdefault("downloaded", {})[dataset.name] = {
                "hf_path": dataset.hf_path,
                "timestamp": str(Path().resolve()),
            }
            if dataset.name in self.status.get("failed", {}):
                del self.status["failed"][dataset.name]
            self._save_status()
            
            logger.info(f"✓ {dataset.name} - Download complete")
            return True
            
        except Exception as e:
            logger.error(f"✗ {dataset.name} - Failed: {str(e)}")
            self.status.setdefault("failed", {})[dataset.name] = str(e)
            self._save_status()
            return False
    
    def download_stage(self, stage: str, dry_run: bool = False) -> Dict:
        """Download all datasets for a specific stage."""
        if stage == "pretrain":
            datasets = PRETRAIN_DATASETS
        elif stage == "sft":
            datasets = SFT_DATASETS
        elif stage == "all":
            datasets = ALL_DATASETS
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Remove duplicates (some datasets appear in both stages)
        seen = set()
        unique_datasets = []
        for d in datasets:
            if d.name not in seen:
                seen.add(d.name)
                unique_datasets.append(d)
        
        results = {"success": [], "skipped": [], "failed": []}
        
        for dataset in unique_datasets:
            if not self.force and self.is_downloaded(dataset):
                results["skipped"].append(dataset.name)
                logger.info(f"✓ {dataset.name} - Already downloaded")
            elif self.download(dataset, dry_run=dry_run):
                results["success"].append(dataset.name)
            else:
                results["failed"].append(dataset.name)
        
        return results


# ============================================================================
# CLI FUNCTIONS
# ============================================================================

def list_datasets():
    """List all available datasets."""
    print("\n" + "=" * 70)
    print("MOLMO2 DATASETS (Based on Technical Report)")
    print("=" * 70)
    
    # Pre-training
    print("\n┌─ STAGE 1: PRE-TRAINING (5 datasets, ~80GB)")
    print("│")
    total_size = 0
    for i, d in enumerate(PRETRAIN_DATASETS, 1):
        print(f"│  {i}. {d.name:<30} {d.ratio:>5}  ~{d.size_gb:>3}GB")
        print(f"│     └─ {d.description}")
        total_size += d.size_gb
    print(f"│")
    print(f"└─ Total: {len(PRETRAIN_DATASETS)} datasets, ~{total_size}GB")
    
    # SFT
    print("\n┌─ STAGE 2 & 3: SFT (100+ datasets)")
    print("│  Note: Stage 3 uses SAME data with longer sequences")
    print("│")
    
    categories = {}
    for d in SFT_DATASETS:
        cat = d.category or "other"
        categories.setdefault(cat, []).append(d)
    
    total_sft_size = 0
    for cat, datasets in categories.items():
        cat_size = sum(d.size_gb for d in datasets)
        total_sft_size += cat_size
        print(f"│  ┌─ {cat.upper()} ({len(datasets)} datasets, ~{cat_size}GB)")
        for d in datasets:
            print(f"│  │  • {d.name:<30} ~{d.size_gb:>3}GB")
        print(f"│  │")
    
    print(f"│")
    print(f"└─ Total: {len(SFT_DATASETS)} datasets, ~{total_sft_size}GB")
    
    print("\n" + "=" * 70)
    print(f"GRAND TOTAL: {len(set(d.name for d in ALL_DATASETS))} unique datasets")
    print("=" * 70 + "\n")


def check_status(data_dir: Path):
    """Check download status of all datasets."""
    downloader = DatasetDownloader(data_dir)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD STATUS")
    print("=" * 70)
    
    downloaded = []
    not_downloaded = []
    
    for dataset in ALL_DATASETS:
        if downloader.is_downloaded(dataset):
            downloaded.append(dataset)
        else:
            not_downloaded.append(dataset)
    
    # Remove duplicates
    downloaded_names = list(set(d.name for d in downloaded))
    not_downloaded_names = list(set(d.name for d in not_downloaded) - set(downloaded_names))
    
    print(f"\n✓ Downloaded ({len(downloaded_names)}):")
    for name in sorted(downloaded_names):
        print(f"  • {name}")
    
    print(f"\n○ Not downloaded ({len(not_downloaded_names)}):")
    for name in sorted(not_downloaded_names):
        print(f"  • {name}")
    
    total = len(downloaded_names) + len(not_downloaded_names)
    pct = len(downloaded_names) / total * 100 if total > 0 else 0
    print(f"\nProgress: {len(downloaded_names)}/{total} ({pct:.1f}%)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Molmo2 Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stage pretrain   Download Stage 1 pre-training data (~80GB)
  %(prog)s --stage sft        Download Stage 2&3 SFT data (~500GB)
  %(prog)s --stage all        Download all datasets
  %(prog)s --list             List all available datasets
  %(prog)s --check            Check which datasets are downloaded
        """
    )
    
    parser.add_argument(
        "--stage",
        choices=["pretrain", "sft", "all"],
        help="Which stage to download: pretrain (Stage 1), sft (Stage 2&3), or all"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "./data/molmo2")),
        help="Directory to save datasets (default: ./data/molmo2)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check download status"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_datasets()
        return
    
    # Handle --check
    if args.check:
        check_status(args.data_dir)
        return
    
    # Require --stage if not --list or --check
    if not args.stage:
        parser.print_help()
        print("\nError: Please specify --stage (pretrain, sft, or all)")
        sys.exit(1)
    
    # Print header
    print("\n" + "=" * 70)
    print("MOLMO2 DATASET DOWNLOADER")
    print("=" * 70)
    print(f"\nStage: {args.stage}")
    print(f"Data directory: {args.data_dir.absolute()}")
    print(f"Force re-download: {args.force}")
    print(f"Dry run: {args.dry_run}")
    
    if args.stage == "pretrain":
        print(f"\nDatasets: 5 (Pre-training)")
        print("Storage: ~80GB")
    elif args.stage == "sft":
        print(f"\nDatasets: {len(SFT_DATASETS)} (SFT - also used for Stage 3)")
        print("Storage: ~500GB")
    else:
        print(f"\nDatasets: {len(set(d.name for d in ALL_DATASETS))} (All stages)")
        print("Storage: ~600GB")
    
    print("=" * 70 + "\n")
    
    # Download
    downloader = DatasetDownloader(args.data_dir, force=args.force)
    results = downloader.download_stage(args.stage, dry_run=args.dry_run)
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  ✓ Success: {len(results['success'])}")
    print(f"  ○ Skipped (already downloaded): {len(results['skipped'])}")
    print(f"  ✗ Failed: {len(results['failed'])}")
    
    if results['failed']:
        print(f"\nFailed datasets:")
        for name in results['failed']:
            print(f"  • {name}")
        print("\nTry running again or download manually from HuggingFace.")
    
    print(f"\nData saved to: {args.data_dir.absolute()}")
    
    if not args.dry_run:
        print("\nNext steps:")
        if args.stage == "pretrain":
            print("  python examples/train_stage1.py")
        elif args.stage == "sft":
            print("  python examples/train_stage2.py  # Stage 2")
            print("  python examples/train_stage3.py  # Stage 3 (same data, longer seq)")
        else:
            print("  python examples/train_stage1.py  # Start with Stage 1")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
