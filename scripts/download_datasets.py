#!/usr/bin/env python3
"""
Simple script to download all Molmo2 datasets.

Usage:
    # Install dependencies first
    pip install -r requirements.txt
    
    # Download all datasets
    python scripts/download_datasets.py --all
    
    # Download specific dataset
    python scripts/download_datasets.py --dataset molmo2-cap
    
    # Download by stage
    python scripts/download_datasets.py --stage pretraining
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# All Molmo2 datasets available on HuggingFace
MOLMO2_DATASETS = {
    # 9 New Molmo2 datasets from the paper
    "molmo2-cap": "allenai/Molmo2-Cap",
    "molmo2-askmodelanything": "allenai/Molmo2-AskModelAnything",
    "molmo2-capqa": "allenai/Molmo2-VideoCapQA",
    "molmo2-subtitleqa": "allenai/Molmo2-VideoSubtitleQA",
    "molmo2-videopoint": "allenai/Molmo2-VideoPoint",
    "molmo2-videotrack": "allenai/Molmo2-VideoTrack",
    "molmo2-multiimageqa": "allenai/Molmo2-MultiImageQA",
    "molmo2-synmultiimageqa": "allenai/Molmo2-SynMultiImageQA",
    "molmo2-multiimagepoint": "allenai/Molmo2-MultiImagePoint",
    
    # Evaluation datasets
    "molmo2-capeval": "allenai/Molmo2-CapEval",
    "molmo2-videopointeval": "allenai/Molmo2-VideoPointEval",
    "molmo2-videocounteval": "allenai/Molmo2-VideoCountEval",
    "molmo2-videotrackeval": "allenai/Molmo2-VideoTrackEval",
}

# PixMo datasets (used in pre-training and SFT)
PIXMO_DATASETS = {
    "pixmo-cap": "allenai/pixmo-cap",
    "pixmo-points": "allenai/pixmo-points",
    "pixmo-count": "allenai/pixmo-count",
    "pixmo-docs": "allenai/pixmo-docs",
    "pixmo-ask-model-anything": "allenai/pixmo-ask-model-anything",
    "pixmo-cap-qa": "allenai/pixmo-cap-qa",
    "pixmo-clocks": "allenai/pixmo-clocks",
}

# Academic datasets (available on HuggingFace)
ACADEMIC_DATASETS = {
    "llava-instruct": "liuhaotian/LLaVA-Instruct-150K",
    "vqa-v2": "HuggingFaceM4/VQAv2",
    "tallyqa": "HuggingFaceM4/TallyQA",
    "chartqa": "HuggingFaceM4/ChartQA",
    "docvqa": "naver-clova-ix/docvqa",
    "textvqa": "textvqa/textvqa",
    "st-vqa": "naver-clova-ix/st-vqa",
    "infographicvqa": "naver-clova-ix/infographicvqa",
    "ai2d": "allenai/ai2d",
    "nlvr2": "HuggingFaceM4/NLVR2",
    "a-okvqa": "HuggingFaceM4/A-OKVQA",
    "ok-vqa": "HuggingFaceM4/OK-VQA",
    "scienceqa": "derek-thomas/ScienceQA",
}

# Text-only datasets
TEXT_DATASETS = {
    "tulu-v2-sft": "allenai/tulu-v2-sft-mixture",
}

# Training stage groupings
STAGE_DATASETS = {
    "pretraining": {
        **{k: v for k, v in PIXMO_DATASETS.items() if k in ["pixmo-cap", "pixmo-points", "pixmo-count"]},
        **TEXT_DATASETS,
    },
    "sft": {
        **MOLMO2_DATASETS,
        **PIXMO_DATASETS,
        **ACADEMIC_DATASETS,
        **TEXT_DATASETS,
    },
    "eval": {
        k: v for k, v in MOLMO2_DATASETS.items() if "eval" in k
    },
}


def download_dataset(name, hf_path, output_dir, cache_dir=None):
    """Download a single dataset from HuggingFace."""
    print(f"\n{'='*70}")
    print(f"Downloading: {name}")
    print(f"From: {hf_path}")
    print(f"To: {output_dir / name}")
    print('='*70)
    
    try:
        # Load dataset from HuggingFace
        print(f"Loading dataset from HuggingFace...")
        dataset = load_dataset(hf_path, cache_dir=cache_dir, trust_remote_code=True)
        
        # Create output directory
        dataset_dir = output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to disk
        print(f"Saving to {dataset_dir}...")
        
        if isinstance(dataset, dict):
            # Dataset has splits (train, validation, test)
            for split_name, split_data in dataset.items():
                output_file = dataset_dir / f"{split_name}.parquet"
                split_data.to_parquet(str(output_file))
                print(f"  ✓ Saved {split_name}: {len(split_data)} examples → {output_file}")
        else:
            # Single split dataset
            output_file = dataset_dir / "data.parquet"
            dataset.to_parquet(str(output_file))
            print(f"  ✓ Saved {len(dataset)} examples → {output_file}")
        
        print(f"✓ Successfully downloaded {name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {name}: {str(e)}")
        print(f"  You may need to:")
        print(f"  1. Accept the dataset license on HuggingFace")
        print(f"  2. Login with: huggingface-cli login")
        print(f"  3. Check if the dataset path is correct")
        return False


def download_all_datasets(output_dir, cache_dir=None, dataset_dict=None, skip_errors=True):
    """Download all datasets in the given dictionary."""
    if dataset_dict is None:
        dataset_dict = {**MOLMO2_DATASETS, **PIXMO_DATASETS, **ACADEMIC_DATASETS, **TEXT_DATASETS}
    
    total = len(dataset_dict)
    successful = 0
    failed = []
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {total} DATASETS")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir or 'default (~/.cache/huggingface)'}")
    print(f"{'='*70}\n")
    
    for i, (name, hf_path) in enumerate(dataset_dict.items(), 1):
        print(f"\n[{i}/{total}] Processing {name}...")
        
        success = download_dataset(name, hf_path, output_dir, cache_dir)
        
        if success:
            successful += 1
        else:
            failed.append(name)
            if not skip_errors:
                print(f"\nStopping due to error. Use --skip-errors to continue on errors.")
                break
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Total datasets: {total}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")
        print(f"\nThese datasets may require:")
        print(f"  1. HuggingFace login: huggingface-cli login")
        print(f"  2. Dataset access approval on HuggingFace website")
        print(f"  3. Manual download from original sources")
    
    print(f"\nDatasets saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Molmo2 datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_datasets.py --all
  
  # Download specific dataset
  python scripts/download_datasets.py --dataset molmo2-cap
  
  # Download by stage
  python scripts/download_datasets.py --stage pretraining
  python scripts/download_datasets.py --stage sft
  
  # Download only Molmo2 datasets
  python scripts/download_datasets.py --molmo2-only
  
  # Download only PixMo datasets
  python scripts/download_datasets.py --pixmo-only
  
  # Custom output directory
  python scripts/download_datasets.py --all --output-dir /data/molmo2
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Download a specific dataset by name"
    )
    parser.add_argument(
        "--stage",
        choices=["pretraining", "sft", "eval"],
        help="Download datasets for a specific training stage"
    )
    parser.add_argument(
        "--molmo2-only",
        action="store_true",
        help="Download only the 9 new Molmo2 datasets"
    )
    parser.add_argument(
        "--pixmo-only",
        action="store_true",
        help="Download only PixMo datasets"
    )
    parser.add_argument(
        "--academic-only",
        action="store_true",
        help="Download only academic datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/molmo2_datasets",
        help="Output directory for datasets (default: ./data/molmo2_datasets)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=True,
        help="Continue downloading even if some datasets fail (default: True)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets without downloading"
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\nAvailable datasets:\n")
        print("Molmo2 Datasets (9 new):")
        for name in MOLMO2_DATASETS:
            print(f"  - {name}")
        print("\nPixMo Datasets:")
        for name in PIXMO_DATASETS:
            print(f"  - {name}")
        print("\nAcademic Datasets:")
        for name in ACADEMIC_DATASETS:
            print(f"  - {name}")
        print("\nText Datasets:")
        for name in TEXT_DATASETS:
            print(f"  - {name}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to download
    if args.dataset:
        # Single dataset
        all_datasets = {**MOLMO2_DATASETS, **PIXMO_DATASETS, **ACADEMIC_DATASETS, **TEXT_DATASETS}
        if args.dataset in all_datasets:
            download_dataset(args.dataset, all_datasets[args.dataset], output_dir, args.cache_dir)
        else:
            print(f"Error: Dataset '{args.dataset}' not found.")
            print(f"Use --list to see available datasets.")
            return
    
    elif args.stage:
        # Stage-specific datasets
        dataset_dict = STAGE_DATASETS[args.stage]
        print(f"\nDownloading {args.stage.upper()} stage datasets ({len(dataset_dict)} datasets)")
        download_all_datasets(output_dir, args.cache_dir, dataset_dict, args.skip_errors)
    
    elif args.molmo2_only:
        print(f"\nDownloading Molmo2 datasets only ({len(MOLMO2_DATASETS)} datasets)")
        download_all_datasets(output_dir, args.cache_dir, MOLMO2_DATASETS, args.skip_errors)
    
    elif args.pixmo_only:
        print(f"\nDownloading PixMo datasets only ({len(PIXMO_DATASETS)} datasets)")
        download_all_datasets(output_dir, args.cache_dir, PIXMO_DATASETS, args.skip_errors)
    
    elif args.academic_only:
        print(f"\nDownloading Academic datasets only ({len(ACADEMIC_DATASETS)} datasets)")
        download_all_datasets(output_dir, args.cache_dir, ACADEMIC_DATASETS, args.skip_errors)
    
    elif args.all:
        # All datasets
        all_datasets = {**MOLMO2_DATASETS, **PIXMO_DATASETS, **ACADEMIC_DATASETS, **TEXT_DATASETS}
        download_all_datasets(output_dir, args.cache_dir, all_datasets, args.skip_errors)
    
    else:
        parser.print_help()
        print("\nError: Please specify what to download (--all, --dataset, --stage, etc.)")


if __name__ == "__main__":
    main()
