"""
Download and prepare Molmo2 datasets for training.

This script downloads all datasets used in Molmo2 training:
- Pre-training: PixMo-Cap (dense captions), PixMo-Points, PixMo-Count
- SFT: Molmo2-VideoCapQA, Molmo2-VideoPoint, Molmo2-VideoTrack, etc.
- Long-context: Extended video datasets

Based on Molmo2 data release: https://huggingface.co/collections/allenai/molmo2-data
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset configurations based on Molmo2 paper and HuggingFace collection
MOLMO2_DATASETS = {
    # === Pre-training Stage Datasets (PixMo) ===
    "pretraining": {
        "pixmo-cap": {
            "hf_path": "allenai/pixmo-cap",
            "description": "Dense image captioning (200 words avg)",
            "task_type": "image_caption",
            "weight": 0.50  # Adjusted for complete dataset list
        },
        "pixmo-askmodelanything": {
            "hf_path": "allenai/pixmo-ask-model-anything",
            "description": "Human-authored image QA triplets (162k)",
            "task_type": "image_qa",
            "weight": 0.10
        },
        "pixmo-capqa": {
            "hf_path": "allenai/pixmo-cap-qa",
            "description": "Synthetic QA from captions",
            "task_type": "image_qa",
            "weight": 0.05
        },
        "pixmo-points": {
            "hf_path": "allenai/pixmo-points",
            "description": "Image pointing with referring expressions",
            "task_type": "image_pointing",
            "weight": 0.15
        },
        "pixmo-point-explanations": {
            "hf_path": "allenai/pixmo-point-explanations",
            "description": "Explanations with inline points",
            "task_type": "image_pointing",
            "weight": 0.05
        },
        "pixmo-docs": {
            "hf_path": "allenai/pixmo-docs",
            "description": "Charts, tables, diagrams QA",
            "task_type": "document_qa",
            "weight": 0.05
        },
        "pixmo-clocks": {
            "hf_path": "allenai/pixmo-clocks",
            "description": "Virtual watch faces and time annotations",
            "task_type": "clock_reading",
            "weight": 0.05
        },
        "pixmo-count": {
            "hf_path": "allenai/pixmo-count",
            "description": "Object counting QA",
            "task_type": "counting",
            "weight": 0.05
        },
    },
    
    # === SFT Stage Datasets (Molmo2 original datasets) ===
    "sft": {
        "molmo2-cap": {
            "hf_path": "allenai/Molmo2-Cap",
            "description": "Image dense captioning for SFT",
            "task_type": "image_caption",
        },
        "molmo2-videocapqa": {
            "hf_path": "allenai/Molmo2-VideoCapQA",
            "description": "Video captioning and QA",
            "task_type": "video_qa",
        },
        "molmo2-videosubtitleqa": {
            "hf_path": "allenai/Molmo2-VideoSubtitleQA",
            "description": "Video QA with subtitle context",
            "task_type": "video_qa",
        },
        "molmo2-askmodelanything": {
            "hf_path": "allenai/Molmo2-AskModelAnything",
            "description": "Human-authored image QA",
            "task_type": "image_qa",
        },
        "molmo2-videopoint": {
            "hf_path": "allenai/Molmo2-VideoPoint",
            "description": "Video temporal pointing",
            "task_type": "video_pointing",
        },
        "molmo2-videotrack": {
            "hf_path": "allenai/Molmo2-VideoTrack",
            "description": "Video object tracking",
            "task_type": "video_tracking",
        },
        "molmo2-multiimageqa": {
            "hf_path": "allenai/Molmo2-MultiImageQA",
            "description": "Multi-image question answering",
            "task_type": "multi_image_qa",
        },
        "molmo2-synmultiimageqa": {
            "hf_path": "allenai/Molmo2-SynMultiImageQA",
            "description": "Synthetic multi-image QA",
            "task_type": "multi_image_qa",
        },
        "molmo2-multiimagepoint": {
            "hf_path": "allenai/Molmo2-MultiImagePoint",
            "description": "Multi-image pointing",
            "task_type": "multi_image_pointing",
        },
    },
    
    # === Evaluation Datasets ===
    "eval": {
        "molmo2-capeval": {
            "hf_path": "allenai/Molmo2-CapEval",
            "description": "Caption evaluation benchmark",
            "task_type": "image_caption",
        },
        "molmo2-videopointeval": {
            "hf_path": "allenai/Molmo2-VideoPointEval",
            "description": "Video pointing evaluation",
            "task_type": "video_pointing",
        },
        "molmo2-videocounteval": {
            "hf_path": "allenai/Molmo2-VideoCountEval",
            "description": "Video counting evaluation",
            "task_type": "video_counting",
        },
        "molmo2-videotrackeval": {
            "hf_path": "allenai/Molmo2-VideoTrackEval",
            "description": "Video tracking evaluation",
            "task_type": "video_tracking",
        },
    }
}


class Molmo2DataDownloader:
    """Download and prepare Molmo2 datasets."""
    
    def __init__(
        self,
        data_dir: str = "./data/molmo2_datasets",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to save processed datasets
            cache_dir: HuggingFace cache directory (uses default if None)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir or 'default HF cache'}")
    
    def download_dataset(
        self,
        dataset_name: str,
        hf_path: str,
        split: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        """
        Download a single dataset from HuggingFace.
        
        Args:
            dataset_name: Local name for the dataset
            hf_path: HuggingFace dataset path
            split: Dataset split to download (None = all splits)
            streaming: Whether to use streaming mode
        """
        logger.info(f"Downloading {dataset_name} from {hf_path}...")
        
        try:
            # Download dataset
            dataset = load_dataset(
                hf_path,
                split=split,
                streaming=streaming,
                cache_dir=self.cache_dir,
            )
            
            # Save to local directory in parquet format for efficient loading
            save_path = self.data_dir / dataset_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            if not streaming:
                if isinstance(dataset, dict):
                    # Multiple splits
                    for split_name, split_data in dataset.items():
                        output_file = save_path / f"{split_name}.parquet"
                        split_data.to_parquet(str(output_file))
                        logger.info(f"  Saved {split_name} to {output_file}")
                else:
                    # Single split
                    output_file = save_path / "train.parquet"
                    dataset.to_parquet(str(output_file))
                    logger.info(f"  Saved to {output_file}")
            
            logger.info(f"✓ Successfully downloaded {dataset_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_name}: {str(e)}")
            raise
    
    def download_stage(self, stage: str, dataset_names: Optional[List[str]] = None) -> None:
        """
        Download all datasets for a training stage.
        
        Args:
            stage: Training stage ("pretraining", "sft", or "eval")
            dataset_names: Specific datasets to download (None = all in stage)
        """
        if stage not in MOLMO2_DATASETS:
            raise ValueError(f"Unknown stage: {stage}. Choose from {list(MOLMO2_DATASETS.keys())}")
        
        stage_datasets = MOLMO2_DATASETS[stage]
        
        if dataset_names:
            # Download specific datasets
            datasets_to_download = {
                name: config for name, config in stage_datasets.items()
                if name in dataset_names
            }
        else:
            # Download all datasets in stage
            datasets_to_download = stage_datasets
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {stage.upper()} stage datasets")
        logger.info(f"Total: {len(datasets_to_download)} datasets")
        logger.info(f"{'='*60}\n")
        
        for name, config in datasets_to_download.items():
            logger.info(f"\n[{name}] {config['description']}")
            self.download_dataset(
                dataset_name=name,
                hf_path=config["hf_path"],
            )
    
    def download_all(self, include_eval: bool = False) -> None:
        """
        Download all Molmo2 training datasets.
        
        Args:
            include_eval: Whether to include evaluation datasets
        """
        # Download pre-training datasets
        self.download_stage("pretraining")
        
        # Download SFT datasets
        self.download_stage("sft")
        
        # Optionally download evaluation datasets
        if include_eval:
            self.download_stage("eval")
        
        logger.info("\n" + "="*60)
        logger.info("✓ All datasets downloaded successfully!")
        logger.info("="*60)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary of downloaded datasets."""
        logger.info("\nDataset Summary:")
        logger.info("-" * 60)
        
        for stage in ["pretraining", "sft", "eval"]:
            stage_dir = self.data_dir
            if not stage_dir.exists():
                continue
            
            logger.info(f"\n{stage.upper()} Stage:")
            for dataset_name in MOLMO2_DATASETS[stage].keys():
                dataset_path = self.data_dir / dataset_name
                if dataset_path.exists():
                    files = list(dataset_path.glob("*.parquet"))
                    logger.info(f"  ✓ {dataset_name}: {len(files)} split(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Molmo2 datasets for training"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretraining", "sft", "eval", "all"],
        default="all",
        help="Which stage datasets to download"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Specific datasets to download (if not specified, downloads all in stage)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/molmo2_datasets",
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Include evaluation datasets when downloading all"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = Molmo2DataDownloader(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    
    # Download datasets
    if args.stage == "all":
        downloader.download_all(include_eval=args.include_eval)
    else:
        downloader.download_stage(args.stage, args.datasets)


if __name__ == "__main__":
    main()
