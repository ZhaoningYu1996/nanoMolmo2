"""
Inspect downloaded Molmo2 datasets to understand their structure.

This script helps examine the actual data format before writing dataloaders.
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_parquet_file(file_path: Path, num_samples: int = 3) -> None:
    """
    Inspect a parquet file and print its structure.
    
    Args:
        file_path: Path to parquet file
        num_samples: Number of sample rows to display
    """
    logger.info(f"\nInspecting: {file_path}")
    logger.info("=" * 80)
    
    # Load dataset
    df = pd.read_parquet(file_path)
    
    # Basic info
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nColumn types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    # Sample data
    logger.info(f"\nFirst {num_samples} samples:")
    logger.info("-" * 80)
    
    for idx in range(min(num_samples, len(df))):
        logger.info(f"\n[Sample {idx + 1}]")
        row = df.iloc[idx]
        
        for col in df.columns:
            value = row[col]
            
            # Format value based on type
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=2)[:200]  # Truncate long values
                if len(json.dumps(value)) > 200:
                    value_str += "..."
            elif isinstance(value, str) and len(value) > 200:
                value_str = value[:200] + "..."
            else:
                value_str = str(value)
            
            logger.info(f"  {col}: {value_str}")


def inspect_dataset(dataset_path: Path, num_samples: int = 3) -> None:
    """
    Inspect all splits in a dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to show per split
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_path.name}")
    logger.info(f"{'='*80}")
    
    # Find all parquet files
    parquet_files = list(dataset_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {dataset_path}")
        return
    
    logger.info(f"Found {len(parquet_files)} split(s)")
    
    # Inspect each file
    for parquet_file in sorted(parquet_files):
        inspect_parquet_file(parquet_file, num_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Molmo2 dataset structure"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        nargs="?",
        help="Dataset name to inspect (if not specified, lists all datasets)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/molmo2_datasets",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to display per split"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run download_molmo2_data.py first to download datasets")
        return
    
    if args.dataset_name:
        # Inspect specific dataset
        dataset_path = data_dir / args.dataset_name
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return
        
        inspect_dataset(dataset_path, args.num_samples)
    else:
        # List all datasets
        logger.info("\nAvailable datasets:")
        logger.info("=" * 80)
        
        dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if not dataset_dirs:
            logger.warning("No datasets found")
            logger.info("Run download_molmo2_data.py to download datasets")
            return
        
        for dataset_dir in sorted(dataset_dirs):
            parquet_files = list(dataset_dir.glob("*.parquet"))
            logger.info(f"  {dataset_dir.name}: {len(parquet_files)} split(s)")
        
        logger.info("\nTo inspect a specific dataset, run:")
        logger.info(f"  python {__file__} <dataset_name>")


if __name__ == "__main__":
    main()
