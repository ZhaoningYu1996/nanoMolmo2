"""
Download and prepare ALL Molmo2 training datasets.

Complete dataset downloader based on Molmo2 technical report:
- Pre-training: 5 datasets (PixMo + Tulu)
- SFT: 100+ datasets (Molmo2 originals + PixMo + Academic datasets)
- Long-context: Same as SFT with extended sequences

Handles:
- HuggingFace datasets (automatic)
- Academic datasets from various sources
- Manual download instructions for datasets requiring it

Reference: Molmo2 Technical Report, Table 13
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from datasets import load_dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# === DATASET CONFIGURATIONS FROM MOLMO2 PAPER ===

# Pre-training datasets (Table 13 - trained for 32k steps)
PRETRAINING_DATASETS = {
    # PixMo datasets
    "pixmo-cap": {
        "hf_path": "allenai/pixmo-cap",
        "description": "Dense image captioning (~200 words avg)",
        "source": "huggingface",
    },
    "pixmo-points": {
        "hf_path": "allenai/pixmo-points",
        "description": "Image pointing with referring expressions",
        "source": "huggingface",
    },
    "pixmo-count": {
        "hf_path": "allenai/pixmo-count",
        "description": "Object counting QA",
        "source": "huggingface",
    },
    
    # Academic datasets
    "cosyn-point": {
        "hf_path": None,  # Requires manual download
        "description": "Synthetic pointing data from CoSyn",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "instructions": "Clone CoSyn repo and follow setup instructions"
    },
    
    # Text data
    "tulu": {
        "hf_path": "allenai/tulu-v2-sft-mixture",
        "description": "Text-only instruction data",
        "source": "huggingface",
    },
}


# SFT stage datasets (100+ datasets from Table 13)
SFT_DATASETS = {
    # === Molmo2 Original Datasets (Human-annotated + Synthetic) ===
    "molmo2-cap": {
        "hf_path": "allenai/Molmo2-Cap",
        "description": "Video dense captioning (104k video-level, 431k clip-level)",
        "source": "huggingface",
        "size": "100k examples"
    },
    "molmo2-askmodelanything": {
        "hf_path": "allenai/Molmo2-AskModelAnything",
        "description": "Human-authored video QA",
        "source": "huggingface",
        "size": "43k examples"
    },
    "molmo2-capqa": {
        "hf_path": "allenai/Molmo2-VideoCapQA",
        "description": "Synthetic QA from video captions",
        "source": "huggingface",
        "size": "1M QA pairs"
    },
    "molmo2-subtitleqa": {
        "hf_path": "allenai/Molmo2-VideoSubtitleQA",
        "description": "Video QA with subtitle context",
        "source": "huggingface",
        "size": "300k QA pairs"
    },
    "molmo2-videopoint": {
        "hf_path": "allenai/Molmo2-VideoPoint",
        "description": "Video temporal pointing (novel)",
        "source": "huggingface",
        "size": "330k examples"
    },
    "molmo2-videotrack": {
        "hf_path": "allenai/Molmo2-VideoTrack",
        "description": "Video object tracking (novel)",
        "source": "huggingface",
        "size": "220k examples"
    },
    "molmo2-multiimageqa": {
        "hf_path": "allenai/Molmo2-MultiImageQA",
        "description": "Multi-image QA",
        "source": "huggingface",
        "size": "45k examples"
    },
    "molmo2-synmultiimageqa": {
        "hf_path": "allenai/Molmo2-SynMultiImageQA",
        "description": "Synthetic multi-image QA",
        "source": "huggingface",
        "size": "188k examples"
    },
    "molmo2-multiimagepoint": {
        "hf_path": "allenai/Molmo2-MultiImagePoint",
        "description": "Multi-image pointing",
        "source": "huggingface",
        "size": "470k examples"
    },
    
    # === PixMo Datasets ===
    "pixmo-cap": {
        "hf_path": "allenai/pixmo-cap",
        "description": "Dense image captioning",
        "source": "huggingface",
        "size": "710k examples"
    },
    "pixmo-askmodelanything": {
        "hf_path": "allenai/pixmo-ask-model-anything",
        "description": "Human-authored image QA",
        "source": "huggingface",
        "size": "71k examples"
    },
    "pixmo-capqa": {
        "hf_path": "allenai/pixmo-cap-qa",
        "description": "Synthetic QA from captions",
        "source": "huggingface",
        "size": "190k examples"
    },
    "pixmo-clocks": {
        "hf_path": "allenai/pixmo-clocks",
        "description": "Clock reading",
        "source": "huggingface",
        "size": "800k examples"
    },
    
    # === Text Data ===
    "tulu": {
        "hf_path": "allenai/tulu-v2-sft-mixture",
        "description": "Text-only instruction data",
        "source": "huggingface",
        "size": "980k examples"
    },
    
    # === Academic Image Datasets ===
    "llava-665k-multi": {
        "hf_path": "liuhaotian/LLaVA-Instruct-150K",
        "description": "LLaVA instruction-following",
        "source": "huggingface",
        "size": "2.5M examples"
    },
    "tallyqa": {
        "hf_path": "HuggingFaceM4/TallyQA",
        "description": "Counting QA",
        "source": "huggingface",
        "size": "250k examples"
    },
    "vqa-v2": {
        "hf_path": "HuggingFaceM4/VQAv2",
        "description": "Visual Question Answering v2",
        "source": "huggingface",
        "size": "440k examples"
    },
    "docvqa": {
        "hf_path": "naver-clova-ix/docvqa",
        "description": "Document VQA",
        "source": "huggingface",
        "size": "39k examples"
    },
    "textvqa": {
        "hf_path": "textvqa/textvqa",
        "description": "Text-based VQA",
        "source": "huggingface",
        "size": "35k examples"
    },
    "chartqa": {
        "hf_path": "HuggingFaceM4/ChartQA",
        "description": "Chart understanding",
        "source": "huggingface",
        "size": "28k examples"
    },
    "st-vqa": {
        "hf_path": "naver-clova-ix/st-vqa",
        "description": "Scene text VQA",
        "source": "huggingface",
        "size": "25k examples"
    },
    "infographicvqa": {
        "hf_path": "naver-clova-ix/infographicvqa",
        "description": "Infographic VQA",
        "source": "huggingface",
        "size": "24k examples"
    },
    "ai2d": {
        "hf_path": "allenai/ai2d",
        "description": "Diagram understanding",
        "source": "huggingface",
        "size": "15k examples"
    },
    "nlvr2": {
        "hf_path": "HuggingFaceM4/NLVR2",
        "description": "Natural language visual reasoning",
        "source": "huggingface",
        "size": "86k examples"
    },
    "a-okvqa": {
        "hf_path": "HuggingFaceM4/A-OKVQA",
        "description": "Knowledge-based VQA",
        "source": "huggingface",
        "size": "34k examples"
    },
    "ok-vqa": {
        "hf_path": "HuggingFaceM4/OK-VQA",
        "description": "Outside knowledge VQA",
        "source": "huggingface",
        "size": "9k examples"
    },
    "scienceqa": {
        "hf_path": "derek-thomas/ScienceQA",
        "description": "Science question answering",
        "source": "huggingface",
        "size": "6.2k examples"
    },
    "spot-the-difference": {
        "hf_path": None,
        "description": "Visual difference detection",
        "source": "manual",
        "url": "https://github.com/harsh19/spot-the-difference",
        "size": "7.5k examples"
    },
    
    # === Chart/Table/Diagram Datasets ===
    "plotqa": {
        "hf_path": None,
        "description": "Plot understanding",
        "source": "manual",
        "url": "https://github.com/NiteshMethani/PlotQA",
        "size": "160k examples"
    },
    "dvqa": {
        "hf_path": None,
        "description": "Bar chart QA",
        "source": "manual",
        "url": "https://github.com/kushalkafle/DVQA_dataset",
        "size": "200k examples"
    },
    "figureqa": {
        "hf_path": None,
        "description": "Figure understanding",
        "source": "manual",
        "url": "https://www.microsoft.com/en-us/research/project/figureqa-dataset/",
        "size": "100k examples"
    },
    "tabwmp": {
        "hf_path": None,
        "description": "Table-based math word problems",
        "source": "manual",
        "url": "https://promptpg.github.io/",
        "size": "23k examples"
    },
    
    # === CoSyn Synthetic Datasets (from Molmo original paper) ===
    "cosyn-chart": {
        "hf_path": None,
        "description": "Synthetic chart QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "1.1M examples"
    },
    "cosyn-doc": {
        "hf_path": None,
        "description": "Synthetic document QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "71k examples"
    },
    "cosyn-table": {
        "hf_path": None,
        "description": "Synthetic table QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "420k examples"
    },
    "cosyn-diagram": {
        "hf_path": None,
        "description": "Synthetic diagram QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "35k examples"
    },
    "cosyn-math": {
        "hf_path": None,
        "description": "Synthetic math QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "67k examples"
    },
    "cosyn-music": {
        "hf_path": None,
        "description": "Synthetic music notation QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "12k examples"
    },
    "cosyn-chemical": {
        "hf_path": None,
        "description": "Synthetic chemistry diagram QA",
        "source": "manual",
        "url": "https://github.com/allenai/CoSyn",
        "size": "8.9k examples"
    },
    
    # === Video Understanding Datasets ===
    "video-localized-narratives": {
        "hf_path": None,
        "description": "Video narration with mouse traces",
        "source": "manual",
        "url": "https://google.github.io/video-localized-narratives/",
        "size": "56k examples"
    },
    "tgif": {
        "hf_path": None,
        "description": "Animated GIF QA",
        "source": "manual",
        "url": "https://github.com/raingo/TGIF-Release",
        "size": "63k examples"
    },
    "tvqa": {
        "hf_path": None,
        "description": "TV show QA",
        "source": "manual",
        "url": "https://tvqa.cs.unc.edu/",
        "size": "120k examples"
    },
    "paxion": {
        "hf_path": None,
        "description": "Action recognition",
        "source": "manual",
        "url": "https://github.com/google-research/google-research/tree/master/paxion",
        "size": "440k examples"
    },
    "moments-in-time": {
        "hf_path": None,
        "description": "Action recognition in videos",
        "source": "manual",
        "url": "http://moments.csail.mit.edu/",
        "size": "710k examples"
    },
    "kinetics": {
        "hf_path": None,
        "description": "Human action videos",
        "source": "manual",
        "url": "https://www.deepmind.com/open-source/kinetics",
        "size": "420k examples"
    },
    "ego4d": {
        "hf_path": None,
        "description": "Egocentric video understanding",
        "source": "manual",
        "url": "https://ego4d-data.org/",
        "size": "53k examples"
    },
    "epic-kitchens": {
        "hf_path": None,
        "description": "Egocentric kitchen activities",
        "source": "manual",
        "url": "https://epic-kitchens.github.io/",
        "size": "37k examples"
    },
    "coin": {
        "hf_path": None,
        "description": "Comprehensive instructional videos",
        "source": "manual",
        "url": "https://coin-dataset.github.io/",
        "size": "30k examples"
    },
    "how2qa": {
        "hf_path": None,
        "description": "Instructional video QA",
        "source": "manual",
        "url": "https://github.com/ych133/How2R-and-How2QA",
        "size": "25k examples"
    },
    "activitynet": {
        "hf_path": None,
        "description": "Activity recognition and captioning",
        "source": "manual",
        "url": "http://activity-net.org/",
        "size": "21k examples"
    },
    "funqa": {
        "hf_path": None,
        "description": "Creative video understanding",
        "source": "manual",
        "url": "https://github.com/jingkang50/FunQA",
        "size": "200k examples"
    },
    "clevrer": {
        "hf_path": None,
        "description": "Causal video reasoning",
        "source": "manual",
        "url": "http://clevrer.csail.mit.edu/",
        "size": "20k examples"
    },
    "star": {
        "hf_path": None,
        "description": "Situated reasoning in videos",
        "source": "manual",
        "url": "https://bobbywu.com/STAR/",
        "size": "10k examples"
    },
    "youcook2": {
        "hf_path": None,
        "description": "Cooking video understanding",
        "source": "manual",
        "url": "http://youcook2.eecs.umich.edu/",
        "size": "18k examples"
    },
    "sutd-trafficqa": {
        "hf_path": None,
        "description": "Traffic video QA",
        "source": "manual",
        "url": "https://github.com/SUTDCV/SUTD-TrafficQA",
        "size": "10k examples"
    },
    "cinepile": {
        "hf_path": None,
        "description": "Long movie understanding",
        "source": "manual",
        "url": "https://github.com/princeton-nlp/CinePile",
        "size": "300k examples"
    },
    "charades-sta": {
        "hf_path": None,
        "description": "Spatio-temporal action localization",
        "source": "manual",
        "url": "https://prior.allenai.org/projects/charades",
        "size": "12k examples"
    },
    "qvhighlights": {
        "hf_path": None,
        "description": "Video highlight detection and QA",
        "source": "manual",
        "url": "https://github.com/jayleicn/moment_detr",
        "size": "7k examples"
    },
    "motionbench": {
        "hf_path": None,
        "description": "Motion understanding",
        "source": "manual",
        "url": "https://github.com/WeijieChen2017/MotionBench",
        "size": "5k examples"
    },
    "countix": {
        "hf_path": None,
        "description": "Repetitive action counting",
        "source": "manual",
        "url": "https://github.com/pedro-morgado/AVSpatialAlignment",
        "size": "4.4k examples"
    },
    "next-qa": {
        "hf_path": None,
        "description": "Video QA with reasoning",
        "source": "manual",
        "url": "https://github.com/doc-doc/NExT-QA",
        "size": "34k examples"
    },
    "sports-qa": {
        "hf_path": None,
        "description": "Sports video QA",
        "source": "manual",
        "url": "https://github.com/google-research/google-research/tree/master/sports_qa",
        "size": "56k examples"
    },
    "intentqa": {
        "hf_path": None,
        "description": "Intent understanding in videos",
        "source": "manual",
        "url": "https://github.com/JoseponLee/IntentQA",
        "size": "24k examples"
    },
    "newsvideoqa": {
        "hf_path": None,
        "description": "News video QA",
        "source": "manual",
        "url": "https://github.com/WENGSYX/NewsVideoQA",
        "size": "8.4k examples"
    },
    "roadtextvqa": {
        "hf_path": None,
        "description": "Driving scene text VQA",
        "source": "manual",
        "url": "https://github.com/zhuang-li/RoadTextVQA",
        "size": "8.4k examples"
    },
    "perceptiontest": {
        "hf_path": None,
        "description": "Video perception benchmark",
        "source": "manual",
        "url": "https://github.com/google-deepmind/perception_test",
        "size": "2k examples"
    },
    "social-iq-2": {
        "hf_path": None,
        "description": "Social intelligence in videos",
        "source": "manual",
        "url": "https://github.com/abwilf/Social-IQ-2.0",
        "size": "0.79k examples"
    },
    
    # === Video Pointing Datasets (Academic) ===
    "mevis": {
        "hf_path": None,
        "description": "Referring video object segmentation",
        "source": "manual",
        "url": "https://henghuiding.github.io/MeViS/",
        "size": "20k examples"
    },
    "refvos": {
        "hf_path": None,
        "description": "Referring video object segmentation",
        "source": "manual",
        "url": "https://github.com/wudongming97/Ref-YouTube-VOS",
        "size": "11k examples"
    },
    "lv-vis": {
        "hf_path": None,
        "description": "Long video instance segmentation",
        "source": "manual",
        "url": "https://github.com/haochenheheda/LVVIS",
        "size": "11k examples"
    },
    "ovis": {
        "hf_path": None,
        "description": "Occluded video instance segmentation",
        "source": "manual",
        "url": "https://github.com/QIU023/OVIS",
        "size": "880 examples"
    },
    "burst": {
        "hf_path": None,
        "description": "Unifying video segmentation",
        "source": "manual",
        "url": "https://github.com/Ali2500/BURST-benchmark",
        "size": "680 examples"
    },
    "ref-davis17": {
        "hf_path": None,
        "description": "Referring video segmentation",
        "source": "manual",
        "url": "https://www.vision.rwth-aachen.de/page/davis2017",
        "size": "450 examples"
    },
    
    # === Video Tracking Datasets (Academic) ===
    "vicas": {
        "hf_path": None,
        "description": "Video caption and segmentation",
        "source": "manual",
        "url": "https://github.com/zhouqunbing/ViCaS",
        "size": "130k examples"
    },
    "revos": {
        "hf_path": None,
        "description": "Referring expression video segmentation",
        "source": "manual",
        "url": "https://github.com/wudongming97/ReVOS",
        "size": "82k examples"
    },
    "trackingnet": {
        "hf_path": None,
        "description": "Large-scale object tracking",
        "source": "manual",
        "url": "https://tracking-net.org/",
        "size": "29k examples"
    },
    "ref-youtube-vos": {
        "hf_path": None,
        "description": "Referring YouTube-VOS",
        "source": "manual",
        "url": "https://youtube-vos.org/dataset/rvos/",
        "size": "26k examples"
    },
    "vasttrack": {
        "hf_path": None,
        "description": "Large-scale visual tracking",
        "source": "manual",
        "url": "https://github.com/HengLan/VastTrack",
        "size": "93k examples"
    },
    "got-10k": {
        "hf_path": None,
        "description": "Generic object tracking",
        "source": "manual",
        "url": "http://got-10k.aitestunion.com/",
        "size": "18k examples"
    },
    "webuav": {
        "hf_path": None,
        "description": "UAV tracking benchmark",
        "source": "manual",
        "url": "https://github.com/983632847/WebUAV-3M",
        "size": "6.3k examples"
    },
    "lasot": {
        "hf_path": None,
        "description": "Large-scale single object tracking",
        "source": "manual",
        "url": "https://vision.cs.stonybrook.edu/~lasot/",
        "size": "2.2k examples"
    },
    "tnl2k": {
        "hf_path": None,
        "description": "Tracking with natural language",
        "source": "manual",
        "url": "https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit",
        "size": "1.8k examples"
    },
    "webuot": {
        "hf_path": None,
        "description": "Underwater object tracking",
        "source": "manual",
        "url": "https://github.com/983632847/WebUOT-1M",
        "size": "1.5k examples"
    },
    "lvos-v2": {
        "hf_path": None,
        "description": "Long-term video object segmentation v2",
        "source": "manual",
        "url": "https://lingyihongfd.github.io/lvos.github.io/",
        "size": "1.2k examples"
    },
    "youtube-vis": {
        "hf_path": None,
        "description": "YouTube video instance segmentation",
        "source": "manual",
        "url": "https://youtube-vos.org/dataset/vis/",
        "size": "1.4k examples"
    },
    "moca-video": {
        "hf_path": None,
        "description": "Motion and content aware segmentation",
        "source": "manual",
        "url": "https://henghuiding.github.io/MoCA/",
        "size": "0.4k examples"
    },
}


# Evaluation datasets
EVAL_DATASETS = {
    "molmo2-capeval": {
        "hf_path": "allenai/Molmo2-CapEval",
        "description": "Caption evaluation benchmark",
        "source": "huggingface",
    },
    "molmo2-videopointeval": {
        "hf_path": "allenai/Molmo2-VideoPointEval",
        "description": "Video pointing evaluation",
        "source": "huggingface",
    },
    "molmo2-videocounteval": {
        "hf_path": "allenai/Molmo2-VideoCountEval",
        "description": "Video counting evaluation",
        "source": "huggingface",
    },
    "molmo2-videotrackeval": {
        "hf_path": "allenai/Molmo2-VideoTrackEval",
        "description": "Video tracking evaluation",
        "source": "huggingface",
    },
}


class Molmo2DataDownloader:
    """Download and prepare complete Molmo2 dataset collection."""
    
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
    
    def download_huggingface_dataset(
        self,
        dataset_name: str,
        hf_path: str,
        split: Optional[str] = None,
    ) -> None:
        """Download a dataset from HuggingFace."""
        logger.info(f"Downloading {dataset_name} from {hf_path}...")
        
        try:
            dataset = load_dataset(
                hf_path,
                split=split,
                cache_dir=self.cache_dir,
            )
            
            save_path = self.data_dir / dataset_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    output_file = save_path / f"{split_name}.parquet"
                    split_data.to_parquet(str(output_file))
                    logger.info(f"  Saved {split_name} to {output_file}")
            else:
                output_file = save_path / "train.parquet"
                dataset.to_parquet(str(output_file))
                logger.info(f"  Saved to {output_file}")
            
            logger.info(f"✓ Successfully downloaded {dataset_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_name}: {str(e)}")
            raise
    
    def download_stage(
        self,
        stage: str,
        dataset_names: Optional[List[str]] = None,
        skip_manual: bool = True
    ) -> None:
        """
        Download datasets for a training stage.
        
        Args:
            stage: Training stage ("pretraining", "sft", or "eval")
            dataset_names: Specific datasets to download (None = all)
            skip_manual: Whether to skip datasets requiring manual download
        """
        # Get dataset collection for stage
        if stage == "pretraining":
            stage_datasets = PRETRAINING_DATASETS
        elif stage == "sft":
            stage_datasets = SFT_DATASETS
        elif stage == "eval":
            stage_datasets = EVAL_DATASETS
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        if dataset_names:
            datasets_to_download = {
                name: config for name, config in stage_datasets.items()
                if name in dataset_names
            }
        else:
            datasets_to_download = stage_datasets
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {stage.upper()} stage datasets")
        logger.info(f"Total: {len(datasets_to_download)} datasets")
        logger.info(f"{'='*60}\n")
        
        hf_count = 0
        manual_count = 0
        
        for name, config in datasets_to_download.items():
            if config["source"] == "huggingface":
                logger.info(f"\n[{name}] {config['description']}")
                self.download_huggingface_dataset(name, config["hf_path"])
                hf_count += 1
            elif config["source"] == "manual":
                if skip_manual:
                    logger.info(f"\n[{name}] SKIPPED (requires manual download)")
                    logger.info(f"  URL: {config.get('url', 'N/A')}")
                    logger.info(f"  Instructions: {config.get('instructions', 'See dataset page')}")
                    manual_count += 1
                else:
                    logger.warning(f"\n[{name}] Requires manual download:")
                    logger.warning(f"  URL: {config.get('url', 'N/A')}")
                    logger.warning(f"  Instructions: {config.get('instructions', 'See dataset page')}")
                    manual_count += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Downloaded {hf_count} datasets from HuggingFace")
        if manual_count > 0:
            logger.info(f"⚠ {manual_count} datasets require manual download")
            logger.info(f"  See instructions above or use --show-manual-instructions")
        logger.info(f"{'='*60}")
    
    def download_all(
        self,
        include_eval: bool = False,
        skip_manual: bool = True
    ) -> None:
        """Download all Molmo2 training datasets."""
        self.download_stage("pretraining", skip_manual=skip_manual)
        self.download_stage("sft", skip_manual=skip_manual)
        
        if include_eval:
            self.download_stage("eval", skip_manual=skip_manual)
        
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print download summary."""
        logger.info("\n" + "="*60)
        logger.info("Dataset Download Summary")
        logger.info("="*60)
        
        for stage_name, stage_datasets in [
            ("Pre-training", PRETRAINING_DATASETS),
            ("SFT", SFT_DATASETS),
            ("Evaluation", EVAL_DATASETS),
        ]:
            logger.info(f"\n{stage_name} Stage:")
            downloaded = 0
            manual = 0
            
            for dataset_name, config in stage_datasets.items():
                dataset_path = self.data_dir / dataset_name
                if dataset_path.exists():
                    files = list(dataset_path.glob("*.parquet"))
                    logger.info(f"  ✓ {dataset_name}: {len(files)} split(s)")
                    downloaded += 1
                elif config["source"] == "manual":
                    manual += 1
            
            logger.info(f"\n  Total: {downloaded} downloaded, {manual} manual")
    
    def show_manual_instructions(self, stage: Optional[str] = None) -> None:
        """Show instructions for all datasets requiring manual download."""
        logger.info("\n" + "="*60)
        logger.info("Datasets Requiring Manual Download")
        logger.info("="*60)
        
        datasets_to_show = {}
        if stage == "pretraining":
            datasets_to_show = PRETRAINING_DATASETS
        elif stage == "sft":
            datasets_to_show = SFT_DATASETS
        elif stage == "eval":
            datasets_to_show = EVAL_DATASETS
        else:
            # Show all
            datasets_to_show = {**PRETRAINING_DATASETS, **SFT_DATASETS, **EVAL_DATASETS}
        
        manual_datasets = {
            name: config for name, config in datasets_to_show.items()
            if config["source"] == "manual"
        }
        
        if not manual_datasets:
            logger.info("\nNo manual downloads required for specified stage.")
            return
        
        for name, config in manual_datasets.items():
            logger.info(f"\n[{name}]")
            logger.info(f"  Description: {config['description']}")
            logger.info(f"  Size: {config.get('size', 'N/A')}")
            logger.info(f"  URL: {config.get('url', 'N/A')}")
            logger.info(f"  Instructions: {config.get('instructions', 'See dataset page for download instructions')}")


def main():
    parser = argparse.ArgumentParser(
        description="Download complete Molmo2 dataset collection"
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
        help="Specific datasets to download"
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
        help="Include evaluation datasets"
    )
    parser.add_argument(
        "--skip-manual",
        action="store_true",
        default=True,
        help="Skip datasets requiring manual download (default: True)"
    )
    parser.add_argument(
        "--show-manual-instructions",
        action="store_true",
        help="Show instructions for manual downloads and exit"
    )
    
    args = parser.parse_args()
    
    downloader = Molmo2DataDownloader(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    
    if args.show_manual_instructions:
        downloader.show_manual_instructions(
            stage=args.stage if args.stage != "all" else None
        )
        return
    
    if args.stage == "all":
        downloader.download_all(
            include_eval=args.include_eval,
            skip_manual=args.skip_manual
        )
    else:
        downloader.download_stage(
            args.stage,
            args.datasets,
            skip_manual=args.skip_manual
        )


if __name__ == "__main__":
    main()
