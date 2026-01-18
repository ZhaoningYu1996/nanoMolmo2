"""
Molmo2 Dataloader - Pure PyTorch

Usage:
    from data import create_dataloader
    
    loader = create_dataloader(batch_size=32)
    for batch in loader:
        images = batch["images"]   # List[List[Image]]
        texts = batch["texts"]     # List[str]
        weights = batch["weights"] # Tensor
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from io import BytesIO
from urllib.request import urlopen
import ssl

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image
import pyarrow.parquet as pq


@dataclass
class Sample:
    """Single training sample."""
    images: List[Image.Image] = field(default_factory=list)
    text: str = ""
    task: str = "caption"
    weight: float = 1.0


def read_parquet(path: str) -> List[Dict]:
    """Read parquet file to list of dicts."""
    return pq.read_table(path).to_pylist()


def download_image(url: str, cache_dir: Path) -> Optional[Image.Image]:
    """Download image with caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
    
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except:
            pass
    
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urlopen(url, timeout=10, context=ctx) as r:
            img = Image.open(BytesIO(r.read())).convert("RGB")
            img.save(cache_path, "JPEG")
            return img
    except:
        return None


class ImageCaptionDataset(Dataset):
    """pixmo-cap: Image captioning."""
    
    def __init__(self, data_dir: Path, max_samples: int = None):
        self.cache = data_dir / "cache"
        data = read_parquet(str(data_dir / "train.parquet"))
        self.data = data[:max_samples] if max_samples else data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Sample]:
        row = self.data[idx]
        img = download_image(row["image_url"], self.cache)
        if not img: return None
        return Sample([img], row["caption"], "caption", 1.0)


class PointingDataset(Dataset):
    """pixmo-points: Point to objects."""
    
    def __init__(self, data_dir: Path, max_samples: int = None):
        self.cache = data_dir / "cache"
        data = read_parquet(str(data_dir / "train.parquet"))
        self.data = data[:max_samples] if max_samples else data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Sample]:
        row = self.data[idx]
        img = download_image(row["image_url"], self.cache)
        if not img: return None
        
        pts = " ".join(f"<point {p.get('x',0):.2f},{p.get('y',0):.2f}>" 
                       for p in row.get("points", []) if isinstance(p, dict))
        text = f"Q: Point to {row.get('label', 'objects')}\nA: {pts}"
        return Sample([img], text, "pointing", 2.0)


class CountingDataset(Dataset):
    """pixmo-count: Count objects."""
    
    def __init__(self, data_dir: Path, max_samples: int = None):
        self.cache = data_dir / "cache"
        data = read_parquet(str(data_dir / "train.parquet"))
        self.data = data[:max_samples] if max_samples else data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Sample]:
        row = self.data[idx]
        img = download_image(row["image_url"], self.cache)
        if not img: return None
        text = f"Q: How many {row.get('label', 'objects')}?\nA: {row.get('count', 0)}"
        return Sample([img], text, "counting", 1.5)


class SyntheticPointingDataset(Dataset):
    """cosyn-point: Synthetic pointing data."""
    
    def __init__(self, data_dir: Path, max_samples: int = None):
        path = data_dir / "train.parquet"
        if not path.exists():
            path = data_dir / "validation.parquet"
        data = read_parquet(str(path))
        self.data = data[:max_samples] if max_samples else data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Sample]:
        row = self.data[idx]
        img_data = row.get("image")
        if not img_data: return None
        
        try:
            if isinstance(img_data, dict):
                img = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
            else:
                img = Image.open(BytesIO(img_data)).convert("RGB")
        except:
            return None
        
        q = row.get("questions", [""])[0] if row.get("questions") else ""
        a = row.get("answer_points", [""])[0] if row.get("answer_points") else ""
        return Sample([img], f"Q: {q}\nA: {a}", "pointing", 2.0)


class TextDataset(Dataset):
    """tulu: Text-only NLP data."""
    
    def __init__(self, data_dir: Path, max_samples: int = None):
        data = read_parquet(str(data_dir / "train.parquet"))
        self.data = data[:max_samples] if max_samples else data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx) -> Sample:
        msgs = self.data[idx].get("messages", [])
        text = "\n".join(f"{m.get('role','')}: {m.get('content','')}" 
                        for m in msgs if isinstance(m, dict))
        return Sample([], text, "text", 1.0)


def collate(batch: List[Optional[Sample]]) -> Dict[str, Any]:
    """Collate samples, filter None."""
    batch = [s for s in batch if s is not None]
    if not batch:
        return {"images": [], "texts": [], "tasks": [], "weights": torch.tensor([])}
    return {
        "images": [s.images for s in batch],
        "texts": [s.text for s in batch],
        "tasks": [s.task for s in batch],
        "weights": torch.tensor([s.weight for s in batch]),
    }


# Dataset registry: (class, sampling ratio)
DATASETS = {
    "pixmo-cap": (ImageCaptionDataset, 0.60),
    "pixmo-points": (PointingDataset, 0.15),
    "pixmo-count": (CountingDataset, 0.10),
    "cosyn-point": (SyntheticPointingDataset, 0.05),
    "tulu-3-sft-mixture": (TextDataset, 0.10),
}


class Molmo2DataLoader:
    """Unified dataloader with weighted sampling."""
    
    def __init__(
        self,
        data_root: str = "./data/molmo2",
        batch_size: int = 32,
        num_workers: int = 4,
        max_samples: int = None,
        datasets: List[str] = None,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.dataset_names = datasets or list(DATASETS.keys())
        self._loader = None
    
    def _setup(self):
        """Load datasets and create DataLoader."""
        all_datasets, all_weights = [], []
        
        for name in self.dataset_names:
            if name not in DATASETS:
                continue
            path = self.data_root / name
            if not path.exists():
                continue
            
            try:
                cls, ratio = DATASETS[name]
                ds = cls(path, self.max_samples)
                all_datasets.append(ds)
                all_weights.extend([ratio / len(ds)] * len(ds) if len(ds) > 0 else [])
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        if not all_datasets:
            raise RuntimeError(f"No datasets found in {self.data_root}")
        
        combined = ConcatDataset(all_datasets)
        sampler = WeightedRandomSampler(all_weights, len(combined), replacement=True)
        
        self._loader = DataLoader(
            combined,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def __iter__(self):
        if self._loader is None:
            self._setup()
        return iter(self._loader)
    
    def __len__(self):
        if self._loader is None:
            self._setup()
        return len(self._loader)


def create_dataloader(
    data_root: str = "./data/molmo2",
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: int = None,
) -> DataLoader:
    """Create Molmo2 dataloader."""
    dl = Molmo2DataLoader(data_root, batch_size, num_workers, max_samples)
    dl._setup()
    return dl._loader


if __name__ == "__main__":
    loader = create_dataloader(batch_size=4, num_workers=0, max_samples=20)
    batch = next(iter(loader))
    print(f"Batch: {len(batch['texts'])} samples, tasks: {batch['tasks']}")
