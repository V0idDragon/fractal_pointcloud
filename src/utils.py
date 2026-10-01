from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        # Fixed tensor shapes are used throughout classification. These flags
        # let cuDNN/GEMM select faster kernels on supported NVIDIA GPUs.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def ensure_dir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)

def now() -> float: return time.perf_counter()

def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent); path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def normalize_cloud(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    r = np.linalg.norm(x, axis=1).max()
    if r <= 1e-12: raise ValueError('Degenerate point cloud')
    return (x / r).astype(np.float32)
