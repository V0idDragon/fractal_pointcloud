from __future__ import annotations
import numpy as np
from .utils import normalize_cloud

def downsample(points: np.ndarray, n_points: int, seed: int=42) -> np.ndarray:
    rng=np.random.default_rng(seed); out=[]
    for p in points:
        idx=rng.choice(len(p),n_points,replace=False) if len(p)>=n_points else rng.choice(len(p),n_points,replace=True)
        out.append(normalize_cloud(p[idx]))
    return np.stack(out).astype(np.float32)

def add_noise(points: np.ndarray, sigma: float, seed: int=42) -> np.ndarray:
    rng=np.random.default_rng(seed); return np.stack([normalize_cloud(p+rng.normal(0,sigma,p.shape).astype(np.float32)) for p in points]).astype(np.float32)
