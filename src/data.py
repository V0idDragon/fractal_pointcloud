from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re
import h5py, numpy as np
from sklearn.model_selection import train_test_split
from .utils import normalize_cloud

@dataclass
class CloudSplit:
    points: np.ndarray
    labels: np.ndarray
    class_names: list[str]


def _load_h5_files(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for path in sorted(files):
        with h5py.File(path, 'r') as f:
            keys = set(f.keys())
            if 'data' not in keys or 'label' not in keys:
                raise ValueError(f'{path} must contain data and label; got {sorted(keys)}')
            x = np.asarray(f['data'], dtype=np.float32)
            y = np.asarray(f['label']).reshape(-1).astype(np.int64)
            xs.append(x); ys.append(y)
    if not xs: raise FileNotFoundError('No matching HDF5 files found')
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def load_modelnet_h5(root: Path, n_points: int = 1024, seed: int = 42) -> tuple[CloudSplit, CloudSplit]:
    train = sorted(root.rglob('*.h5'))
    train_files = [p for p in train if 'train' in p.name.lower()]
    test_files = [p for p in train if 'test' in p.name.lower()]
    if not train_files or not test_files: raise FileNotFoundError(f'Expected train/test h5 files under {root}')
    tx, ty = _load_h5_files(train_files); vx, vy = _load_h5_files(test_files)
    rng = np.random.default_rng(seed)
    tx = _fix_points(tx, n_points, rng); vx = _fix_points(vx, n_points, rng)
    classes = [str(i) for i in sorted(set(ty.tolist()) | set(vy.tolist()))]
    return CloudSplit(tx, ty, classes), CloudSplit(vx, vy, classes)


def load_scanobjectnn(train_path: Path, test_path: Path, n_points: int = 1024, seed: int = 42) -> tuple[CloudSplit, CloudSplit]:
    tx, ty = _load_h5_files([train_path]); vx, vy = _load_h5_files([test_path])
    rng = np.random.default_rng(seed)
    tx = _fix_points(tx, n_points, rng); vx = _fix_points(vx, n_points, rng)
    classes = [str(i) for i in sorted(set(ty.tolist()) | set(vy.tolist()))]
    return CloudSplit(tx, ty, classes), CloudSplit(vx, vy, classes)


def _fix_points(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((x.shape[0], n, 3), dtype=np.float32)
    for i, pts in enumerate(x):
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] >= n: idx = rng.choice(pts.shape[0], n, replace=False)
        else: idx = rng.choice(pts.shape[0], n, replace=True)
        out[i] = normalize_cloud(pts[idx])
    return out


def load_prepared_npy(root: Path, class_names_path: Path | None = None) -> tuple[CloudSplit, CloudSplit]:
    tx=np.load(root/'train_points.npy'); ty=np.load(root/'train_labels.npy')
    vx=np.load(root/'test_points.npy'); vy=np.load(root/'test_labels.npy')
    if class_names_path and class_names_path.exists():
        mp=json.loads(class_names_path.read_text(encoding='utf-8'))
        classes=[k for k,v in sorted(mp.items(), key=lambda kv: kv[1])]
    else: classes=[str(i) for i in sorted(set(ty.tolist()) | set(vy.tolist()))]
    return CloudSplit(tx.astype(np.float32),ty.astype(np.int64),classes), CloudSplit(vx.astype(np.float32),vy.astype(np.int64),classes)
