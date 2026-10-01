from __future__ import annotations
from pathlib import Path
import hashlib, json, math, time
import numpy as np
from scipy.spatial import cKDTree


def fps_indices(points: np.ndarray, m: int, seed: int = 42) -> np.ndarray:
    n=len(points)
    if m>=n: return np.arange(n, dtype=np.int64)
    rng=np.random.default_rng(seed); first=int(rng.integers(0,n))
    idx=np.empty(m,dtype=np.int64); idx[0]=first
    dist=np.full(n,np.inf,dtype=np.float32)
    cur=points[first]
    for j in range(1,m):
        d=((points-cur)**2).sum(axis=1); dist=np.minimum(dist,d)
        idx[j]=int(np.argmax(dist)); cur=points[idx[j]]
    return idx


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float,float]:
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(x)<3 or np.ptp(x)<1e-8: return np.nan, np.nan
    coef=np.polyfit(x,y,1); pred=np.polyval(coef,x)
    sst=float(((y-y.mean())**2).sum()); sse=float(((y-pred)**2).sum())
    r2=1.0-sse/sst if sst>1e-12 else np.nan
    return float(coef[0]), float(r2)


def local_dc(neigh: np.ndarray) -> float:
    k=len(neigh)
    if k<6: return np.nan
    d=np.linalg.norm(neigh[:,None,:]-neigh[None,:,:],axis=-1)
    iu=np.triu_indices(k,1); vals=d[iu]
    vals=vals[np.isfinite(vals)&(vals>1e-8)]
    if len(vals)<10: return np.nan
    radii=np.quantile(vals, [0.10,0.15,0.20,0.30,0.40,0.55,0.70,0.85])
    radii=np.unique(radii)
    if len(radii)<4: return np.nan
    c=np.array([(vals<=r).mean() for r in radii], dtype=np.float64)
    mask=(c>0)&(c<1)&np.isfinite(c)
    if mask.sum()<4: return np.nan
    slope,r2=_fit(np.log(radii[mask]), np.log(c[mask]))
    return slope if np.isfinite(slope) and slope>=0 else np.nan


def local_db(neigh: np.ndarray) -> float:
    if len(neigh)<6: return np.nan
    q=neigh-neigh.min(0,keepdims=True); span=q.max(0); span[span<1e-8]=1.0; q=q/span
    eps=np.array([0.5,0.25,0.125,0.0625],dtype=np.float64)
    counts=[]
    for e in eps:
        cell=np.floor(q/e).astype(np.int32)
        counts.append(np.unique(cell,axis=0).shape[0])
    counts=np.asarray(counts,dtype=np.float64)
    mask=(counts>1)&np.isfinite(counts)
    if mask.sum()<3: return np.nan
    slope,r2=_fit(np.log(1/eps[mask]), np.log(counts[mask]))
    return slope if np.isfinite(slope) and slope>=0 else np.nan


def compute_object_features(points: np.ndarray, k: int, centers: int, seed: int=42) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=len(points); centers_idx=fps_indices(points, min(centers,n), seed)
    tree=cKDTree(points); _, neigh_idx=tree.query(points[centers_idx], k=min(k,n))
    if neigh_idx.ndim==1: neigh_idx=neigh_idx[:,None]
    dc=np.full(len(centers_idx),np.nan,np.float32); db=np.full(len(centers_idx),np.nan,np.float32)
    for j,ids in enumerate(neigh_idx):
        nb=points[ids]
        dc[j]=local_dc(nb); db[j]=local_db(nb)
    valid=np.isfinite(dc)|np.isfinite(db)
    if not valid.any():
        raise RuntimeError(f'No valid fractal descriptors for k={k}')
    # Nearest-anchor propagation keeps per-point feature alignment while limiting the expensive
    # correlation-sum/box-counting calculations to a reproducible subset of centers.
    atree=cKDTree(points[centers_idx])
    dist, ai=atree.query(points, k=min(3,len(centers_idx)))
    if dist.ndim==1: dist=dist[:,None]; ai=ai[:,None]
    w=1.0/np.maximum(dist,1e-6); w/=w.sum(axis=1,keepdims=True)
    dcv=np.nan_to_num(dc, nan=np.nanmedian(dc[np.isfinite(dc)]) if np.isfinite(dc).any() else 0.0)
    dbv=np.nan_to_num(db, nan=np.nanmedian(db[np.isfinite(db)]) if np.isfinite(db).any() else 0.0)
    dc_all=(w*dcv[ai]).sum(axis=1).astype(np.float32); db_all=(w*dbv[ai]).sum(axis=1).astype(np.float32)
    dc_all=np.nan_to_num(dc_all); db_all=np.nan_to_num(db_all)
    return dc_all, db_all, centers_idx


def _dataset_fingerprint(points: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(points).shape.__repr__().encode())
    h.update(np.asarray(points).dtype.str.encode())
    h.update(np.ascontiguousarray(points).view(np.uint8))
    return h.hexdigest()


def cache_base(cache_dir, dataset: str, n_points: int) -> Path:
    """One path convention for precompute and training, independent of output-dir."""
    root = Path(cache_dir) if cache_dir else Path('fractal_cache')
    return root / dataset / f'n{n_points}'


def _valid_cache_array(path: Path, shape: tuple) -> bool:
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        return array.shape == shape and array.dtype == np.float32 and bool(np.isfinite(array).all())
    except (OSError, ValueError, EOFError):
        return False


def precompute_split(points: np.ndarray, out: Path, ks: list[int], centers: int,
                     seed: int, tag: str, require_cached: bool = False) -> dict:
    """Validate/reuse descriptors; preserve all k entries across repeated calls.

    No descriptor calculation occurs inside the training loop. With
    require_cached=True, a cache miss raises before any calculation or training.
    """
    out = Path(out)
    if points.ndim != 3 or points.shape[2] != 3 or not len(points):
        raise ValueError('points must be a non-empty (objects, points, 3) array')
    if centers < 1 or any(k < 6 or k > points.shape[1] for k in ks):
        raise ValueError('centers must be positive and 6 <= k <= number of points')
    if not np.isfinite(points).all():
        raise ValueError('non-finite input coordinates')
    fingerprint = _dataset_fingerprint(points)
    manifest_path = out / 'manifest.json'
    try:
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        old = {}
    if not isinstance(old, dict):
        old = {}
    compatible = (old.get('schema') == 2 and old.get('fingerprint') == fingerprint
                  and old.get('centers') == int(centers) and old.get('seed') == int(seed))
    old_entries = old.get('ks', {}) if compatible else {}
    if not isinstance(old_entries, dict):
        old_entries = {}
    manifest = {'schema': 2, 'tag': tag, 'shape': list(points.shape),
                'dtype': str(points.dtype), 'fingerprint': fingerprint,
                'centers': int(centers), 'seed': int(seed), 'ks': dict(old_entries)}
    shape = (len(points), points.shape[1])

    def commit():
        out.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        tmp.replace(manifest_path)

    for k in dict.fromkeys(ks):
        dc_path, db_path = out/f'dc_k{k}.npy', out/f'db_k{k}.npy'
        entry = old_entries.get(str(k), {})
        valid = (isinstance(entry, dict) and bool(entry)
                 and _valid_cache_array(dc_path, shape) and _valid_cache_array(db_path, shape))
        if valid:
            manifest['ks'][str(k)] = {**entry, 'dc': dc_path.name, 'db': db_path.name,
                                     'shape': list(shape), 'cached': True}
            print(f'  cache HIT {tag} k={k}: {len(points)} objects', flush=True)
            continue
        if require_cached:
            raise RuntimeError(f'Cache missing or incompatible: {out}, k={k}. '
                               'Run python -m scripts.precompute_fractal with the same '
                               '--cache-dir, n-points, centers, seed and input data first.')
        print(f'  cache MISS {tag} k={k}: computing descriptors once', flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dc = np.empty(shape, np.float32)
        db = np.empty_like(dc)
        t = time.perf_counter()
        for i, cloud in enumerate(points):
            dc[i], db[i], _ = compute_object_features(cloud, k, centers, seed+i)
            if (i+1) % 25 == 0 or i+1 == len(points):
                print(f'  fractal {tag} k={k}: {i+1}/{len(points)}', flush=True)
        dc_tmp, db_tmp = dc_path.with_suffix('.tmp.npy'), db_path.with_suffix('.tmp.npy')
        np.save(dc_tmp, dc)
        np.save(db_tmp, db)
        dc_tmp.replace(dc_path)
        db_tmp.replace(db_path)
        manifest['ks'][str(k)] = {'dc': dc_path.name, 'db': db_path.name, 'cached': False,
                                'seconds': time.perf_counter()-t, 'shape': list(shape)}
        commit()
    commit()
    return manifest
