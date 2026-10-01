"""Evaluate saved classifiers on clean, noisy, sparse and k-sensitivity inputs.

This script deliberately never calls the training loop.  Fractal descriptors are
computed once per *condition* and loaded from an independent fingerprinted cache.
It is intended for a practical robustness appendix, not for claiming a new
training result.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from src.data import load_modelnet_h5, load_scanobjectnn
from src.fractal import cache_base, precompute_split
from src.train import PCDataset, make_model, model_forward
from src.utils import seed_everything, ensure_dir


def _feature_flags(name: str) -> tuple[bool, bool]:
    return "dc" in name, "db" in name


def _metric(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }


@torch.inference_mode()
def predict(model, points, labels, dc, db, ids, device, batch_size):
    ds = PCDataset(points, labels, dc, db, indices=None, id_offset=0)
    # Replace the dataset ids with the requested stable IDs.  This is important
    # for the cached PointNet2 grouping topology.
    ds.ids = np.asarray(ids, dtype=np.int64)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ys, ps = [], []
    if hasattr(model, "clear_topology_cache"):
        model.clear_topology_cache()
    model.eval()
    for x, y, sid in dl:
        out = model_forward(model, x.to(device), sid.to(device))
        ys.extend(y.numpy().tolist()); ps.extend(out.argmax(1).cpu().numpy().tolist())
    return _metric(np.asarray(ys), np.asarray(ps)), np.asarray(ys), np.asarray(ps)


def _load_arrays(a):
    if a.dataset == "modelnet40_h5":
        tr, te = load_modelnet_h5(Path(a.root), a.n_points, a.seed)
    else:
        tr, te = load_scanobjectnn(Path(a.train_h5), Path(a.test_h5), a.n_points, a.seed)
    return te.points.astype(np.float32), te.labels.astype(np.int64), te.class_names


def _noise(x, sigma, seed):
    rng = np.random.default_rng(seed)
    # The clean clouds are already in a unit-scale normalisation.  Do not
    # renormalise after adding noise: otherwise the perturbation is silently
    # removed and the test becomes difficult to interpret.
    return (x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)).astype(np.float32)


def _sparse(x, n, seed, target_n):
    if not 6 <= n <= target_n == x.shape[1]:
        raise ValueError('density must be between 6 and the trained input size')
    rng = np.random.default_rng(seed)
    out = np.empty((len(x), target_n, 3), np.float32)
    maps = np.empty((len(x), target_n), np.int64)
    for i, cloud in enumerate(x):
        idx = rng.choice(len(cloud), n, replace=False)
        # Keep the model input size fixed. Repeated points are padding only;
        # fractal descriptors are computed on the n unique sampled points.
        pad_local = rng.choice(n, target_n - n, replace=True)
        maps[i] = np.concatenate([np.arange(n), pad_local])
        out[i] = cloud[idx[maps[i]]]
    return out, maps


def _descriptor_arrays(points, k, centers, cache_root, dataset, tag, seed, clean_cache=None):
    # Compute on the exact array in this condition; the manifest checks shape,
    # dtype and all bytes before reporting a cache hit.
    base = cache_base(clean_cache if tag == 'clean' and clean_cache else Path(cache_root) / tag, dataset, points.shape[1])
    precompute_split(points, base / "test", [k], centers, seed, tag, require_cached=False)
    dc = np.load(base / "test" / f"dc_k{k}.npy", allow_pickle=False)
    db = np.load(base / "test" / f"db_k{k}.npy", allow_pickle=False)
    return dc, db


def _write_csv(path, rows):
    ensure_dir(path.parent)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def evaluate_checkpoint(ckpt_path: Path, a, out_root: Path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved = state.get("args", {})
    model_name = saved.get("model", "pointnet2")
    feature = saved.get("feature_set", "xyz")
    trained_k = int(saved.get("fractal_k", 64))
    centers = int(saved.get("fractal_centers", a.centers))
    n_points = int(saved.get("n_points", a.n_points))
    # Re-load with the same sampling order as training.  Test points are after
    # the training split in the original loader's RNG stream, so loading both
    # splits is intentional even though only test labels are evaluated here.
    if saved.get('dataset') != a.dataset:
        raise ValueError(f'Checkpoint dataset {saved.get("dataset")} differs from --dataset {a.dataset}')
    aa = argparse.Namespace(**vars(a)); aa.n_points = n_points
    training_seed = int(saved.get('seed', 42))
    aa.seed = training_seed
    x, y, class_names = _load_arrays(aa)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    in_ch = 3 + int("dc" in feature) + int("db" in feature)
    model = make_model(model_name, in_ch, len(class_names), int(saved.get("graph_k", 20)), n_points, bool(saved.get("exact_fps", False)))
    model.load_state_dict(state["model"]); model.to(device).eval()
    checkpoint_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    ckpt_out = out_root / (ckpt_path.parent.name + '_' + checkpoint_hash[:10])
    ensure_dir(ckpt_out)
    rows = []
    conditions = [("clean", x, trained_k, x, None)]
    for sigma in a.noise:
        nx = _noise(x, sigma, a.seed)
        conditions.append((f"noise_{sigma:g}", nx, trained_k, nx, None))
    for n in a.density:
        if n < n_points:
            sx, maps = _sparse(x, n, a.seed, n_points)
            # Descriptor estimation sees only unique sampled points.  The model
            # receives a fixed-size cloud, so descriptors follow the exact
            # repeated-point map used for padding.
            unique = sx[:, :n, :]
            conditions.append((f"density_{n}", sx, trained_k, unique, maps))
    for k in a.sensitivity:
        if feature != 'xyz' and int(k) != trained_k:
            conditions.append((f"sensitivity_k{k}", x, int(k), x, None))

    clean = None
    for tag, pts, k, descriptor_points, maps in conditions:
        t0 = time.perf_counter()
        seed_everything(training_seed)
        need_dc, need_db = _feature_flags(feature)
        dc = db = None
        if need_dc or need_db:
            if k > descriptor_points.shape[1]:
                raise ValueError(f'{tag}: descriptor k={k} exceeds unique point count')
            dc0, db0 = _descriptor_arrays(descriptor_points, k, centers, a.cache_dir, a.dataset, tag, training_seed, a.clean_cache_dir)
            if maps is None:
                dc, db = dc0, db0
            else:
                dc = np.take_along_axis(dc0, maps, axis=1)
                db = np.take_along_axis(db0, maps, axis=1)
        metrics, yt, yp = predict(model, pts, y, dc if need_dc else None, db if need_db else None,
                                  np.arange(len(y), dtype=np.int64), device, a.batch_size)
        row = {"checkpoint": str(ckpt_path), "model": model_name,
               "feature_set": feature, "trained_k": trained_k, "condition": tag,
               "descriptor_k": k, "n_points": n_points, 'training_seed': training_seed,
               'corruption_seed': a.seed, 'checkpoint_sha256': checkpoint_hash,
               'seconds': time.perf_counter()-t0, **metrics}
        if tag == 'clean':
            clean = metrics
            old_path = ckpt_path.parent / 'metrics.json'
            if old_path.exists():
                old = json.loads(old_path.read_text(encoding='utf-8'))
                gaps = {name: abs(metrics[name]-float(old['test_'+name])) for name in metrics if 'test_'+name in old}
                if any(v > 1e-6 for v in gaps.values()):
                    _write_csv(ckpt_out / 'clean_mismatch.csv', [row])
                    raise RuntimeError(f'Clean does not reproduce {old_path}: {gaps}. Stopping before corruptions; check code, paths, sampling and seed.')
        for name, value in metrics.items():
            row['change_'+name+'_pp'] = 100*(value-clean[name])
        rows.append(row)
        np.savetxt(ckpt_out / f"predictions_{tag}.csv", np.column_stack([yt, yp]),
                   fmt="%d", delimiter=",", header="y_true,y_pred", comments="")
        _write_csv(ckpt_out / "robustness_summary.csv", rows)
        print(f'{model_name} {feature} {tag}: acc={metrics["accuracy"]:.4f} F1={metrics["macro_f1"]:.4f}', flush=True)
    _write_csv(ckpt_out / "robustness_summary.csv", rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", default=[],
                   help="path to model_best.pt; repeat for multiple saved models")
    p.add_argument('--checkpoints', nargs='+', default=[])
    p.add_argument("--dataset", choices=["modelnet40_h5", "scanobjectnn"], required=True)
    p.add_argument("--root", help="ModelNet HDF5 directory")
    p.add_argument("--train-h5"); p.add_argument("--test-h5")
    p.add_argument("--n-points", type=int, default=1024)
    p.add_argument("--centers", type=int, default=32)
    p.add_argument("--cache-dir", default="fractal_cache_robustness")
    p.add_argument('--clean-cache-dir', help='optional original training cache root for validating/reusing clean features')
    p.add_argument("--output-dir", default="results/robustness")
    p.add_argument("--noise", type=float, nargs="*", default=[.005, .01, .02])
    p.add_argument("--density", type=int, nargs="*", default=[512, 256])
    p.add_argument("--sensitivity", type=int, nargs="*", default=[16, 32, 128])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(); seed_everything(a.seed)
    a.checkpoint = list(dict.fromkeys(a.checkpoint+a.checkpoints))
    if not a.checkpoint: p.error('supply --checkpoint or --checkpoints')
    if a.batch_size < 1 or any(s < 0 for s in a.noise): p.error('invalid batch size or negative noise')
    all_rows = []
    for ck in a.checkpoint:
        all_rows.extend(evaluate_checkpoint(Path(ck), a, Path(a.output_dir)))
    _write_csv(Path(a.output_dir) / "robustness_all.csv", all_rows)
    print(json.dumps({"checkpoints": len(a.checkpoint), "conditions_per_checkpoint": len(all_rows)//max(len(a.checkpoint),1),
                      "output": str(Path(a.output_dir).resolve())}, indent=2))


if __name__ == "__main__":
    main()
