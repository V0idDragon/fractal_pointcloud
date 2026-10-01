"""Precompute shared fractal descriptors before any model training."""
from pathlib import Path
import argparse
import sys

# Support both `python scripts/precompute_fractal.py` and `python -m scripts.precompute_fractal`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fractal import precompute_split, cache_base
from src.data import load_prepared_npy, load_modelnet_h5, load_scanobjectnn
from src.utils import seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['prepared_modelnet', 'modelnet40_h5', 'scanobjectnn'], required=True)
    p.add_argument('--root')
    p.add_argument('--class-mapping')
    p.add_argument('--train-h5')
    p.add_argument('--test-h5')
    p.add_argument('--n-points', type=int, default=1024)
    p.add_argument('--k', type=int, nargs='+', default=[16, 32, 64, 128])
    p.add_argument('--centers', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='results/deep', help='Compatibility option; use --cache-dir for cache location.')
    p.add_argument('--cache-dir', default=None)
    a = p.parse_args()
    if a.dataset == 'scanobjectnn' and (not a.train_h5 or not a.test_h5):
        p.error('scanobjectnn requires both --train-h5 and --test-h5')
    if a.dataset != 'scanobjectnn' and not a.root:
        p.error('--root is required for this dataset')
    if a.n_points < 6 or a.centers < 1 or any(k < 6 or k > a.n_points for k in a.k):
        p.error('require n-points >= 6, centers >= 1 and 6 <= k <= n-points')
    seed_everything(a.seed)
    if a.dataset == 'prepared_modelnet':
        tr, te = load_prepared_npy(Path(a.root), Path(a.class_mapping) if a.class_mapping else None)
    elif a.dataset == 'modelnet40_h5':
        tr, te = load_modelnet_h5(Path(a.root), a.n_points, a.seed)
    else:
        tr, te = load_scanobjectnn(Path(a.train_h5), Path(a.test_h5), a.n_points, a.seed)
    base = cache_base(a.cache_dir, a.dataset, tr.points.shape[1])
    print(f'Cache: {base.resolve()}', flush=True)
    for tag, split in [('train', tr), ('test', te)]:
        precompute_split(split.points, base/tag, a.k, a.centers, a.seed, tag)
    print('Precompute complete. Use the same --cache-dir, n-points, centers and seed for training.', flush=True)


if __name__ == '__main__':
    main()
