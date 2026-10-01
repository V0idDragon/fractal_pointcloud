"""Small ModelNet40 screening: one architecture, four feature sets.

The screening is intentionally cheap (default 512 points / 20 epochs). It is
used to choose the 1--2 candidates worth running for 100 epochs at 1024
points; it is not a replacement for the final experiment.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/modelnet40_h5/modelnet40_ply_hdf5_2048')
    p.add_argument('--model', choices=['pointnet2', 'dgcnn'], default='pointnet2')
    p.add_argument('--n-points', type=int, default=512)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--k', type=int, default=64)
    p.add_argument('--centers', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--cache-dir', default='results/modelnet40_screen/fractal_cache')
    p.add_argument('--output-dir', default='results/modelnet40_screen')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(); project = Path(__file__).resolve().parents[1]
    root = Path(a.root); root = root if root.is_absolute() else project / root
    out = Path(a.output_dir); out = out if out.is_absolute() else project / out
    cache = Path(a.cache_dir); cache = cache if cache.is_absolute() else project / cache
    files = list(root.rglob('*.h5')) if root.exists() else []
    if not any('train' in f.name.lower() for f in files) or not any('test' in f.name.lower() for f in files):
        raise SystemExit(f'No train/test HDF5 files found under {root}')
    base = [sys.executable, '-m', 'src.train', '--dataset', 'modelnet40_h5',
            '--root', str(root), '--model', a.model, '--fractal-k', str(a.k),
            '--fractal-centers', str(a.centers), '--n-points', str(a.n_points),
            '--epochs', str(a.epochs), '--batch-size', str(a.batch_size),
            '--workers', '0', '--lr', '0.001', '--weight-decay', '0.0001',
            '--val-fraction', '0.10', '--seed', str(a.seed), '--device', a.device,
            '--cache-dir', str(cache), '--require-fractal-cache']
    pre = [sys.executable, '-m', 'scripts.precompute_fractal', '--dataset', 'modelnet40_h5',
           '--root', str(root), '--n-points', str(a.n_points), '--k', str(a.k),
           '--centers', str(a.centers), '--seed', str(a.seed), '--cache-dir', str(cache)]
    jobs = []
    for fs in ('xyz', 'xyz_dc', 'xyz_db', 'xyz_dc_db'):
        folder = out / f'{a.model}__{fs}__k{a.k}'
        cmd = base + ['--feature-set', fs, '--output-dir', str(folder)]
        if a.amp: cmd.append('--amp')
        jobs.append((fs, cmd))
    print(f'ModelNet40 screening: {a.model}, {a.n_points} points, {a.epochs} epochs, k={a.k}')
    for fs, cmd in jobs: print('  ', fs, ' '.join(map(str, cmd)))
    if a.dry_run: return
    subprocess.run(pre, cwd=project, check=True)
    for fs, cmd in jobs:
        leaf = out / f'{a.model}__{fs}__k{a.k}' / f'modelnet40_h5__{a.model}__{fs}__fk{a.k}'
        if (leaf / 'metrics.json').exists() and (leaf / 'history.csv').exists():
            print('SKIP existing', fs); continue
        subprocess.run(cmd, cwd=project, check=True)
    rows = []
    for fs, _ in jobs:
        path = out / f'{a.model}__{fs}__k{a.k}' / f'modelnet40_h5__{a.model}__{fs}__fk{a.k}' / 'metrics.json'
        if path.exists(): rows.append(json.loads(path.read_text(encoding='utf-8')))
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out / 'screening_summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        best = max(rows, key=lambda r: r.get('best_val_macro_f1', -1))
        print('Validation-selected candidate:', best['feature_set'], 'val_macro_f1=', best['best_val_macro_f1'])
        print('Run that candidate plus XYZ at 1024 points / 100 epochs for the final comparison.')


if __name__ == '__main__': main()
