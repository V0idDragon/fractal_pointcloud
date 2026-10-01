"""Run four validation-selected final experiments with shared fractal cache.

This runner uses only the Python standard library. It leaves src, data, the
virtual environment and previous screening results unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['objbg', 'objonly', 'modelnet40'], default='objbg')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--n-points', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--centers', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--cache-dir')
    p.add_argument('--output-dir')
    p.add_argument('--root', default='data/modelnet40_h5/modelnet40_ply_hdf5_2048',
                   help='ModelNet40 HDF5 directory, not the raw OFF dataset')
    p.add_argument('--train-h5')
    p.add_argument('--test-h5')
    p.add_argument('--pointnet-feature', choices=['xyz_dc', 'xyz_db', 'xyz_dc_db'], default='xyz_dc',
                   help='single PointNet2 fractal candidate; baseline XYZ is always included')
    p.add_argument('--dgcnn-feature', choices=['xyz_dc', 'xyz_db', 'xyz_dc_db'], default='xyz_dc',
                   help='single DGCNN fractal candidate; validation-selected default is xyz_dc')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(a, project: Path):
    if min(a.epochs, a.n_points, a.batch_size, a.centers) <= 0 or a.n_points < 64:
        raise ValueError('epochs/batch-size/centers must be positive; n-points must be >= 64')
    if a.batch_size < 2:
        raise ValueError('Batch size >= 2 is required by the model BatchNorm')
    if a.amp and not a.device.startswith('cuda'):
        raise ValueError('--amp requires --device cuda (or cuda:N)')

    def abs_path(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (project/p).resolve()

    dataset = 'modelnet40_h5' if a.dataset == 'modelnet40' else 'scanobjectnn'
    if a.dataset == 'modelnet40':
        data_args = ['--root', str(abs_path(a.root))]
    else:
        folder = 'main_split' if a.dataset == 'objbg' else 'main_split_nobg'
        base = Path('data/scanobjectnn')/folder
        tr = abs_path(a.train_h5 or str(base/'training_objectdataset.h5'))
        te = abs_path(a.test_h5 or str(base/'test_objectdataset.h5'))
        data_args = ['--train-h5', str(tr), '--test-h5', str(te)]
    cache_defaults = {
        'objbg': 'results/scanobjectnn_objbg/fractal_cache_v3',
        'objonly': 'results/scanobjectnn_objonly/fractal_cache_v3',
        'modelnet40': 'results/modelnet40/fractal_cache',
    }
    cache = abs_path(a.cache_dir or cache_defaults[a.dataset])
    out = abs_path(a.output_dir or
                   f'results/final_{a.dataset}_n{a.n_points}_e{a.epochs}_s{a.seed}')
    common = ['--dataset', dataset, *data_args, '--n-points', str(a.n_points),
              '--seed', str(a.seed), '--cache-dir', str(cache)]
    precompute = [sys.executable, '-m', 'scripts.precompute_fractal', *common,
                  '--centers', str(a.centers), '--k', '64']
    job_specs = [('pointnet2', 'xyz', 64), ('pointnet2', a.pointnet_feature, 64),
                 ('dgcnn', 'xyz', 64), ('dgcnn', a.dgcnn_feature, 64)]
    jobs = []
    for model, feature, k in job_specs:
        folder = out/f'{model}__{feature}__k{k}'
        cmd = [sys.executable, '-m', 'src.train', *common,
               '--model', model, '--feature-set', feature, '--fractal-k', str(k),
               '--fractal-centers', str(a.centers), '--epochs', str(a.epochs),
               '--batch-size', str(a.batch_size), '--workers', '0',
               '--graph-k', '20', '--lr', '0.001', '--weight-decay', '0.0001',
               '--val-fraction', '0.10', '--device', a.device,
               '--require-fractal-cache', '--output-dir', str(folder)]
        if a.amp: cmd.append('--amp')
        leaf = folder/f'{dataset}__{model}__{feature}__fk{k}'
        jobs.append({'model': model, 'feature_set': feature, 'fractal_k': k,
                     'command': cmd, 'output': str(folder), 'leaf': str(leaf)})
    return {'dataset': a.dataset, 'epochs': a.epochs, 'n_points': a.n_points,
            'seed': a.seed, 'cache_dir': str(cache), 'output_dir': str(out),
            'precompute': precompute, 'jobs': jobs}


def code_signature(project):
    h = hashlib.sha256()
    for p in sorted((project/'src').rglob('*.py')):
        h.update(p.relative_to(project).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def signature(job, source_signature):
    value = {'command': job['command'], 'src_sha256': source_signature}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def completed(job, run_signature, epochs):
    folder = Path(job['output'])
    leaf = Path(job['leaf'])
    record = folder/'minimal_run.json'
    if record.exists():
        state = json.loads(record.read_text(encoding='utf-8'))
        if state.get('signature') != run_signature:
            raise ValueError(f'Existing run has different settings/code: {folder}. '
                             'Use a different --output-dir.')
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError(f'Output folder already has untracked results: {folder}. '
                         'Use a different --output-dir.')
    else:
        return False
    required = ['metrics.json', 'config.json', 'model_best.pt', 'history.csv',
                'confusion_matrix_test.csv', 'classification_report_test.csv']
    if not all((leaf/f).is_file() for f in required): return False
    with (leaf/'history.csv').open(encoding='utf-8', newline='') as f:
        history = list(csv.DictReader(f))
    if len(history) != epochs or int(history[-1]['epoch']) != epochs: return False
    metrics = json.loads((leaf/'metrics.json').read_text(encoding='utf-8'))
    config = json.loads((leaf/'config.json').read_text(encoding='utf-8'))
    return (config['epochs'] == epochs and metrics['model'] == job['model']
            and metrics['feature_set'] == job['feature_set']
            and metrics['fractal_k'] == job['fractal_k'])


def collect(plan):
    rows = []
    for job in plan['jobs']:
        path = Path(job['leaf'])/'metrics.json'
        if path.exists(): rows.append(json.loads(path.read_text(encoding='utf-8')))
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = Path(plan['output_dir'])/'selected_summary.csv'
    with target.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f'Summary ({len(rows)}/4 completed): {target}', flush=True)


def validate_inputs(plan, project):
    if not (project/'src/train.py').exists():
        raise ValueError('Place this runner inside your fractal_pointcloud_deep project.')
    train_source = (project/'src/train.py').read_text(encoding='utf-8')
    if '--require-fractal-cache' not in train_source:
        raise ValueError('src/train.py is too old: install the verified v3/v5 patch first.')
    cmd = plan['precompute']
    if '--root' in cmd:
        root = Path(cmd[cmd.index('--root')+1])
        files = list(root.rglob('*.h5'))
        if not any('train' in p.name.lower() for p in files) or not any('test' in p.name.lower() for p in files):
            raise ValueError(f'Expected train/test HDF5 files under {root}; raw OFF files are not supported.')
    else:
        for flag in ['--train-h5', '--test-h5']:
            path = Path(cmd[cmd.index(flag)+1])
            if not path.is_file(): raise ValueError(f'Missing input: {path}')


def main(argv=None):
    a = parser().parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    plan = build_plan(a, project)
    print('Four runs only: XYZ and XYZ+Dc (k=64) for PointNet2 and DGCNN.', flush=True)
    print('Baseline k=64 is a naming placeholder; XYZ does not use fractal k.', flush=True)
    for job in plan['jobs']:
        print(f"  {job['model']} / {job['feature_set']} / k={job['fractal_k']}", flush=True)
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    validate_inputs(plan, project)
    source = code_signature(project)
    pending = []
    for job in plan['jobs']:
        sig = signature(job, source)
        if completed(job, sig, a.epochs):
            print(f"SKIP completed: {job['model']} / {job['feature_set']}", flush=True)
        else:
            pending.append((job, sig))
    atomic_json(Path(plan['output_dir'])/'selected_plan.json', plan)
    if any(job['feature_set'] != 'xyz' for job, _ in pending):
        print('Validate/prepare shared cache: k=64 only.', flush=True)
        subprocess.run(plan['precompute'], cwd=project, check=True)
    for job, sig in pending:
        record = Path(job['output'])/'minimal_run.json'
        atomic_json(record, {'signature': sig, 'src_sha256': source, 'status': 'running',
                             'command': job['command']})
        print(f"\nRUN {job['model']} / {job['feature_set']}", flush=True)
        subprocess.run(job['command'], cwd=project, check=True)
        if not completed(job, sig, a.epochs):
            raise RuntimeError(f"Run exited without complete results: {job['output']}")
        atomic_json(record, {'signature': sig, 'src_sha256': source, 'status': 'completed',
                             'command': job['command']})
        collect(plan)
    collect(plan)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
