import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import fractal


def points():
    return np.random.default_rng(42).normal(size=(6, 128, 3)).astype(np.float32)


def test_cache_reused_repeatedly_for_all_k(tmp_path, monkeypatch):
    x = points()
    fractal.precompute_split(x, tmp_path, [16, 32, 64, 128], 4, 42, 'train')
    def fail(*a, **kw):
        raise AssertionError('Descriptors were recomputed on a valid cache')
    monkeypatch.setattr(fractal, 'compute_object_features', fail)
    for _ in range(3):
        for k in [16, 32, 64, 128]:
            result = fractal.precompute_split(x, tmp_path, [k], 4, 42, 'train', require_cached=True)
            assert result['ks'][str(k)]['cached']
            assert result['ks'][str(k)]['shape'] == [6, 128]
    assert set(json.loads((tmp_path/'manifest.json').read_text())['ks']) == {'16','32','64','128'}


@pytest.mark.parametrize('change', ['points','seed','centers','truncated','nan'])
def test_incompatible_cache_rejected(tmp_path, change):
    x = points()
    fractal.precompute_split(x, tmp_path, [16], 4, 42, 'train')
    seed, centers = 42, 4
    if change == 'points': x[0, 0, 0] += 1
    if change == 'seed': seed = 43
    if change == 'centers': centers = 5
    if change == 'truncated': (tmp_path/'dc_k16.npy').write_bytes(b'broken')
    if change == 'nan':
        bad = np.load(tmp_path/'dc_k16.npy'); bad[0,0] = np.nan
        np.save(tmp_path/'dc_k16.npy', bad)
    with pytest.raises(RuntimeError, match='Cache missing or incompatible'):
        fractal.precompute_split(x, tmp_path, [16], centers, seed, 'train', require_cached=True)


def test_precompute_entrypoints_and_missing_test_h5():
    root = Path(__file__).resolve().parents[1]
    for entry in [['scripts/precompute_fractal.py'], ['-m','scripts.precompute_fractal']]:
        result = subprocess.run([sys.executable,*entry,'--help'], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([sys.executable,*entry,'--dataset','scanobjectnn','--train-h5','train.h5'],
                                cwd=root,capture_output=True,text=True)
        assert result.returncode == 2
        assert 'requires both --train-h5 and --test-h5' in result.stderr


@pytest.mark.parametrize('model', ['pointnet2','dgcnn'])
def test_training_uses_precompute_without_descriptor_calls(tmp_path, monkeypatch, model):
    import torch
    from src.train import main
    torch.set_num_threads(2)
    x = points(); y = np.array([0,1,0,1,0,1], dtype=np.int64)
    data = tmp_path/'data'; data.mkdir()
    cache = tmp_path/'cache'
    for tag in ['train','test']:
        np.save(data/f'{tag}_points.npy', x); np.save(data/f'{tag}_labels.npy', y)
        fractal.precompute_split(x, fractal.cache_base(cache, 'prepared_modelnet', 128)/tag,
                                [16], 4, 42, tag)
    def fail(*a, **kw):
        raise AssertionError('Training tried to recompute descriptors')
    monkeypatch.setattr(fractal, 'compute_object_features', fail)
    monkeypatch.setattr(sys, 'argv', ['run.py','--dataset','prepared_modelnet','--root',str(data),
        '--model',model,'--feature-set','xyz_dc_db','--fractal-k','16','--fractal-centers','4',
        '--n-points','128','--epochs','1','--batch-size','2','--val-fraction','0.33',
        '--device','cpu','--cache-dir',str(cache),'--require-fractal-cache',
        '--output-dir',str(tmp_path/'results')])
    main()
    metrics = list((tmp_path/'results').rglob('metrics.json'))
    assert len(metrics) == 1
    assert json.loads(metrics[0].read_text())['n_test'] == 6
