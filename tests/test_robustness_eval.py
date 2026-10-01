"""Regression checks for saved-model evaluation; no optimizer or training."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.evaluate_robustness import _noise, _sparse, evaluate_checkpoint, predict
from src.data import load_scanobjectnn
from src.fractal import compute_object_features
from src.train import make_model


def test_sparse_padding_keeps_feature_alignment():
    x = np.arange(2*32*3).reshape(2,32,3).astype(np.float32)
    sparse, mapping = _sparse(x, 12, 42, 32)
    unique = sparse[:,:12]
    assert np.array_equal(np.take_along_axis(unique, mapping[:,:,None], axis=1), sparse)
    assert all(len(np.unique(c, axis=0)) == 12 for c in sparse)
    same, ids = _sparse(x, 32, 42, 32)
    assert same.shape == x.shape and ids.shape == (2,32)


def test_noise_uses_same_draw_across_amplitudes_and_keeps_scale():
    x = np.ones((2,32,3), np.float32)
    low = _noise(x, .01, 42)-x
    high = _noise(x, .02, 42)-x
    assert np.allclose(high, 2*low, atol=2e-7)
    assert np.array_equal(_noise(x, 0, 42), x)


@pytest.mark.parametrize('name,feature', [('pointnet2','xyz'),('pointnet2','xyz_dc'),('dgcnn','xyz'),('dgcnn','xyz_dc_db')])
def test_checkpoint_evaluation_reproduces_clean_and_preserves_weights(tmp_path, name, feature):
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    for split, count in [('train',8),('test',4)]:
        with h5py.File(tmp_path/f'{split}.h5','w') as f:
            f['data'] = rng.normal(size=(count,80,3)).astype(np.float32)
            f['label'] = np.arange(count)%2
    seed = 17
    tr,te = load_scanobjectnn(tmp_path/'train.h5',tmp_path/'test.h5',64,seed)
    dc = db = None
    if feature != 'xyz':
        pairs = [compute_object_features(c,16,8,seed+i)[:2] for i,c in enumerate(te.points)]
        dc = np.stack([v[0] for v in pairs]); db = np.stack([v[1] for v in pairs])
    model = make_model(name, 3+int('dc' in feature)+int('db' in feature),2,8,64)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    clean,_,_ = predict(model,te.points,te.labels,dc if 'dc' in feature else None,
                        db if 'db' in feature else None,np.arange(4),torch.device('cpu'),2)
    leaf = tmp_path/name/feature; leaf.mkdir(parents=True)
    ck = leaf/'model_best.pt'
    torch.save({'model':model.state_dict(),'args':{'dataset':'scanobjectnn','model':name,
        'feature_set':feature,'seed':seed,'n_points':64,'fractal_k':16,'fractal_centers':8,'graph_k':8},
        'class_names':['0','1']},ck)
    (leaf/'metrics.json').write_text(json.dumps({'test_'+k:v for k,v in clean.items()}))
    a = argparse.Namespace(dataset='scanobjectnn',train_h5=str(tmp_path/'train.h5'),
        test_h5=str(tmp_path/'test.h5'),root=None,seed=42,n_points=64,centers=8,device='cpu',
        noise=[.01],density=[32],sensitivity=[8],cache_dir=str(tmp_path/'cache'),
        clean_cache_dir=None,batch_size=2)
    rows = evaluate_checkpoint(ck,a,tmp_path/'out')
    assert rows[0]['training_seed'] == seed
    assert rows[0]['accuracy'] == clean['accuracy']
    assert rows[0]['macro_f1'] == clean['macro_f1']
    assert len(rows) == (3 if feature == 'xyz' else 4)
    after = torch.load(ck,weights_only=True)['model']
    assert all(torch.equal(before[k],after[k]) for k in before)


def test_clean_mismatch_is_rejected(tmp_path):
    # This is exercised through the full checkpoint test by a deliberately
    # different old metric: corruptions must never hide a broken clean audit.
    rng=np.random.default_rng(3)
    for split in ['train','test']:
        with h5py.File(tmp_path/f'{split}.h5','w') as f:
            f['data']=rng.normal(size=(4,32,3)).astype('f'); f['label']=[0,1,0,1]
    m=make_model('dgcnn',3,2,8,32)
    ck=tmp_path/'model_best.pt'
    torch.save({'model':m.state_dict(),'args':{'dataset':'scanobjectnn','model':'dgcnn','feature_set':'xyz','n_points':32,'graph_k':8}},ck)
    (tmp_path/'metrics.json').write_text('{"test_accuracy": -1}')
    a=argparse.Namespace(dataset='scanobjectnn',train_h5=str(tmp_path/'train.h5'),test_h5=str(tmp_path/'test.h5'),root=None,seed=42,n_points=32,centers=8,device='cpu',noise=[.01],density=[],sensitivity=[],cache_dir=str(tmp_path/'cache'),clean_cache_dir=None,batch_size=2)
    with pytest.raises(RuntimeError,match='Clean does not reproduce'):
        evaluate_checkpoint(ck,a,tmp_path/'out')
