import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np
from src.fractal import compute_object_features

def test_fractal_features_shape_and_finite():
    rng=np.random.default_rng(1)
    p=rng.normal(size=(128,3)).astype(np.float32)
    dc,db,idx=compute_object_features(p,k=16,centers=8,seed=1)
    assert dc.shape==db.shape==(128,)
    assert idx.shape==(8,)
    assert np.isfinite(dc).all() and np.isfinite(db).all()
