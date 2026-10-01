from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

p=argparse.ArgumentParser(description='Run comparable XYZ/Dc/Db ablations and append metrics to summary.csv')
p.add_argument('--dataset',required=True,choices=['prepared_modelnet','modelnet40_h5','scanobjectnn'])
p.add_argument('--root'); p.add_argument('--class-mapping'); p.add_argument('--train-h5'); p.add_argument('--test-h5')
p.add_argument('--model',default='pointnet2',choices=['pointnet2','dgcnn']); p.add_argument('--feature-sets',nargs='+',default=['xyz','xyz_dc','xyz_db','xyz_dc_db'])
p.add_argument('--fractal-k',type=int,nargs='+',default=[32]); p.add_argument('--fractal-centers',type=int,default=32); p.add_argument('--n-points',type=int,default=1024)
p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--workers',type=int,default=0); p.add_argument('--output-dir',default='results/deep'); p.add_argument('--cache-dir',default=None); p.add_argument('--device',default='auto'); p.add_argument('--amp',action='store_true'); p.add_argument('--exact-fps',action='store_true'); p.add_argument('--seed',type=int,default=42); p.add_argument('--graph-k',type=int,default=20)
a=p.parse_args()
base=[sys.executable,'-m','src.train','--dataset',a.dataset,'--model',a.model,'--fractal-centers',str(a.fractal_centers),'--n-points',str(a.n_points),'--epochs',str(a.epochs),'--batch-size',str(a.batch_size),'--workers',str(a.workers),'--output-dir',a.output_dir,'--device',a.device,'--seed',str(a.seed),'--graph-k',str(a.graph_k)]
if a.cache_dir: base += ['--cache-dir',a.cache_dir]
if a.amp: base += ['--amp']
if a.exact_fps: base += ['--exact-fps']
if a.root: base += ['--root',a.root]
if a.class_mapping: base += ['--class-mapping',a.class_mapping]
if a.train_h5: base += ['--train-h5',a.train_h5]
if a.test_h5: base += ['--test-h5',a.test_h5]
for k in a.fractal_k:
  for fs in a.feature_sets:
    cmd=base+['--feature-set',fs,'--fractal-k',str(k)]
    print('\n$', ' '.join(cmd), flush=True); subprocess.run(cmd,check=True)
