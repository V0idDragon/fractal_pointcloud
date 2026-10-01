from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.robust import downsample,add_noise
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--mode',choices=['downsample','noise'],required=True); p.add_argument('--value',type=float,required=True); p.add_argument('--seed',type=int,default=42)
a=p.parse_args(); inp=Path(a.input); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
for split in ('train','test'):
  x=np.load(inp/f'{split}_points.npy')
  y=np.load(inp/f'{split}_labels.npy')
  z=downsample(x,int(a.value),a.seed) if a.mode=='downsample' else add_noise(x,float(a.value),a.seed)
  np.save(out/f'{split}_points.npy',z); np.save(out/f'{split}_labels.npy',y)
for name in ('class_mapping.json','train_paths.json','test_paths.json'):
  pth=inp/name
  if pth.exists(): (out/name).write_bytes(pth.read_bytes())
(out/'variant.json').write_text(json.dumps({'mode':a.mode,'value':a.value,'seed':a.seed},indent=2),encoding='utf-8')
print(out)
