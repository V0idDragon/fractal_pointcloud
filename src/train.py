from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix,classification_report
from .data import load_prepared_npy,load_modelnet_h5,load_scanobjectnn
from .fractal import precompute_split, cache_base
from .models.pointnet2 import PointNet2Classifier
from .models.dgcnn import DGCNNClassifier
from .utils import seed_everything,ensure_dir,save_json

class PCDataset(Dataset):
    def __init__(self,points,labels,dc=None,db=None,indices=None,id_offset=0):
        base=np.arange(len(labels),dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64)
        self.ids=base+int(id_offset); self.x=points if indices is None else points[indices]; self.y=labels if indices is None else labels[indices]; self.dc=None if dc is None else (dc if indices is None else dc[indices]); self.db=None if db is None else (db if indices is None else db[indices])
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        chans=[self.x[i]]
        if self.dc is not None: chans.append(self.dc[i,:,None])
        if self.db is not None: chans.append(self.db[i,:,None])
        return torch.from_numpy(np.concatenate(chans,1)),torch.tensor(int(self.y[i]),dtype=torch.long),torch.tensor(int(self.ids[i]),dtype=torch.long)

def model_forward(model,x,ids):
    return model(x,ids) if isinstance(model,PointNet2Classifier) else model(x)

def eval_epoch(model,loader,device):
    model.eval(); ys=[]; ps=[]; loss_sum=0.; n=0
    ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y,ids in loader:
            x=x.to(device); y=y.to(device); out=model_forward(model,x,ids); loss=ce(out,y); loss_sum += float(loss)*len(y); n+=len(y); ys.extend(y.cpu().numpy()); ps.extend(out.argmax(1).cpu().numpy())
    return {'loss':loss_sum/max(n,1),'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(ys,ps),'y_true':ys,'y_pred':ps}

def make_model(name,in_ch,num_classes,graph_k,n_points,exact_fps=False):
    return PointNet2Classifier(in_ch,num_classes,n_points,exact_fps=exact_fps) if name=='pointnet2' else DGCNNClassifier(in_ch,num_classes,graph_k)

def load_dataset(args):
    if args.dataset=='prepared_modelnet': return load_prepared_npy(Path(args.root),Path(args.class_mapping) if args.class_mapping else None)
    if args.dataset=='modelnet40_h5': return load_modelnet_h5(Path(args.root),args.n_points,args.seed)
    if args.dataset=='scanobjectnn': return load_scanobjectnn(Path(args.train_h5),Path(args.test_h5),args.n_points,args.seed)
    raise ValueError(args.dataset)

def write_csv(path,rows,fieldnames):
    ensure_dir(path.parent)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['prepared_modelnet','modelnet40_h5','scanobjectnn'],required=True); ap.add_argument('--root'); ap.add_argument('--class-mapping'); ap.add_argument('--train-h5'); ap.add_argument('--test-h5'); ap.add_argument('--model',choices=['pointnet2','dgcnn'],default='pointnet2'); ap.add_argument('--feature-set',choices=['xyz','xyz_dc','xyz_db','xyz_dc_db'],default='xyz'); ap.add_argument('--fractal-k',type=int,default=32); ap.add_argument('--fractal-centers',type=int,default=32); ap.add_argument('--n-points',type=int,default=1024); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--workers',type=int,default=0); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--graph-k',type=int,default=20); ap.add_argument('--val-fraction',type=float,default=.1); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--output-dir',default='results/deep'); ap.add_argument('--cache-dir',default=None,help='shared fractal cache root'); ap.add_argument('--require-fractal-cache',action='store_true',help='fail on a cache miss instead of computing features'); ap.add_argument('--device',default='auto'); ap.add_argument('--amp',action='store_true',help='CUDA mixed precision'); ap.add_argument('--exact-fps',action='store_true',help='use slow canonical farthest-point sampling in PointNet2'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args(); seed_everything(args.seed)
    out=Path(args.output_dir)/f'{args.dataset}__{args.model}__{args.feature_set}__fk{args.fractal_k}'; ensure_dir(out)
    train,test=load_dataset(args); xtr,ytr=train.points,train.labels; xte,yte=test.points,test.labels
    if args.smoke: xtr,ytr=xtr[:min(24,len(xtr))],ytr[:min(24,len(ytr))]; xte,yte=xte[:min(12,len(xte))],yte[:min(12,len(yte))]; args.epochs=min(args.epochs,2); args.batch_size=min(args.batch_size,4); args.n_points=min(args.n_points,256); xtr=xtr[:,:args.n_points]; xte=xte[:,:args.n_points]
    need_dc='dc' in args.feature_set; need_db='db' in args.feature_set; dc_tr=db_tr=dc_te=db_te=None
    if need_dc or need_db:
        cache=cache_base(args.cache_dir,args.dataset,xtr.shape[1])
        precompute_split(xtr,cache/'train', [args.fractal_k], args.fractal_centers,args.seed,'train',require_cached=args.require_fractal_cache)
        precompute_split(xte,cache/'test', [args.fractal_k], args.fractal_centers,args.seed,'test',require_cached=args.require_fractal_cache)
        if need_dc: dc_tr=np.load(cache/'train'/f'dc_k{args.fractal_k}.npy')[:len(xtr)]; dc_te=np.load(cache/'test'/f'dc_k{args.fractal_k}.npy')[:len(xte)]
        if need_db: db_tr=np.load(cache/'train'/f'db_k{args.fractal_k}.npy')[:len(xtr)]; db_te=np.load(cache/'test'/f'db_k{args.fractal_k}.npy')[:len(xte)]
    idx=np.arange(len(ytr)); ncls=len(np.unique(ytr)); val_n=max(1,int(round(len(ytr)*args.val_fraction))); stratify=ytr if ncls>1 and val_n>=ncls and np.min(np.bincount(ytr))>=2 else None; train_idx,val_idx=train_test_split(idx,test_size=args.val_fraction,random_state=args.seed,stratify=stratify)
    ds_tr=PCDataset(xtr,ytr,dc_tr,db_tr,train_idx,id_offset=0); ds_val=PCDataset(xtr,ytr,dc_tr,db_tr,val_idx,id_offset=0); ds_te=PCDataset(xte,yte,dc_te,db_te,id_offset=len(xtr))
    pin=args.device.startswith('cuda') or (args.device=='auto' and torch.cuda.is_available()); workers=args.workers
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=True,num_workers=workers,pin_memory=pin,drop_last=(len(ds_tr)>args.batch_size and args.batch_size>1)); dl_val=DataLoader(ds_val,batch_size=args.batch_size,shuffle=False,num_workers=workers,pin_memory=pin); dl_te=DataLoader(ds_te,batch_size=args.batch_size,shuffle=False,num_workers=workers,pin_memory=pin)
    in_ch=3+int(need_dc)+int(need_db)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cuda' if args.device.startswith('cuda') else 'cpu'))
    model=make_model(args.model,in_ch,len(train.class_names),args.graph_k,args.n_points,args.exact_fps).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1)); ce=nn.CrossEntropyLoss()
    amp_enabled=args.amp and device.type=='cuda'
    try:
        scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    except AttributeError:
        scaler=torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history=[]; best=-1; best_path=out/'model_best.pt'; t0=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); s=0.; n=0; ep0=time.perf_counter()
        for x,y,ids in dl_tr:
            x=x.to(device,non_blocking=pin); y=y.to(device,non_blocking=pin); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(args.amp and device.type=='cuda')):
                loss=ce(model_forward(model,x,ids),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); s+=float(loss.detach())*len(y); n+=len(y)
        sched.step(); va=eval_epoch(model,dl_val,device); row={'epoch':epoch,'train_loss':s/max(n,1),**{k:v for k,v in va.items() if k not in ('y_true','y_pred')},'lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-ep0}; history.append(row); print(row,flush=True)
        if va['macro_f1']>best: best=va['macro_f1']; torch.save({'model':model.state_dict(),'args':vars(args),'class_names':train.class_names},best_path)
    state=torch.load(best_path,map_location=device); model.load_state_dict(state['model']); te=eval_epoch(model,dl_te,device); total=time.perf_counter()-t0
    all_labels=np.arange(len(train.class_names),dtype=np.int64)
    cm=confusion_matrix(te['y_true'],te['y_pred'],labels=all_labels)
    np.savetxt(out/'confusion_matrix_test.csv',cm.astype(int),fmt='%d',delimiter=',')
    write_csv(out/'history.csv',history,list(history[0].keys()) if history else ['epoch'])
    report=classification_report(te['y_true'],te['y_pred'],labels=all_labels,target_names=train.class_names,output_dict=True,zero_division=0)
    report_rows=[{'class':k,**v} for k,v in report.items() if isinstance(v,dict)]
    write_csv(out/'classification_report_test.csv',report_rows,['class','precision','recall','f1-score','support'])
    metrics={'dataset':args.dataset,'model':args.model,'feature_set':args.feature_set,'fractal_k':args.fractal_k,'fractal_centers':args.fractal_centers,'n_points':args.n_points,'n_train':len(xtr),'n_test':len(xte),'device':str(device),'params':sum(p.numel() for p in model.parameters()),'test_accuracy':te['accuracy'],'test_macro_f1':te['macro_f1'],'test_balanced_accuracy':te['balanced_accuracy'],'training_seconds':total,'best_val_macro_f1':best}
    save_json(out/'metrics.json',metrics); save_json(out/'config.json',vars(args)); save_json(out/'class_mapping.json',{str(i):n for i,n in enumerate(train.class_names)})
    summary_path=Path(args.output_dir)/'summary.csv'; fields=list(metrics.keys())
    rows=[]
    if summary_path.exists():
        with summary_path.open('r',newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        fields=list(dict.fromkeys(list(rows[0].keys()) if rows else []) + fields)
    rows.append({k:metrics.get(k,'') for k in fields})
    write_csv(summary_path,rows,fields)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
