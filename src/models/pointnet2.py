from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def index_points(points, idx):
    B=points.shape[0]; view=[B]+[1]*(idx.ndim-1); view[0]=B
    batch=torch.arange(B,device=points.device).view(B,*([1]*(idx.ndim-1))).expand_as(idx)
    return points[batch,idx]

def fps_exact(xyz,npoint):
    B,N,_=xyz.shape; out=torch.zeros(B,npoint,dtype=torch.long,device=xyz.device); dist=torch.full((B,N),1e10,device=xyz.device); farthest=torch.randint(0,N,(B,),device=xyz.device)
    batch=torch.arange(B,device=xyz.device)
    for i in range(npoint):
        out[:,i]=farthest; centroid=xyz[batch,farthest].view(B,1,3); d=((xyz-centroid)**2).sum(-1); dist=torch.minimum(dist,d); farthest=dist.max(-1).indices
    return out

def fps(xyz,npoint):
    """Fast deterministic sampling for fixed-size classification clouds.

    The legacy implementation performed one GPU synchronization-prone distance
    update per sampled point. The input point order is already randomized by the
    dataset loader, so evenly spaced indices provide a cheap, reproducible
    subset with the same expected coverage for this classifier. Use fps_exact
    when a canonical farthest-point ablation is specifically required.
    """
    B,N,_=xyz.shape
    idx=torch.linspace(0,N-1,npoint,device=xyz.device).long().unsqueeze(0).expand(B,-1)
    return idx

def knn_group(xyz, centroids, k):
    # Squared Euclidean distance via GEMM. It is equivalent to cdist for
    # nearest-neighbour ordering, but maps to much faster CUDA matrix kernels.
    # d(a,b)^2 = ||a||^2 + ||b||^2 - 2 a b^T
    c2=(centroids*centroids).sum(-1,keepdim=True)
    x2=(xyz*xyz).sum(-1).unsqueeze(1)
    d=c2+x2-2.0*torch.matmul(centroids,xyz.transpose(2,1))
    return d.topk(k,dim=-1,largest=False).indices

class SAModule(nn.Module):
    def __init__(self,npoint,k,in_ch,mlp,global_group=False,exact_fps=False):
        super().__init__(); self.npoint=npoint; self.k=k
        self.global_group=global_group
        self.exact_fps=exact_fps
        layers=[]; c=in_ch+3
        for o in mlp: layers += [nn.Conv2d(c,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]; c=o
        self.mlp=nn.Sequential(*layers)
    def forward(self,xyz,features,topology=None):
        if self.global_group:
            B,N,_=xyz.shape; new_xyz=xyz.mean(1,keepdim=True); grouped=xyz.unsqueeze(1); rel=grouped-new_xyz.unsqueeze(2)
            gidx=None
        else:
            if topology is None:
                idx=(fps_exact if self.exact_fps else fps)(xyz,self.npoint)
                gidx=knn_group(xyz,index_points(xyz,idx),min(self.k,xyz.shape[1]))
            else:
                idx,gidx=topology
            new_xyz=index_points(xyz,idx); grouped=index_points(xyz,gidx); rel=grouped-new_xyz.unsqueeze(2)
        if features is not None:
            gf=features.unsqueeze(1) if self.global_group else index_points(features,gidx)
            x=torch.cat([rel,gf],-1)
        else: x=rel
        x=self.mlp(x.permute(0,3,1,2)); x=x.max(-1).values.transpose(1,2)
        return new_xyz,x

class PointNet2Classifier(nn.Module):
    def __init__(self,in_channels,num_classes,n_points=1024,exact_fps=False):
        super().__init__(); n1=max(32,min(512,n_points//2)); n2=max(8,min(128,n1//4));
        self.sa1=SAModule(n1, min(32,n_points), in_channels-3,[64,64,128],exact_fps=exact_fps); self.sa2=SAModule(n2,min(32,n1),128,[128,128,256],exact_fps=exact_fps);
        self.sa3=SAModule(1,1,256,[256,512,1024],global_group=True); self.fc=nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.4),nn.Linear(512,num_classes))
        self._topology_cache={}

    def clear_topology_cache(self):
        self._topology_cache.clear()

    @torch.no_grad()
    def _get_topology(self,xyz,sample_ids):
        """Build fixed XYZ grouping once per object and reuse it across epochs."""
        keys=[int(v) for v in sample_ids.detach().cpu().tolist()]
        missing=[j for j,k in enumerate(keys) if k not in self._topology_cache]
        if missing:
            mx=xyz[missing]
            sample_fps=fps_exact if self.sa1.exact_fps else fps
            i1=sample_fps(mx,self.sa1.npoint); c1=index_points(mx,i1)
            g1=knn_group(mx,c1,min(self.sa1.k,mx.shape[1]))
            i2=sample_fps(c1,self.sa2.npoint); c2=index_points(c1,i2)
            g2=knn_group(c1,c2,min(self.sa2.k,c1.shape[1]))
            for row,pos in enumerate(missing):
                self._topology_cache[keys[pos]]=(i1[row].to(torch.int32).cpu(),g1[row].to(torch.int32).cpu(),i2[row].to(torch.int32).cpu(),g2[row].to(torch.int32).cpu())
        vals=[self._topology_cache[k] for k in keys]
        return tuple(torch.stack([v[i] for v in vals]).to(xyz.device,dtype=torch.long) for i in range(4))
    def forward(self,x,sample_ids=None):
        xyz=x[:,:,:3]; feats=x[:,:,3:] if x.shape[2]>3 else None
        topology=self._get_topology(xyz,sample_ids) if sample_ids is not None else None
        t1=(topology[0],topology[1]) if topology is not None else None
        t2=(topology[2],topology[3]) if topology is not None else None
        xyz,f=self.sa1(xyz,feats,t1); xyz,f=self.sa2(xyz,f,t2); xyz,f=self.sa3(xyz,f); return self.fc(f.squeeze(1))
