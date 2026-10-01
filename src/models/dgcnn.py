from __future__ import annotations
import torch
from torch import nn

def knn(x,k):
    # Neighbor selection must stay in FP32 even under AMP (avoid FP16 overflow).
    with torch.no_grad(), torch.autocast(device_type=x.device.type,enabled=False):
        x=x.float()
        inner=-2*torch.matmul(x.transpose(2,1),x); xx=(x*x).sum(1,keepdim=True); d=-(-xx-inner-xx.transpose(2,1)); return d.topk(k,dim=-1).indices

def graph_feature(x,k):
    B,C,N=x.shape; idx=knn(x,k); device=x.device; base=torch.arange(B,device=device).view(B,1,1)*N; idx=(idx+base).reshape(-1)
    xt=x.transpose(2,1).contiguous(); neigh=xt.reshape(B*N,C)[idx].view(B,N,k,C); center=xt.view(B,N,1,C).expand(-1,-1,k,-1)
    return torch.cat((neigh-center,center),-1).permute(0,3,1,2).contiguous()

class DGCNNClassifier(nn.Module):
    def __init__(self,in_channels,num_classes,graph_k=20):
        super().__init__(); self.k=graph_k
        self.c1=nn.Sequential(nn.Conv2d(in_channels*2,64,1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(.2));
        self.c2=nn.Sequential(nn.Conv2d(128,64,1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(.2));
        self.c3=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.LeakyReLU(.2));
        self.c4=nn.Sequential(nn.Conv2d(256,256,1,bias=False),nn.BatchNorm2d(256),nn.LeakyReLU(.2));
        self.c5=nn.Sequential(nn.Conv1d(64+64+128+256,1024,1,bias=False),nn.BatchNorm1d(1024),nn.LeakyReLU(.2));
        self.fc=nn.Sequential(nn.Linear(2048,512,bias=False),nn.BatchNorm1d(512),nn.LeakyReLU(.2),nn.Dropout(.5),nn.Linear(512,num_classes))
    def forward(self,x):
        x=x.transpose(1,2); x1=self.c1(graph_feature(x,self.k)).max(-1).values; x2=self.c2(graph_feature(x1,self.k)).max(-1).values; x3=self.c3(graph_feature(x2,self.k)).max(-1).values; x4=self.c4(graph_feature(x3,self.k)).max(-1).values; x=self.c5(torch.cat((x1,x2,x3,x4),1)); return self.fc(torch.cat((x.max(2).values,x.mean(2)),1))
