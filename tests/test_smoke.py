import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch, numpy as np
from src.models.pointnet2 import PointNet2Classifier
from src.models.dgcnn import DGCNNClassifier

def test_pointnet2_forward_backward():
    x=torch.randn(2,256,5); y=torch.tensor([0,1]); m=PointNet2Classifier(5,3,256); z=m(x); assert z.shape==(2,3); z.sum().backward()

def test_dgcnn_forward_backward():
    x=torch.randn(2,256,5); m=DGCNNClassifier(5,3,10); z=m(x); assert z.shape==(2,3); z.sum().backward()
