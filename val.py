import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from backbone.model import simpleMLP
from loss.loss import mse,MAPE
import torch.nn.functional as F
import time

use_pretrain = True
backbone = simpleMLP(in_channels=7,
                     # hidden_channels=[1024,2048,4096,1024,512,18],
                     hidden_channels=[16, 32, 64, 128, 18],
                     norm_layer=nn.BatchNorm1d,
                     dropout=0, inplace=False, use_sigmoid=False).cuda()

if use_pretrain:
    weights_pth = 'final.pt'
    try:
        backbone.load_state_dict(torch.load(weights_pth))
    except:
        print(f'No {weights_pth}')

# backbone = backbone.eval()
a  = [ 3, 103.5, 164, 121, 4250, 25, 24565]
b = [ 2, 86.6, 92, 58, 4800, 54, 6479]

Softmax = nn.Softmax()
a = torch.tensor([a]).cuda()
b = torch.tensor([b]).cuda()
backbone.eval()
print(a.shape)
with torch.no_grad():
    r = backbone(a)
    r = torch.argmax(r)
    print(r)
    r = backbone(b)
    r = torch.argmax(r)
    print(r)