from turtle import forward
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from Fast_MRI_utility import *

def softthreshold(x,shrinkage):
    return torch.sgn(x)*torch.relu(torch.abs(x)-shrinkage)

class LISTA(nn.Module):
    def __init__(self, K) -> None:
        super(LISTA,self).__init__()

        #self.conv1 = nn.Conv2d(1,1,padding=1,kernel_size=3)
        #self.conv2 = nn.Conv2d(1,1,padding=1,kernel_size=3)
        self.conv1 = nn.ModuleList(nn.Conv2d(1,1,padding=1,kernel_size=3) for i in range(K))
        self.conv2 = nn.ModuleList(nn.Conv2d(1,1,padding=1,kernel_size=3) for i in range(K))
        self.shrinkage = nn.Parameter(torch.zeros(3),requires_grad=True)
        self.K = K

    def forward(self, y):
        x = torch.zeros((y.size(0),1,32,32))

        for id in range(self.K):
            x = self.conv1[id](x)
            y = self.conv2[id](y)
            x = x + y
            x = x + (torch.sqrt((x-self.shrinkage[id])**2 + 1) - torch.sqrt((x + self.shrinkage[id])**2 + 1)) / 2
    
        return x

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,16,padding=1,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,64,padding=1,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,padding=1,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1,padding=1,kernel_size=3),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

def ISTA_MRI(p_kspace, mask, mu, K=3, shrinkage=0.1):
    # accelerated MRI
    rcst_MRI = recreate_MRI(p_kspace)
    # create full kpace from accelerated MRI
    x_k = kspace(rcst_MRI)
    for iter in range(K):
        x_k = (1 - mu * mask) * x_k + mu * p_kspace
        x_k = softthreshold(recreate_MRI(x_k).abs(), shrinkage)
        x_k = kspace(x_k)
    return recreate_MRI(x_k)

class LISTA_MRI(nn.Module):
    def __init__(self, K, mu) -> None:
        super(LISTA_MRI, self).__init__()
        self.conv_list = nn.ModuleList(ConvNet() for i in range(K))
        self.mu = mu

    def forward(self, ksp, mask):
        # accelerated MRI
        rcst_MRI = recreate_MRI(ksp)
        # create full kpace from accelerated MRI
        x_k = kspace(rcst_MRI)
        for proximal in self.conv_list:
            x_k = (1 - self.mu * mask) * x_k + self.mu * ksp
            x_k = proximal(recreate_MRI(x_k).abs().unsqueeze(1))
            x_k = kspace(x_k).squeeze()
        return recreate_MRI(x_k).abs()
        