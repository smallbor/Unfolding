from turtle import forward
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable


class LISTA(nn.Module):
    def __init__(self, K) -> None:
        super(LISTA,self).__init__()

        self.conv1 = nn.Conv2d(1,1,padding=1,kernel_size=3)
        self.conv2 = nn.Conv2d(1,1,padding=1,kernel_size=3)
        self.shrinkage = Variable(torch.zeros(3), requires_grad=True)
        self.K = K

    def forward(self, y):
        x = torch.zeros((y.size(0),1,32,32))

        for id in range(3):
            x = self.conv1(x)
            y = self.conv2(y)
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