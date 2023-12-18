import torch
import torch.nn as nn
import math
from .GDN import *
import torch.nn.functional as F
import torchvision
from .Custom_Function import *
from .Attention import *
from .Custom_Function import ResidualBlockM,ResidualBlockWithStrideM,ResidualBlockUpsampleM,conv3x3M
from .AttentionMask import *

class Conv_GDN_MaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2,padding=2):
        super(Conv_GDN_MaxPool, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.gdn = GDN(out_channels)

    def forward(self, x):
        conved = self.conv1x1(x)
        y = self.gdn(self.conv(x))
        return y + conved

class ConvT_IGDN_MaxUnPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2,padding=2,output_padding=1):
        super(ConvT_IGDN_MaxUnPool, self).__init__()
        self.conv1x1 = nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, padding=0,output_padding=1)
     
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2,output_padding=output_padding)
        self.igdn = GDN(in_channels,inverse=True)

    def forward(self, x):
        x = self.igdn(x)
        y = self.convt(x)
        conved = self.conv1x1(x)
        
        
        return y + conved

class AnalysisMask_transform(torch.nn.Module):
    def __init__(self,N=32,M=320):
        super(AnalysisMask_transform,self).__init__() 
        
        self.x1 = nn.Conv2d(1, N, 5, stride=2, padding=2)
        self.gdn1 = GDN(N)
        self.x2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn2 = GDN(N)
        self.x3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn3 = GDN(N)
        self.x4 = nn.Conv2d(N, M, 5, stride=2,padding=2)
        """
        self.x1 = Conv_GDN_MaxPool(1, N, 5, stride=2, padding=2)
        self.x2 = Conv_GDN_MaxPool(N, N, 5, stride=2, padding=2)
        self.x3 = Conv_GDN_MaxPool(N, N, 5, stride=2, padding=2)
        self.x4 = nn.Conv2d(N, M, 1, stride=1,padding=0)
        """
    def forward(self, inputs):
        
        y1 = self.gdn1(self.x1(inputs))
        y2 = self.gdn2(self.x2(y1))
        y3 = self.gdn3(self.x3(y2))
        y4 = self.x4(y3)
        """
        y1 = self.x1(inputs)
        y2 = self.x2(y1)
        y3 = self.x3(y2)
        y4 = self.x4(y3)
        """
        return y4

    
class SynthesisMask_transform(torch.nn.Module):
    def __init__(self,N=32,M=320):
        super(SynthesisMask_transform, self).__init__()
        
        self.x1 = nn.ConvTranspose2d(M,N,5,stride=2,padding=2,output_padding=1)
        self.igdn1 = GDN(N,inverse=True)
        self.x2 = nn.ConvTranspose2d(N,N,5,stride=2,padding=2,output_padding=1)
        self.igdn2 = GDN(N,inverse=True)
        self.x3 = nn.ConvTranspose2d(N,N,5,stride=2,padding=2,output_padding=1)
        self.igdn3 = GDN(N,inverse=True)
        self.x4 = nn.ConvTranspose2d(N,1,5,stride=2,padding=2,output_padding=1)
        """
        self.x1 = nn.ConvTranspose2d(M,N,1,stride=1,padding=0,output_padding=0)
        self.x2 = ConvT_IGDN_MaxUnPool(N,N,5,stride=2,padding=2,output_padding=1)
        self.x3 = ConvT_IGDN_MaxUnPool(N,N,5,stride=2,padding=2,output_padding=1)
        self.x4 = ConvT_IGDN_MaxUnPool(N,1,5,stride=2,padding=2,output_padding=1)
        self.DSE = DSE()
        """
    def forward(self, inputs):
        y = self.igdn1(self.x1(inputs))
        y = self.igdn2(self.x2(y))
        y = self.igdn3(self.x3(y))
        y = self.x4(y)
        """
        y = self.x1(inputs)
        y = self.x2(y)
        y = self.x3(y)
        y = self.x4(y)
        y = self.DSE(y)
        """
        return y