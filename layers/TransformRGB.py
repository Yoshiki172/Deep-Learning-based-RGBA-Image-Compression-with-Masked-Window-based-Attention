import torch
import torch.nn as nn
import math
from .GDN import *
import torch.nn.functional as F
import torchvision
# from .Custom_Function import *
from .Masked_Attention import *

#from .Attention import *
from compressai.layers import (
    conv3x3,
    subpel_conv3x3,
)
# import custom_conv2d
class EnhancementBlock(nn.Module):
    def __init__(self, num_filters=32):
        super(EnhancementBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + input
        return x

class DSE(nn.Module):
    def __init__(self, num_filters=32):
        super(DSE, self).__init__()
        self.input_conv = nn.Conv2d(3, num_filters, 1, stride=1)
        self.enh1 = EnhancementBlock(num_filters)
        self.enh2 = EnhancementBlock(num_filters)
        self.enh3 = EnhancementBlock(num_filters)
        self.output_conv = nn.Conv2d(num_filters, 3, 1, stride=1)

    def forward(self, input):
        #input = input.to('cpu')
        identity  = input
        x_first = self.input_conv(input)
        x = self.enh1(x_first)
        x = self.enh2(x)
        x = self.enh3(x)
        x = x + x_first
        x = self.output_conv(x)
        x = x + identity
        return x


class Analysis_transform(torch.nn.Module):
    def __init__(self,N=192,M=320):
        super(Analysis_transform,self).__init__()
        self.x1 = nn.Conv2d(3, N, 5, stride=2, padding=2)
        self.gdn1 = GDN(N)
        self.x2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn2 = GDN(N)
        self.attention1 = Win_noShift_Attention(dim=N,num_heads=8,window_size=8,shift_size=4)
        self.x3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn3 = GDN(N)
        self.x4 = nn.Conv2d(N, M, 1, stride=1,padding=0)
        self.attention2 = Win_noShift_Attention(dim=M,num_heads=8,window_size=4,shift_size=2)
        
    def forward(self, input,mask,me1,me2,me3,me4):
        y = self.gdn1(self.x1(input))
        y = self.gdn2(self.x2(y))
        y = self.attention1(y,me2)
        #y = self.attention1(y)
        y = self.gdn3(self.x3(y))
        y = self.x4(y)
        y = self.attention2(y,me3)
        #y = self.attention2(y)
        
        return y

class Synthesis_transform(torch.nn.Module):
    def __init__(self,N=196,M=320):
        super(Synthesis_transform, self).__init__()
        self.attention1 = Win_noShift_Attention(dim=M,num_heads=8,window_size=4,shift_size=2)
        self.x1 = nn.Conv2d(M, N, 1, stride=1,padding=0)
        self.igdn1 = GDN(N,inverse=True)
        self.x2 = nn.ConvTranspose2d(N,N,5,stride=2,padding=2,output_padding=1)
        self.igdn2 = GDN(N,inverse=True)
        self.attention2 = Win_noShift_Attention(N,num_heads=8,window_size=8,shift_size=4)
        self.x3 = nn.ConvTranspose2d(N,N,5,stride=2,padding=2,output_padding=1)
        self.igdn3 = GDN(N,inverse=True)
        self.x4 = nn.ConvTranspose2d(N,3,5,stride=2,padding=2,output_padding=1)
        self.dse = DSE(32)
    def forward(self, input,reconmask,md1,md2,md3,md4):
        y = self.attention1(input,md3)
        #y = self.attention1(input)
        y = self.igdn1(self.x1(y))
        y = self.igdn2(self.x2(y))
        y = self.attention2(y,md2)
        #y = self.attention2(y)
        y = self.igdn3(self.x3(y))
        y = self.x4(y)
        y = self.dse(y)
        return y
    
class HyperAnalysis(nn.Module):
    """
    Local reference
    """
    def __init__(self, M=192, N=192):
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        x = self.reduction(x)

        return x

class HyperSynthesis(nn.Module):
    """
    Local Reference
    """
    def __init__(self, M=192, N=192) -> None:
        super().__init__()
        self.M = M
        self.N = N

        self.increase = nn.Sequential(
            subpel_conv3x3(N, M, 2),
            nn.GELU(),
            conv3x3(M, M),
            nn.GELU(),
            subpel_conv3x3(M, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 2, 2),
        )

    def forward(self, x):
        x = self.increase(x)

        return x

