import torch
import math
import torch.nn as nn
import torch.nn.functional as f
from .Custom_Function import *
from .win_attention import *
from compressai.layers import (
    conv3x3,
)
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    
class Win_noShift_Attention_with_mask(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0,phase="Encoder"):
        super().__init__()
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

        self.phase = phase
        self.WBA = WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size)
        self.RU1 = ResidualUnit()
        self.RU2 = ResidualUnit()
        self.RU3 = ResidualUnit()
        self.conv = conv1x1(N, N)

    def insert_zeros(self,x,mask):
        mask = mask.expand(-1, x.shape[1], -1, -1)
        if self.phase == "Encoder":
            mask_one = (mask > 0.0).float()
        else:
            mask_one = (mask > 0.1).float()
        
        x = x * mask_one
        return x
    
    def forward(self, x, mask):
        identity = x
        a = self.conv_a(x)
        
        b = self.insert_zeros(x,mask)
        b = self.WBA(b)
        b = self.RU1(b)
        b = self.RU2(b)
        b = self.RU3(b)
        b = self.conv(b)
        
        #b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out