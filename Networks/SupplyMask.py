import torch
import torch.nn as nn
import math
from .GDN import *
import torch.nn.functional as F

class SupplyMaskToTransform(torch.nn.Module):
    def __init__(self,kernel=3):
        super(SupplyMaskToTransform,self).__init__()
        self.pool = nn.AvgPool2d(kernel, stride=2,padding=1)
    def forward(self, inputs):
        mask1 = self.pool(inputs)#(256→128)
        mask2 = self.pool(mask1)#(128→64)
        mask3 = self.pool(mask2)#(64→32)
        mask4 = self.pool(mask3)#(32→16)
        mask5 = self.pool(mask4)
        mask6 = self.pool(mask5)
        return mask1,mask2,mask3,mask4,mask5,mask6



