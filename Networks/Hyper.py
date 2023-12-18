import torch
import torch.nn as nn
import math
from .GDN import *
import torch.nn.functional as F

from .Custom_Function import *

class Analysis_Hyper(torch.nn.Module):
    def __init__(self,filters=192,out_filters=320):
        super(Analysis_Hyper,self).__init__()
        
        self.x1 = nn.Conv2d(out_filters,filters, 3, stride=1,padding=1)
        self.x2 = nn.Conv2d(filters, filters, 5, stride=2, padding=2)
        self.x3 = nn.Conv2d(filters, filters, 5, stride=2, padding=2)
        """
        self.x1 = CustomConv2DPyMV3(out_filters, filters, 3, stride=1, padding=1)
        self.x2 = CustomConv2DPyMV3(filters, filters, 5, stride=2, padding=2)
        self.x3 =  CustomConv2DPyMV3(filters, filters, 5, stride=2, padding=2)
        """
    def forward(self, inputs, mask1,mask2,mask3):
        
        x = torch.abs(inputs)
        x = F.relu(self.x1(x),inplace=True)
        x = F.relu(self.x2(x),inplace=True)
        x = self.x3(x)
        """
        x = torch.abs(inputs)
        x = F.relu(self.x1(x,mask1),inplace=True)
        x = F.relu(self.x2(x,mask1),inplace=True)
        x = self.x3(x,mask2)
        """
        return x 

class AnalysisMask_Hyper(torch.nn.Module):
    def __init__(self,filters=192,out_filters=320):
        super(AnalysisMask_Hyper,self).__init__()
        self.x1 = nn.Conv2d(out_filters,filters, 3, stride=1)
        self.x2 = nn.Conv2d(filters, filters, 5, stride=2, padding=2)
        self.x3 = nn.Conv2d(filters, filters, 5, stride=2, padding=2)
        
    def forward(self, inputs):
        y = torch.abs(inputs)
        y1 = F.relu(self.x1(y),inplace=True)
        y2 = F.relu(self.x2(y1),inplace=True)
        y3 = self.x3(y2)
        return y3
    
class Synthesis_Hyper(torch.nn.Module):
    def __init__(self,filters=192,out_filters=320):
        super(Synthesis_Hyper,self).__init__()
        self.x1 = nn.ConvTranspose2d(filters,filters,5,stride=2,padding=2,output_padding=1)
        self.x2 = nn.ConvTranspose2d(filters,filters,5,stride=2,padding=2,output_padding=1)
        self.x3 = nn.ConvTranspose2d(filters,out_filters,3,stride=1,padding=1,output_padding=0)
        
    def forward(self, inputs,mask1,mask2,mask3):
        x = F.relu(self.x1(inputs),inplace=True)
        x = F.relu(self.x2(x),inplace=True)
        x = F.relu(self.x3(x),inplace=True)
        return x

class SynthesisMask_Hyper(torch.nn.Module):
    def __init__(self,filters=192,out_filters=320):
        super(SynthesisMask_Hyper,self).__init__()
        self.x1 = nn.ConvTranspose2d(filters,filters,5,stride=2,padding=2,output_padding=1)
        self.x2 = nn.ConvTranspose2d(filters,filters,5,stride=2,padding=2,output_padding=1)
        self.x3 = nn.ConvTranspose2d(filters,out_filters,3,stride=1,padding=1,output_padding=0)
        
    def forward(self, inputs):
        y1 = F.relu(self.x1(inputs),inplace=True)
        y2 = F.relu(self.x2(y1),inplace=True)
        y3 = F.relu(self.x3(y2),inplace=True)
        
        return y3
    
    


