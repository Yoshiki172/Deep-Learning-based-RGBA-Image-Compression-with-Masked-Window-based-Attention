import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
#Attention this function only supported for 4 batches 
from pytorch_memlab import profile

class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(MyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalConv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.biasbool, self.padding_mode, self.device, self.dtype).to(self.device)
        #self.weight = self.normalConv.weight.data
        self.weight = torch.nn.Parameter(self.normalConv.weight.data, requires_grad=True)
        if self.biasbool == True:
            self.bias = torch.nn.Parameter(self.normalConv.bias.data, requires_grad=True)
        else:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_channels).to(self.device), requires_grad=True)
        
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation).to(self.device)
    #@profile
    def forward(self, x, mask):
        input = x.clone().detach()
        batch, in_channels, in_height, in_width = x.size()
        _, _, kernel_height, kernel_width = self.weight.size()
        
        x = x.to(self.device)
        mask = mask.to(self.device)
        mask = self.UnFold(mask)
        mask = mask.transpose_(1,2)
        x = self.UnFold(x)
        x = x.transpose_(1,2)
        
        mask = torch.any(mask != 0, dim=2)
        summask = torch.sum(mask, dim=1)
  
        x = x[mask].unsqueeze(dim=0)
        x = x.permute(0,2,1)
        
        weight_view = self.weight.view((self.out_channels,-1))
        bias_view = self.bias.view(1, self.bias.shape[0], 1).to(self.device)
        x = torch.matmul(weight_view, x) + bias_view
        
        del weight_view#,bias_view
        
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, self.out_channels, mask.shape[1])

        mask_new = []
        zeros_new = []
        new_tensor = []
        
        start = 0
        
        for i in range(batch):
            mask_new.append(mask[i,0:self.out_channels,0:mask.shape[2]])
            #zeros_new.append(zeros[i,0:self.out_channels,0:mask[0].shape[1]])
            zeros_new.append(torch.zeros(self.out_channels, mask.shape[2]).to(self.device))
            end = start + summask[i] 
            new_tensor.append(x[0:1,0:self.out_channels,start:end])
            start = end
            zeros_new[i].masked_scatter_(mask_new[i],new_tensor[i]) 
      
        del mask_new,new_tensor,summask
        x = torch.stack(zeros_new)
        
        out_height = (in_height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1
        x = x.reshape(batch,self.out_channels,out_height,out_width)
        
        return x
"""
class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(MyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalConv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.biasbool, self.padding_mode, self.device, self.dtype).to(self.device)
        
    #@profile
    def forward(self, x):
        x = self.normalConv(x)
        return x
"""
"""
class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)).to(self.device)
        nn.init.kaiming_uniform_(self.weight, a=0, nonlinearity='relu')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, alpha):
        _, _, h, w = input.size()
        input_unfold = F.unfold(input, self.kernel_size, padding=self.padding, stride=self.stride).to(self.device)
        alpha_unfold = F.unfold(alpha.repeat(1, self.in_channels, 1, 1), self.kernel_size, padding=self.padding, stride=self.stride).to(self.device)
        input_unfold = input_unfold * alpha_unfold
        out_unfold = input_unfold.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        print('out_unfold:',out_unfold.shape)
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = F.fold(out_unfold, (out_h, out_w), (1, 1), padding=0, stride=1)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        
        return out
"""
  
class MyConvT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        super(MyConvT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding        
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.biaspool = bias
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
          assert self.stride > self.output_padding
        except AssertionError:
          import sys
          sys.exit("Error: Output padding must be smaller than either stride")
        self.normalConv = torch.nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups, self.biasbool, self.dilation, self.padding_mode, self.device, self.dtype).to(self.device)
        self.weight = self.normalConv.weight.data
        
        if self.biasbool == True:
            self.bias = self.normalConv.bias.data
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
        
        #Attention padding value is always 0
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=0, dilation=self.dilation).to(self.device)
    
    def insert_zeros(self,input):
      out = torch.zeros(input.shape[0],input.shape[1],input.shape[2]+input.shape[2]*(self.stride-1),input.shape[3]+input.shape[3]*(self.stride-1)).to(self.device)
      out[:,:,::self.stride,::self.stride] = input.clone()
      out = out[:,:,0:out.shape[2]-self.stride+1+self.output_padding,0:out.shape[3]-self.stride+1+self.output_padding]
      
      return out
    
    def forward(self, x, mask):
        batch, in_channels, in_height, in_width = x.size()
        _, _, kernel_height, kernel_width = self.weight.size()
        

        if self.stride > 1:
          x = self.insert_zeros(x)
        
        #x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = F.pad(x, (self.kernel_size-1,self.kernel_size-1,self.kernel_size-1,self.kernel_size-1), mode='constant', value=0)
        x = x[:,:,self.padding:x.shape[2]-self.padding,self.padding:x.shape[3]-self.padding]
        
        mask = self.UnFold(mask)
        mask = mask.transpose_(1,2)
        
        x = self.UnFold(x)
        x = x.transpose_(1,2)
        
        self.weight = self.weight.transpose(0,1)
        self.weight = torch.rot90(self.weight,2,[2,3])
        
        #print("rotate W :",self.weight)
        
        mask = torch.any(mask != 0, dim=2)
        summask = torch.sum(mask, dim=1)
        
        x = x[mask].unsqueeze(dim=0)
        
        x = x.permute(0,2,1)
        
        #x = (x.matmul(self.weight.view(self.weight.size(0), -1).t())+self.bias).transpose(1, 2)
        x = torch.matmul(self.weight.reshape((self.out_channels,-1)), x)+self.bias.reshape(1,self.bias.shape[0],1)
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, self.out_channels, mask.shape[1])
        zeros = torch.zeros(mask.shape[0],mask.shape[1],mask.shape[2]).to(self.device)
        
        mask_new = []
        zeros_new = []
        new_tensor = []
        
        start = 0
        
        with torch.no_grad():
            for i in range(batch):
                mask_new.append(mask[i,0:self.out_channels,0:mask.shape[2]])
                zeros_new.append(zeros[i,0:self.out_channels,0:mask[0].shape[1]])
                end = start + summask[i] 
                new_tensor.append(x[0:1,0:self.out_channels,start:end])
                start = end
                zeros_new[i].masked_scatter_(mask_new[i],new_tensor[i])         
        
        
        x = torch.stack(zeros_new)
        
        #x = x.reshape((4,self.out_channels,int(math.sqrt(x.shape[2])),int(math.sqrt(x.shape[2]))))
        
        output_height = (in_height - 1) * self.stride + kernel_height - 2 * self.padding + self.output_padding
        output_width = (in_width - 1) * self.stride + kernel_width - 2 * self.padding + self.output_padding

        x = x.reshape((batch,self.out_channels,output_height,output_width))
        
        return x

import torch
from torch.nn.parameter import Parameter

if __name__ == '__main__':
    """
    input = torch.tensor([[[[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                            [0, 2, 2, 2, 0],
                            [0, 2, 2, 2, 0],
                            [0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0]],
                            
                            [[0, 0, 0, 0, 0],
                            [0, 3, 3, 3, 0],
                            [0, 3, 3, 3, 0],
                            [0, 3, 3, 3, 0],
                            [0, 0, 0, 0, 0]]],
                          
                          [[[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],
                            
                            [[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]]],
                          
                          [[[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],
                            
                            [[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]]],
                          
                          [[[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],
                            
                            [[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]]]]).float()
    """
    input = torch.tensor([[[[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]],

                            [[2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2]],
                            
                            [[3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3]]],
                          
                          [[[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]],

                            [[2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2]],
                            
                            [[3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3]]],
                          
                          [[[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]],

                            [[2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2]],
                            
                            [[3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3]]],
                          
                          [[[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]],

                            [[2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2, 2]],
                            
                            [[3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3],
                            [3, 3, 3, 3, 3, 3, 3, 3, 3]]]]).float()
    
    mask = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]],
                          
                          [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]],
                          
                          [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]],
                          
                          [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]]]).float()
    
    
    def masked_reconstruction_error(img1, img2, mask):
      # マスクを適用
      img1_masked = img1 * mask
      img2_masked = img2 * mask

      # 再構成誤差を計算（Mean Squared Error）
      mse = F.mse_loss(img1_masked, img2_masked, reduction='none')  # ピクセルごとの二乗誤差
      masked_mse = mse * mask  # マスクされていない部分のみを対象にする
      total_mse = torch.sum(masked_mse)  # マスクされていない部分の二乗誤差の合計
      num_unmasked_pixels = torch.sum(mask)  # マスクされていないピクセルの数
      
      if num_unmasked_pixels > 0:
          reconstruction_error = total_mse / num_unmasked_pixels  # マスクされていない部分の平均二乗誤差
      else:
          reconstruction_error = torch.tensor(0.0, device=total_mse.device)  # マスクされていない部分がない場合、0を返す
      
      return reconstruction_error

    """
    print(masked_reconstruction_error(input,input,mask))
    """
    #conv = MyConv(3,3,3,stride=1,padding=3,bias=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #y = conv(input)
    torch.set_printoptions(linewidth=100000)
    torch.set_printoptions(edgeitems=1000)
    #print(y,y.shape)
    print('maskシェイプ',mask.shape)
    print('inputシェイプ',input.shape)
    input = input.to(device)
    conv = MyConv(3,3,3,stride=2,padding=1,bias=False)
    #conv = MaskedConv2d(3,3,3,stride=2,padding=1,bias=False)
    y = conv(input,mask)
    print("myconvT:",y.shape)
    #print("true:",true[0],true.shape)
    convT = torch.nn.ConvTranspose2d(3,3,3,stride=2,padding=1,bias=False,output_padding=1).to(device)
    y = convT(input)
    print(y.shape)
    
    