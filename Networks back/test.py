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
import custom_conv2d
torch.set_printoptions(linewidth=100000)
torch.set_printoptions(edgeitems=1000)


class CustomConvT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        super(CustomConvT, self).__init__()
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
        self.device = torch.device('cuda')
        self.dtype = dtype
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
          assert self.stride > self.output_padding
        except AssertionError:
          import sys
          sys.exit("Error: Output padding must be smaller than either stride")
        self.normalConv = torch.nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups, self.biasbool, self.dilation, self.padding_mode, self.device, self.dtype).to(self.device)
        self.weighted = torch.nn.Parameter(self.normalConv.weight.data, requires_grad=True)
        print("weight_shape:",self.weighted.shape)
        self.weight = torch.nn.Parameter(torch.rot90(self.normalConv.weight.data.transpose(0,1),2,[2,3]), requires_grad=True)
       
        if self.biasbool == True:
            self.bias = torch.nn.Parameter(self.normalConv.bias.data, requires_grad=True)
        else:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_channels).to(self.device), requires_grad=True)
       
        #self.weight = torch.rot90(self.weight,2,[2,3])
        self.weight_view = self.weight.reshape((self.out_channels,-1))
        self.bias_view = self.bias.view(1, self.bias.shape[0], 1).to(self.device)
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=0, dilation=self.dilation).to(self.device)
        self.for_channels = in_channels **3
   
    def insert_zeros(self,input):
        out = torch.zeros(input.shape[0],input.shape[1],input.shape[2]+input.shape[2]*(self.stride-1),input.shape[3]+input.shape[3]*(self.stride-1)).to(self.device)
        out[:,:,::self.stride,::self.stride] = input.clone()
        out = out[:,:,0:out.shape[2]-self.stride+1+self.output_padding,0:out.shape[3]-self.stride+1+self.output_padding]
        return out
 
    def insert_zeros2(self,input):
            H_out = input.shape[2] + (input.shape[2] - 1) * (self.stride - 1) + self.output_padding
            W_out = input.shape[3] + (input.shape[3] - 1) * (self.stride - 1) + self.output_padding


            out = torch.zeros(input.shape[0], input.shape[1], H_out, W_out, device=self.device)
            out[:, :, ::self.stride, ::self.stride] = input


            return out
   
    """
    def insert_zeros2(input,stride,outputpadding):
            H_out = input.shape[2] + (input.shape[2] - 1) * (stride - 1) + output_padding
            W_out = input.shape[3] + (input.shape[3] - 1) * (stride - 1) + output_padding


            out = torch.zeros(input.shape[0], input.shape[1], H_out, W_out)
            out[:, :, ::stride, ::stride] = input
            return out
    """
   
    def forward(self, x, mask):
       
        batch, in_channels, in_height, in_width = x.size()
        _, _, kernel_height, kernel_width = self.weight.size()
        x,mask = x.to(self.device), mask.to(self.device)
        output = F.conv_transpose2d(x,self.weighted,bias=self.bias,stride=self.stride,padding=self.padding,output_padding=self.output_padding,groups=1,dilation=1)
        if self.stride > 1:
          #x = self.insert_zeros2(x)
          x = custom_conv2d.insert_zeros(x,self.stride,self.output_padding,self.device)
       
        x = torch.cat([x,mask],dim=1)
        x = F.pad(x, (self.kernel_size-1,self.kernel_size-1,self.kernel_size-1,self.kernel_size-1), mode='constant', value=0)
        x = x[:,:,self.padding:x.shape[2]-self.padding,self.padding:x.shape[3]-self.padding]
       
        x = self.UnFold(x)
       
        #x,mask = custom_conv2d.preprocessing(x,mask,self.in_channels)
        #x,mask = self.preprocessing(x,mask,self.in_channels)
        mask = x[:,kernel_height * kernel_width*self.in_channels:kernel_height * kernel_width*(self.in_channels+1),:].transpose_(1,2)
        x = x[:,0:kernel_height * kernel_width*self.in_channels,:].transpose_(1,2)
       
        mask = torch.any(mask != 0, dim=2)
        x = x[mask].unsqueeze(dim=0)
        x = x.permute(0,2,1)
        summask = torch.sum(mask, dim=1)


        x = torch.matmul(self.weight_view, x) + self.bias_view
        x = custom_conv2d.calmaskV3convt(input,x,mask,summask,self.weight,self.bias,self.out_channels,self.stride,self.padding,self.output_padding,self.device)
        """
        #x = custom_conv2d.calmaskV3(input,x,mask,summask,self.weight,self.bias,self.out_channels,self.stride,self.padding)
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, self.out_channels, mask.shape[1])
        zeros_tensor = torch.zeros(batch, self.out_channels, mask.shape[2]).to(self.device)
       
        start = 0
        for i in range(batch):
            end = start + summask[i]
            zeros_tensor[i].masked_scatter_(mask[i, 0:self.out_channels, 0:mask.shape[2]], x[0:1, 0:self.out_channels, start:end])
            start = end
        x = zeros_tensor    
       
        print(x.shape)
        #x = x.reshape((4,self.out_channels,int(math.sqrt(x.shape[2])),int(math.sqrt(x.shape[2]))))
       
        output_height = (in_height - 1) * self.stride + kernel_height - 2 * self.padding + self.output_padding
        output_width = (in_width - 1) * self.stride + kernel_width - 2 * self.padding + self.output_padding


        x = x.reshape((batch,self.out_channels,output_height,output_width))
        """
        return x,output


if __name__ == "__main__":
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
   
    mask = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]],
                           
                           [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]],
                           
                           [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]],
                           
                           [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]]]).float()
   
    mask = torch.ones((4, 1, 18, 18))
    print(mask.shape)
    CONVT = CustomConvT(3,4,5,stride=2,padding=2,output_padding=1,bias=False)
    out,output = CONVT(input,mask)
    out[out <= 0.00001] = 0
    #print(out,out.shape)
    #output[output <= 0.00001] = 0
    #print(output,output.shape)


