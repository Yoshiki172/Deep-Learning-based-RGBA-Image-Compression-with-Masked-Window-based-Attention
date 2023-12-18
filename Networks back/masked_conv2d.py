import custom_conv2d
import torch
import torch.nn as nn
from .GDN import GDN
import torch.nn.functional as F
import torchvision


class MaskedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device='cuda:0'):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.device = device
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=self.biasbool,device=device)
        self.weight = self.conv2d.weight
        self.device = self.conv2d.weight.device
        if self.biasbool == True:
            self.bias = self.conv2d.bias
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation).to(self.device)
    def forward(self, x,mask):
        if torch.all(mask == 1):
            x = self.conv2d(x)           
        else:
            input = x
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
            x = custom_conv2d.calmaskV3(input,x,mask,summask,self.weight,self.bias,self.out_channels,self.stride,self.padding,self.device)

        return x