import torch
import torch.nn as nn
import math
import custom_conv2d
from .Custom_Function import *

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class Entropy(nn.Module):
    def __init__(self,input_filters):
        super(Entropy, self).__init__()
        
        self.maskedconv = MaskedConv2d('A', input_filters, input_filters*2, 5, stride=1, padding=2)
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        #self.conv1 = nn.Conv2d(input_filters*4,640, 1, stride=1)
        self.conv1 = nn.Conv2d(input_filters*3,640, 1, stride=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(640, input_filters*9, 1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sigma,y):
        y = self.maskedconv(y)
        x = torch.cat([y, sigma], dim=1)
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))   
        x = self.conv3(x)
        # print("split_size: ", x.shape[1])
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(x, split_size_or_sections=int(x.shape[1]/9), dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        # print("probs shape: ", probs.shape)
        probs = self.softmax(probs)
        # probs = torch.nn.Softmax(dim=-1)(probs)
        means = torch.stack([mean0, mean1, mean2], dim=-1)
        variances = torch.stack([scale0, scale1, scale2], dim=-1)

        return means, variances, probs

class CustomConv2DPy_maskedConv(torch.nn.Module):
    def __init__(self,mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device='cuda:0'):
        super(CustomConv2DPy_maskedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.device = device
        self.conv2d = MaskedConv2d(mask_type, self.in_channels, self.out_channels, self.kernel_size, self.stride, padding=self.padding).to(self.device)
        self.weight = self.conv2d.weight
        self.device = self.conv2d.weight.device
        if self.biasbool == True:
            self.bias = self.conv2d.bias
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation).to(self.device)
    def forward(self, x,mask):
        input = x
        batch, in_channels, in_height, in_width = x.size()
        _, _, kernel_height, kernel_width = self.weight.size()
        x = x.to(self.device)
        #mask = torch.ones([batch,1,in_height,in_width])
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
        x = custom_conv2d.calmaskV3(input,x,mask,summask,self.weight,self.bias,self.out_channels,self.stride,self.padding)
        return x

class Entropy2(nn.Module):
    def __init__(self,input_filters):
        super(Entropy2, self).__init__()
        
        #self.maskedconv = MaskedConv2d('A', input_filters, input_filters*2, 5, stride=1, padding=2)
        self.maskedconv = CustomConv2DPy_maskedConv('A',input_filters,input_filters*2,kernel_size=5,stride=1,padding=2)
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        
        #self.conv1 = nn.Conv2d(input_filters*3,640, 1, stride=1)
        self.conv1 = CustomConv2DPyMV3(input_filters*3,640,1,stride=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        #self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.conv2 = CustomConv2DPyMV3(640,640,1,stride=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        #self.conv3 = nn.Conv2d(640, input_filters*9, 1, stride=1)
        self.conv3 = CustomConv2DPyMV3(640,input_filters*9,1,stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sigma,y,mask):
        y = self.maskedconv(y,mask)
        x = torch.cat([y, sigma], dim=1)
        
        x = self.relu1(self.conv1(x,mask))
        x = self.relu2(self.conv2(x,mask))   
        x = self.conv3(x,mask)
        # print("split_size: ", x.shape[1])
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(x, split_size_or_sections=int(x.shape[1]/9), dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        # print("probs shape: ", probs.shape)
        probs = self.softmax(probs)
        # probs = torch.nn.Softmax(dim=-1)(probs)
        means = torch.stack([mean0, mean1, mean2], dim=-1)
        variances = torch.stack([scale0, scale1, scale2], dim=-1)

        return means, variances, probs
if __name__ == "__main__":
    z = torch.rand([8,60,32,32]).cuda()
    mask = torch.rand([8,60,32,32]).cuda()
    phi = torch.rand([8,60,32,32]).cuda()
    entropy = Entropy2(60).cuda()
    means, variances, probs = entropy(phi,z,mask)
    print("means: ", means.shape)
    print("variances: ", variances.shape)
    print("probs: ", probs.shape)