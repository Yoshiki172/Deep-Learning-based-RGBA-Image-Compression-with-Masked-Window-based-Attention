import torch
import math
import torch.nn as nn
import torch.nn.functional as f
from .Custom_Function import *
from .win_attention import *
from compressai.layers import (
    conv3x3,
)
from torchvision import transforms
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class EnhancementBlock(nn.Module):
    def __init__(self, num_filters=32):
        super(EnhancementBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1,padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + input
        return x

class DSE(nn.Module):
    def __init__(self, in_ch = 1,num_filters=32):
        super(DSE, self).__init__()
        self.input_conv = nn.Conv2d(in_ch, num_filters, 1, stride=1)
        self.enh1 = EnhancementBlock(num_filters)
        self.enh2 = EnhancementBlock(num_filters)
        self.enh3 = EnhancementBlock(num_filters)
        self.output_conv = nn.Conv2d(num_filters, in_ch, 1, stride=1)

    def forward(self, input):
        identity  = input
        x_first = self.input_conv(input)
        x = self.enh1(x_first)
        x = self.enh2(x)
        x = self.enh3(x)
        x = x + x_first
        x = self.output_conv(x)
        x = x + identity
        return x

class ResBlock(nn.Module):
    def __init__(self, num_filters=128):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters//2, 1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters//2, num_filters//2, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_filters//2, num_filters, 1, stride=1)

    def forward(self, x):
        res = self.relu1(self.conv1(x))
        res = self.relu2(self.conv2(res))
        res = self.conv3(res)
        res += x
        return res

class ResBlockMask(nn.Module):
    def __init__(self, num_filters=128):
        super(ResBlockMask, self).__init__()
        self.conv1 = CustomConv2DPyMV3(num_filters, num_filters//2, 1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = CustomConv2DPyMV3(num_filters//2, num_filters//2, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = CustomConv2DPyMV3(num_filters//2, num_filters, 1, stride=1)

    def forward(self, x ,mask):
        res = self.relu1(self.conv1(x,mask))
        res = self.relu2(self.conv2(res,mask))
        res = self.conv3(res,mask)
        res += x
        return res

class AttentionMask(nn.Module):
    def __init__(self, num_filters=128):
        super(AttentionMask, self).__init__()
        #self.conv1 = nn.Conv2d(num_filters, num_filters, 1, stride=1)
        self.conv1 = CustomConv2DPyMV3(num_filters, num_filters, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.trunk_ResBlock1 = ResBlock(num_filters)
        self.trunk_ResBlock2 = ResBlock(num_filters)
        self.trunk_ResBlock3 = ResBlock(num_filters)
        self.attention_ResBlock1 = ResBlock(num_filters)
        self.attention_ResBlock2 = ResBlock(num_filters)
        self.attention_ResBlock3 = ResBlock(num_filters)
        self.NLN = Non_local_Block(num_filters,num_filters)
    def forward(self, x,mask):
        trunk_branch = self.trunk_ResBlock1(x)
        trunk_branch = self.trunk_ResBlock2(trunk_branch)
        trunk_branch = self.trunk_ResBlock3(trunk_branch)
        
        attention_branch = self.NLN(x)
        attention_branch = self.attention_ResBlock1(attention_branch)
        attention_branch = self.attention_ResBlock2(attention_branch)
        attention_branch = self.attention_ResBlock3(attention_branch)
        attention_branch = self.conv1(attention_branch,mask)
        attention_branch = self.sigmoid(attention_branch)

        # print("x.shape: ", x.shape)
        # print("attention.shape: ", attention_branch.shape)
        # print("trunk_branch.shape: ", trunk_branch.shape)
        result = x + torch.mul(attention_branch, trunk_branch)
        return result


class Attention(nn.Module):
    def __init__(self, num_filters=128):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.trunk_ResBlock1 = ResBlock(num_filters)
        self.trunk_ResBlock2 = ResBlock(num_filters)
        self.trunk_ResBlock3 = ResBlock(num_filters)
        self.attention_ResBlock1 = ResBlock(num_filters)
        self.attention_ResBlock2 = ResBlock(num_filters)
        self.attention_ResBlock3 = ResBlock(num_filters)
        self.NLN = Non_local_Block(num_filters,num_filters)
        #self.EDB = EdgeDetection_Block(num_filters)

    def forward(self, x):
        trunk_branch = self.trunk_ResBlock1(x)
        trunk_branch = self.trunk_ResBlock2(trunk_branch)
        trunk_branch = self.trunk_ResBlock3(trunk_branch)
        
        attention_branch = self.NLN(x)
        #attention_branch = self.EDB(x)
        attention_branch = self.attention_ResBlock1(attention_branch)
        attention_branch = self.attention_ResBlock2(attention_branch)
        attention_branch = self.attention_ResBlock3(attention_branch)
        attention_branch = self.conv1(attention_branch)
        attention_branch = self.sigmoid(attention_branch)
        
        # print("x.shape: ", x.shape)
        # print("attention.shape: ", attention_branch.shape)
        # print("trunk_branch.shape: ", trunk_branch.shape)
        result = x + torch.mul(attention_branch, trunk_branch)
        
        return result

class EdgeDetection_Block(nn.Module):
    def __init__(self, num_channel):
        super(EdgeDetection_Block, self).__init__()
        self.sobel_horizontal = torch.tensor([[[[-1, 0, 1],
                                                [-1, 0, 1],
                                                [-1, 0, 1]]]], dtype=torch.float32, requires_grad=False)
        
        self.sobel_vertical = torch.tensor([[[[-1, -1, -1],
                                              [0,  0,  0],
                                              [1,  1,  1]]]], dtype=torch.float32, requires_grad=False)

        self.average = torch.tensor([[[[1, 1, 1],
                                        [1,  1,  1],
                                        [1,  1,  1]]]], dtype=torch.float32, requires_grad=False)/9
        
        self.pwConvF = nn.Conv2d(num_channel, 1, 1, 1, 0,bias=False)
        self.pwConvE = nn.Conv2d(1, num_channel, 1, 1, 0,bias=False)
        
    def forward(self, x):
        #self.sobel_horizontal = self.sobel_horizontal.expand(x.size(1),x.size(1),-1,-1).to(x.device)
        #self.sobel_vertical = self.sobel_vertical.expand(x.size(1),x.size(1),-1,-1).to(x.device)
        #self.average = self.average.expand(x.size(1),x.size(1),-1,-1).to(x.device)
        self.sobel_horizontal = self.sobel_horizontal.to(x.device)
        self.sobel_vertical = self.sobel_vertical.to(x.device)
        self.average = self.average.to(x.device)
        z = self.pwConvF(x)
        z = F.conv2d(z, self.average, padding=1)
        sobel_x = F.conv2d(z, self.sobel_horizontal, padding=1)
        sobel_y = F.conv2d(z, self.sobel_vertical, padding=1)
        z = torch.sqrt(torch.clamp(sobel_x**2,min=1e-10,max=1.0)+torch.clamp(sobel_y**2,min=1e-10,max=1.0))
        
        z = self.pwConvE(z)
        z = torch.mul(x,z)
        if self.training:
            pass
        else:
            pic = z[:,0:3,:,:]
            torchvision.utils.save_image(pic, "atten_mask.png")
        z += x
        return z


class Non_local_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Non_local_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.out_channel, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x, phi_x)
        f_div_C = f.softmax(f1, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        W_y = self.W(y)
        if self.training:
            pass
        else:
            pic = W_y[:,0:3,:,:]
            torchvision.utils.save_image(pic, "atten_mask.png")
        z = W_y+x

        return z
    
    
class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
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

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

