import custom_conv2d
import torch
import torch.nn as nn
from .GDN import GDN
import torch.nn.functional as F
import torchvision

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )
def subpel_conv1x1(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class CustomConv2DPy(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device='cuda:0'):
        super(CustomConv2DPy, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.weight = self.conv2d.weight
        self.bias = self.conv2d.bias
    def forward(self, x):
        x = custom_conv2d.cal(x,self.weight,self.bias,self.out_channels,self.stride,self.padding)
        torch.cuda.empty_cache()
        return x

class CustomConv2DPyM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device='cuda:0'):
        super(CustomConv2DPyM, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.biasbool = bias
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=self.biasbool,device=device)
        self.weight = self.conv2d.weight
        self.device = self.conv2d.weight.device
        if self.biasbool == True:
            self.bias = self.conv2d.bias
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
        
    def forward(self, x,mask):
        x = custom_conv2d.calmaskV2(x,mask,self.weight,self.bias,self.out_channels,self.stride,self.padding)
        return x
    
class CustomConv2DPyMV3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device='cuda:0'):
        super(CustomConv2DPyMV3, self).__init__()
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
        if self.training:
            """
            mask_one = (mask > 0.1).float()
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
            x = self.conv2d(x)            
            """
            mask_one = F.max_pool2d(mask_one,kernel_size=3,stride=self.stride,padding=1)
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
        else:
            """
            mask_one = (mask > 0.1).float()
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            mask = masks[:,0:1,:,:]
            torchvision.utils.save_image(mask, "outputKodak/maskconv.png",alpha=True)
            x = x * masks
            """
            x = self.conv2d(x)
            
            """
            mask_one = F.max_pool2d(mask_one,kernel_size=3,stride=self.stride,padding=1)
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
        """
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
        """
            
        return x

class CustomConvTransposed2DPyMV3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, device='cuda:0'):
        super(CustomConvTransposed2DPyMV3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.device = device
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=self.biasbool, dilation=self.dilation, padding_mode='zeros', device=device)
        self.weight = self.conv2d.weight
        self.device = self.conv2d.weight.device
        if self.biasbool == True:
            self.bias = self.conv2d.bias
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation).to(self.device)
    def forward(self, x,mask,maskUp):
        if self.training:
            """
            mask_one = (mask > 0.1).float()
            mask_down = F.max_pool2d(mask_one,kernel_size=3,stride=self.stride,padding=1)
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
            x = self.conv2d(x) 
            """           
            mask_one = (maskUp > 0.1).float()
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
        else:
            """
            mask_one = (mask > 0.1).float()
            mask_down = F.max_pool2d(mask_one,kernel_size=3,stride=self.stride,padding=1)
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
            x = self.conv2d(x) 
            """           
            mask_one = (maskUp > 0.1).float()
            masks = mask_one.expand(-1,x.shape[1],-1,-1)
            x = x * masks
            """
        return x

def subpel_conv5x5(in_ch: int, out_ch: int, r: int = 1) -> nn.Module:
    """3x3 sub-pixel convolution for up-sampling."""
    return CustomConv2DPyMV3(in_ch, out_ch * r**2, kernel_size=5, padding=2)

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Module:
    """3x3 sub-pixel convolution for up-sampling."""
    return CustomConv2DPyMV3(in_ch, out_ch * r**2, kernel_size=3, padding=1)

def subpel_conv1x1(in_ch: int, out_ch: int, r: int = 1) -> nn.Module:
    """3x3 sub-pixel convolution for up-sampling."""
    return CustomConv2DPyMV3(in_ch, out_ch * r**2, kernel_size=1, padding=0)

def subpel(r: int = 1) -> nn.Module:
    """3x3 sub-pixel shuffle"""
    return nn.PixelShuffle(r)

def conv5x5M(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """5x5 convolution with padding."""
    return CustomConv2DPyMV3(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)

def conv3x3M(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return CustomConv2DPyMV3(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def conv1x1M(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return CustomConv2DPyMV3(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3UpM(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return CustomConvTransposed2DPyMV3(in_ch, out_ch, kernel_size=3, stride=stride, padding=1,output_padding=1)
def conv5x5UpM(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return CustomConvTransposed2DPyMV3(in_ch, out_ch, kernel_size=5, stride=stride, padding=2,output_padding=1)
def conv1x1UpM(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return CustomConvTransposed2DPyMV3(in_ch, out_ch, kernel_size=1, stride=stride,output_padding=1)

class ResidualBlockM(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3M(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3M(out_ch, out_ch)
        
        if in_ch != out_ch:
            self.skip = conv1x1M(in_ch, out_ch)
        else:
            self.skip = None
        
    def forward(self, x,mask):
        identity = x

        out = self.conv1(x,mask)
        out = self.leaky_relu(out)
        out = self.conv2(out,mask)
        out = self.leaky_relu(out)
        if self.skip is not None:
            identity = self.skip(x,mask)
        out = out + identity
        return out
    
class ResidualBlockWithStrideM(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3M(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3M(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1M(in_ch, out_ch, stride=stride)
        else:
            self.skip = None
        
    def forward(self, x,mask1,mask2):
        identity = x
        out = self.conv1(x,mask1)
        out = self.leaky_relu(out)
        out = self.conv2(out,mask2)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x,mask1)

        out += identity
        return out
    
class ResidualBlockUpsampleM(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = conv3x3UpM(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3M(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample_conv = conv1x1UpM(in_ch, out_ch, upsample)
    def forward(self, x,mask1,mask2):
        identity = x
        out = self.subpel_conv(x,mask1,mask2)
        out = self.leaky_relu(out)
        out = self.conv(out,mask2)
        out = self.igdn(out)
        identity = self.upsample_conv(x,mask1,mask2)
        out += identity
        return out