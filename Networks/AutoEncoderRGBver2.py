import torch
import math
from Networks.bitEstimator import *
from Networks.Hyper import *
from Networks.entropy import *
from Networks.SupplyMask import *
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from Networks.TransformRGBver3 import Analysis_transform,Synthesis_transform
from .Attention import *
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from .SWAtten import *

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64    
def ste_round(x):
    return torch.round(x) - x.detach() + x
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def reconstruct_error(input,output,input_mask,output_mask):
    #Input mask extended to 3 channels
    input_mask = input_mask.expand(-1, 3, -1, -1)
    #Set values greater than 0 to 1
    input_mask_one = (input_mask > 0.0).float()

    img1_masked = input * input_mask_one
    img2_masked = output * input_mask_one

    # （Mean Squared Error）
    mse = F.mse_loss(img1_masked, img2_masked, reduction='none')  # Squared error per pixel
    mse = torch.sum(mse,dim=(1,2,3))
    
    num_unmasked_pixels = torch.sum(input_mask_one,dim=(1,2,3)) #Number of unmasked pixels
    num_unmasked_pixels = torch.clamp(num_unmasked_pixels, min=1)

    reconstruction_error = torch.mean(torch.div(mse,num_unmasked_pixels))  #Mean squared error of the unmasked area
    
    return reconstruction_error
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

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
        identity  = input
        x_first = self.input_conv(input)
        x = self.enh1(x_first)
        x = self.enh2(x)
        x = self.enh3(x)
        x = x + x_first
        x = self.output_conv(x)
        x = x + identity
        return x

class AutoEncoder(CompressionModel):
    def __init__(self):
        super().__init__()
        #self.N = 192
        #self.M = 320
        self.N = 192
        self.M = 80
        self.Encoder = Analysis_transform(self.N,self.M)
        self.Decoder = Synthesis_transform(self.N,self.M)
        self.EncMakeMask = SupplyMaskToTransform()
        self.DecMakeMask = SupplyMaskToTransform()

        self.num_slices = 10
        self.max_support_slices = 5
       
        self.h_a = nn.Sequential(
            conv3x3(self.M, 320, stride=2),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            subpel_conv3x3(192, 192, 2),
            nn.GELU(),
            conv3x3(192, 224),
            nn.GELU(),
            subpel_conv3x3(224, 256, 2),
            nn.GELU(),
            conv3x3(256, 288),
            nn.GELU(),
            subpel_conv3x3(288, self.M, 2),
        )

        self.h_scale_s = nn.Sequential(
            subpel_conv3x3(192, 192, 2),
            nn.GELU(),
            conv3x3(192, 224),
            nn.GELU(),
            subpel_conv3x3(224, 256, 2),
            nn.GELU(),
            conv3x3(256, 288),
            nn.GELU(),
            subpel_conv3x3(288, self.M, 2),
        )
       
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.M + (self.M//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (self.M//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.M + (self.M//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (self.M//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.M + (self.M//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (self.M//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.DecMakeMask = SupplyMaskToTransform()
    def forward(self, input,mask,reconmask,me1,me2,me3,me4):    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #reconmask = 256*256
        #md1 = 128*128
        #md2 = 64*64
        #md3 = 32*32
        #md4 = 16*16
        
        """add rounding process"""
        reconmask = reconmask * 255
        reconmask = torch.round(reconmask)
        reconmask = reconmask / 255
        md1,md2,md3,md4,_,_ = self.DecMakeMask(reconmask)
        #y = self.Encoder(input)
        y = self.Encoder(input,reconmask,me1,me2,me3,me4)
    
        y_shape = y.shape[2:]
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            #mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            #scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            
            #print(y_slice.shape,scale.shape,mu.shape)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        sigmas = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        #x_hat = self.Decoder(y_hat)
        x_hat = self.Decoder(y_hat,reconmask,md1,md2,md3,md4)
        
        
        ############################################################
        
        ##############################calculate_bit##############################
        
        Y_bits = torch.sum(torch.clamp(-1.0 * torch.log(y_likelihoods + 1e-10) / math.log(2.0), 0, 50))
        Z_bits = torch.sum(torch.clamp(-1.0 * torch.log(z_likelihoods + 1e-10) / math.log(2.0), 0, 50))

        ############################################################
        
        #mse_loss = torch.mean((outputs - inputs).pow(2)) + torch.mean((outputsmask - inputsmask).pow(2))
        #mse_loss = (self.masked_reconstruction_error(inputs,outputs,outputsmask) + torch.mean((outputsmask - inputsmask).pow(2)) ) /2
      
        #mse_loss = torch.mean((x_hat - input).pow(2))
        mse_loss = reconstruct_error(input,x_hat,mask,reconmask)
        batch_size = input.shape[0]
        
        total_z_bpp = (Z_bits)/(batch_size*input.shape[2]*input.shape[3])
        total_y_bpp = (Y_bits)/(batch_size*input.shape[2]*input.shape[3])
        
        total_bpp = total_y_bpp + total_z_bpp
        return x_hat,mse_loss,total_bpp,total_y_bpp,total_z_bpp
