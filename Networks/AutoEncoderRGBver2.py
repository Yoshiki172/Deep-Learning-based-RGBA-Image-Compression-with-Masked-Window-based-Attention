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
    """マスクの前処理(0に限りになく小さい値を0にするため)"""
    #output_mask = output_mask * 255
    #output_mask = torch.round(output_mask)/255
    #入力マスクを３チャンネルに拡張
    input_mask = input_mask.expand(-1, 3, -1, -1)
    #0より大きい値を1にする
    input_mask_one = (input_mask > 0.0).float()
    #出力マスクを３チャンネルに拡張
    #output_mask = output_mask.expand(-1, 3, -1, -1)
    #0より大きい値を1にする
    #output_mask_one = (output_mask > 0.0).float()

    #torchvision.utils.save_image(output_mask_one, "outputKodak/mask.jpg")
    img1_masked = input * input_mask_one
    img2_masked = output * input_mask_one

    # 再構成誤差を計算（Mean Squared Error）
    mse = F.mse_loss(img1_masked, img2_masked, reduction='none')  # ピクセルごとの二乗誤差
    mse = torch.sum(mse,dim=(1,2,3))
    #ここ変えて！！！！！！！！！！！！！！！
    num_unmasked_pixels = torch.sum(input_mask_one,dim=(1,2,3)) # マスクされていないピクセルの数
    num_unmasked_pixels = torch.clamp(num_unmasked_pixels, min=1)

    reconstruction_error = torch.mean(torch.div(mse,num_unmasked_pixels))  # マスクされていない部分の平均二乗誤差
    
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
        #return x_hat,mse_loss,total_bpp,total_y_bpp,total_z_bpp
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    def compress(self,input,mask):
        me1,me2,me3,me4,_,_ = self.EncMakeMask(mask)
        y = self.Encoder(input,mask,me1,me2,me3,me4)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape,mask):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        md1,md2,md3,md4,_,_ = self.DecMakeMask(mask)
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.Decoder(y_hat,mask,md1,md2,md3,md4).clamp_(0, 1)

        return {"x_hat": x_hat}
