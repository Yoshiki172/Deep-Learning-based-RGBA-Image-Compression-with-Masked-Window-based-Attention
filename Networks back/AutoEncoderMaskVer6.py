import torch
import math
from Networks.bitEstimator import *
from Networks.MaskTransform import *
from Networks.Hyper import *
from Networks.entropy import *
from Networks.SupplyMask import *
from .AttentionMask import *
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)



class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample = 2):
        super().__init__()
        self.subpel_conv = nn.ConvTranspose2d(in_ch,out_ch,3,stride=upsample,padding=1,output_padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = nn.ConvTranspose2d(in_ch,out_ch,1,stride=upsample,padding=0,output_padding=1)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


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

class EdgeAttention(nn.Module):
    def __init__(self, num_filters=128):
        super(EdgeAttention, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.trunk_ResBlock1 = ResBlock(num_filters)
        self.trunk_ResBlock2 = ResBlock(num_filters)
        self.trunk_ResBlock3 = ResBlock(num_filters)
        self.attention_ResBlock1 = ResBlock(num_filters)
        self.attention_ResBlock2 = ResBlock(num_filters)
        self.attention_ResBlock3 = ResBlock(num_filters)
    
    def forward(self, x):
        trunk_branch = self.trunk_ResBlock1(x)
        trunk_branch = self.trunk_ResBlock2(trunk_branch)
        trunk_branch = self.trunk_ResBlock3(trunk_branch)
        
        #attention_branch = self.attention_ResBlock1((1-x)*x)
        attention_branch = self.attention_ResBlock1(x)
        attention_branch = self.attention_ResBlock2(attention_branch)
        attention_branch = self.attention_ResBlock3(attention_branch)
        attention_branch = self.conv1(attention_branch)
        attention_branch = self.sigmoid(attention_branch)

        # print("x.shape: ", x.shape)
        # print("attention.shape: ", attention_branch.shape)
        # print("trunk_branch.shape: ", trunk_branch.shape)
        result = x + torch.mul(attention_branch, trunk_branch)
        return result


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64    
def ste_round(x):
    return torch.round(x) - x.detach() + x
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
from compressai.models import CompressionModel
class AutoEncoder(CompressionModel):
    def __init__(self):
        super().__init__()
        #self.maskN = 32
        self.maskN = 192
        self.maskM = 80
        #self.entropy = Entropy(self.maskM)
        #self.bitEstimatorMask = BitEstimator(channel=self.maskN)
        
        self.EncoderMask = nn.Sequential(
            nn.Conv2d(1, self.maskN, 5, stride=2, padding=2),
            GDN(self.maskN),
            nn.Conv2d(self.maskN, self.maskN, 5, stride=2, padding=2),
            GDN(self.maskN),
            EdgeAttention(self.maskN),
            nn.Conv2d(self.maskN, self.maskN, 5, stride=2, padding=2),
            GDN(self.maskN),
            nn.Conv2d(self.maskN, self.maskM, 1, stride=1,padding=0),
            EdgeAttention(self.maskM),
        )

        self.DecoderMask = nn.Sequential(
            EdgeAttention(self.maskM),
            nn.ConvTranspose2d(self.maskM,self.maskN,1,stride=1,padding=0,output_padding=0),
            GDN(self.maskN,inverse=True),
            nn.ConvTranspose2d(self.maskN,self.maskN,5,stride=2,padding=2,output_padding=1),
            GDN(self.maskN,inverse=True),
            EdgeAttention(self.maskN),
            nn.ConvTranspose2d(self.maskN,self.maskN,5,stride=2,padding=2,output_padding=1),
            GDN(self.maskN,inverse=True),
            nn.ConvTranspose2d(self.maskN,1,5,stride=2,padding=2,output_padding=1),
            #DSE(in_ch=1,num_filters=self.maskN)
        )
        
        self.num_slices = 5
        self.max_support_slices = 5


        self.h_a = nn.Sequential(
            conv3x3(self.maskM, 320,stride=2),
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
            subpel_conv3x3(288, self.maskM, 2),
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
            subpel_conv3x3(288, self.maskM, 2),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.maskM + (self.maskM//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (self.maskM//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.maskM + (self.maskM//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (self.maskM//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.maskM + (self.maskM//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (self.maskM//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
    
    def forward(self, mask):    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        y = self.EncoderMask(mask)
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

            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
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
        x_hat = self.DecoderMask(y_hat)
        
        ############################################################
        
        ##############################calculate_bit##############################
        
        Y_bits = torch.sum(torch.clamp(-1.0 * torch.log(y_likelihoods + 1e-10) / math.log(2.0), 0, 50))
        Z_bits = torch.sum(torch.clamp(-1.0 * torch.log(z_likelihoods + 1e-10) / math.log(2.0), 0, 50))

        ############################################################
        
        #mse_loss = torch.mean((outputs - inputs).pow(2)) + torch.mean((outputsmask - inputsmask).pow(2))
        #mse_loss = (self.masked_reconstruction_error(inputs,outputs,outputsmask) + torch.mean((outputsmask - inputsmask).pow(2)) ) /2
      
        mse_loss = torch.mean((x_hat - mask).pow(2))
        batch_size = mask.shape[0]
        total_z_bpp = (Z_bits)/(batch_size*mask.shape[2]*mask.shape[3])
        total_y_bpp = (Y_bits)/(batch_size*mask.shape[2]*mask.shape[3])
        
        total_bpp = total_y_bpp + total_z_bpp
        
        return x_hat,mse_loss,total_bpp,total_y_bpp,total_z_bpp
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def compress(self,x):
        y = self.EncoderMask(x)
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
    
    def decompress(self, strings, shape):
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

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.DecoderMask(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}