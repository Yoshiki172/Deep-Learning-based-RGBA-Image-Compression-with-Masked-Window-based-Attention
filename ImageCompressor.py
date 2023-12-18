#実行例:CUDA_VISIBLE_DEVICES=0 python ImageCompressor.py -c -p checkpoints/RGB/iter_276000.pth.tar -pm checkpoints/mask4096/iter_142000.pth.tar -i ./img.png
import os
import argparse
from Networks.AutoEncoderRGBver2 import AutoEncoder
from Networks.AutoEncoderMask import AutoEncoder as MaskAutoEncoder
from Networks.SupplyMask import *
from Networks.ms_ssim_torch import *
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import col_of_def.dataset as dataset 
import col_of_def.prepare as prepare
#from datasets import Datasets, TestKodakDataset
import numpy as np
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import logging
torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
torch.cuda.empty_cache()
from PIL import Image
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 4
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
image_size = 256
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000

train_root = './Train_Image'
test_root = './Val_Image'

parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-i', '--image', default='',
        help='load image dir')

parser.add_argument('-c', '--compress', action='store_true',
        help='compress image')
parser.add_argument('-dc', '--decompress', action='store_true',
        help='decompress image')

parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('-pm', '--pretrainmask', default = '', 
        help='load pretrain mask model')

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0
def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)
def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )
def image_to_tensor(image_path):
    # 画像を開く
    image = Image.open(image_path)

    # PyTorchが使用する形式に変換するための変換関数を作成
    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL画像をPyTorchのテンソルに変換
    ])

    # 画像をテンソルに変換
    tensor = transform(image)

    return tensor
import gzip
import pickle
def compress_bytes_to_file(data, filename):
    serialized_data = pickle.dumps(data)
    with gzip.open(filename, 'wb') as file:
        file.write(serialized_data)
def decompress_file_to_bytes(filename):
    with gzip.open(filename, 'rb') as file:
        serialized_data = file.read()
    data = pickle.loads(serialized_data)
    return data
def compressor(image_path):
    p = 128
    net.update()
    masknet.update()
    net.eval()
    masknet.eval()
    tensor = image_to_tensor(image_path)
    image_with_alpha = tensor.unsqueeze(0).to(device)
    image_with_alpha, padding = pad(image_with_alpha, p)
    input = image_with_alpha[:,0:3,:,:]
    if(image_with_alpha.shape[1] == 3):
        print(image_with_alpha.shape)
        mask = torch.ones(image_with_alpha.shape[0],1,image_with_alpha.shape[2],image_with_alpha.shape[3]).to(device)
    else:
        mask = image_with_alpha[:,3:4,:,:]
    
    print(input.shape)
    start = time.perf_counter()
    ma = net.compress(input,mask)
    ma['padding'] = padding
    da = masknet.compress(mask)
    filename = 'compressed_file.gz'
    compress_bytes_to_file(ma, filename)
    filename = 'compressed_maskfile.gz'
    compress_bytes_to_file(da, filename)
    ma1 = decompress_file_to_bytes('compressed_file.gz')
    end = time.perf_counter()
    print(end - start)
    print(ma == ma1)
    torchvision.utils.save_image(tensor, "output_image/image.png",alpha=True)
    

def decompressor():
    net.update()
    masknet.update()
    net.eval()
    masknet.eval()
    maindata = "compressed_file.gz"
    maindata =  decompress_file_to_bytes(maindata)
    maskdata = "compressed_maskfile.gz"
    maskdata =  decompress_file_to_bytes(maskdata)
    mask = masknet.decompress(maskdata["strings"], maskdata["shape"])
    img = net.decompress(maindata["strings"], maindata["shape"],mask["x_hat"])
    img = torch.cat([img["x_hat"],mask["x_hat"]],dim=1)
    img = crop(img, maindata["padding"])
    torchvision.utils.save_image(img, "DecodedImage.png",alpha=True)


if __name__ == "__main__":
    args = parser.parse_args()
    
    model = AutoEncoder()
    maskmodel = MaskAutoEncoder()
    EncMakeMask = SupplyMaskToTransform()
    
    if args.pretrain != '':
        _ = load_model(model, args.pretrain)
    
    if args.pretrain != '':
        _ = load_model(maskmodel, args.pretrainmask)
    
    net = model.to(device)
    masknet = maskmodel.to(device)
    
    if args.compress:
        imagepath = args.image
        compressor(imagepath)
    if args.decompress:
        decompressor()
    
   
