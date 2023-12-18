#Execution example:CUDA_VISIBLE_DEVICES=0 python trainmaskP3M.py --config examples/example/config.json -n mask

import os
import argparse
from Networks.AutoEncoderMaskVer6 import AutoEncoder
from Networks.ms_ssim_torch import *
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
#import col_of_def.dataset as dataset 
#import col_of_def.prepare as prepare
import col_of_def.MYdataset as dataset 
import col_of_def.MYprepare as prepare
#from datasets import Datasets, TestKodakDataset
import numpy as np
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import logging

torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
torch.cuda.empty_cache()
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
decay_interval  = 220000
decay_interval2 = 500000
lr_decay = 0.1
lr_decay2 = 0.01
image_size = 256
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000

train_root = './Train_Image'
test_root = './Val_Image'

parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-n', '--name', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config', required=False,
        help = 'hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def save_model_train(model, iter, name):
    previous_iter = iter - 2000
    
    check_directory = os.path.join(name,f"iter_{previous_iter}.pth.tar")
    
    if iter > 595000:
        torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))
    else:
        if os.path.isfile(check_directory):
            os.remove(check_directory)
    
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

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

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, batch_size, \
        print_freq, save_model_freq, cal_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
    if 'batch_size' in config:
        batch_size = config['batch_size']
    if "print_freq" in config:
        print_freq = config['print_freq']
    if "save_model_freq" in config:
        save_model_freq = config['save_model_freq']
    if "cal_step" in config:
        cal_step = config['cal_step']
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']



def adjust_learning_rate(optimizer, global_step, decay_lr):
    global cur_lr
    global warmup_step

    # lr = base_lr * (lr_decay ** (global_step // decay_interval))
    lr = base_lr * decay_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    cur_lr = lr

def constraint(tensor):
    # Create a 3x3 filter and check surrounding values
    kernel = torch.tensor([[[[1., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.]]]], dtype=torch.float32).to(device)
    
    #Calculate the sum around each pixel using Convolution 
    neighbors_sum = F.conv2d(tensor, kernel, padding=1)
    
    # Check adjacency of 0 and 1 pixels
    isolated_zeros = (neighbors_sum == 8)
    isolated_255s = (neighbors_sum == 0)

    #Fix 0 or 1 pixel alone
    tensor[isolated_zeros] = 1
    tensor[isolated_ones] = 0

    return tensor

scaler = torch.cuda.amp.GradScaler() 
def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses = [AverageMeter(print_freq) for _ in range(7)]
    # model_time = 0
    # compute_time = 0
    # log_time = 0
    #for batch_idx, (input,_) in enumerate(train_loader,1):
    #for batch_idx, (mask,_) in enumerate(train_loader,1):
    for batch_idx, (masked_input,mask,image,_,image_with_alpha) in enumerate(train_loader,1):
        start_time = time.time()
        global_step += 1
        mask = mask[:,0:1,:,:]
        
        mask = mask.to(device)
        clipped_recon_mask, mse_loss, bpp,bpp_feature, bpp_z= net(mask)
        # print("debug", clipped_recon_image.shape, " ", mse_loss.shape, " ", bpp.shape)
        # print("debug", mse_loss, " ", bpp_feature, " ", bpp_z, " ", bpp)
        
        mse_loss = torch.mean((clipped_recon_mask - mask).pow(2))

        distribution_loss = bpp
        distortion = mse_loss
        #with torch.cuda.amp.autocast(): 
            
        rd_loss = train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        #scaler.scale(rd_loss).backward()
        rd_loss.backward()
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 5)
        
        #scaler.step(optimizer) 
        optimizer.step()
        # model_time += (time.time()-start_time)
        if (global_step % cal_step) == 0:
            # t0 = time.time()
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)

            # t1 = time.time()
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            mse_losses.update(mse_loss.item())
            # t2 = time.time()
            # compute_time += (t2 - t0)
        if (global_step % print_freq) == 0:
            # begin = time.time()
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', losses.avg, global_step)
            tb_logger.add_scalar('psnr', psnrs.avg, global_step)
            tb_logger.add_scalar('bpp', bpps.avg, global_step)
            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {cur_lr}',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
            ]))
            logger.info(log)
            
            # log_time = time.time() - begin
            # print("Log time", log_time)
            # print("Compute time", compute_time)
            # print("Model time", model_time)
        if global_step > decay_interval:
            adjust_learning_rate(optimizer, global_step, lr_decay)
      
        if (global_step % 2000) == 0:
            save_model_train(model, global_step, save_path)
            
            torchvision.utils.save_image(clipped_recon_mask, "output_image/"+str(global_step)+"mask.jpg")
          
        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, save_path)
            testKodak(global_step)
            net.train()
            torch.cuda.empty_cache()
        #scaler.update()
        
    return global_step


def testKodak(step):
    with torch.no_grad():
        
        test_loader ,test_dataset= prepare.prepare_dataset_Kodak(batch_size=1, rootpath="./Kodak/")
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        #for batch_idx, (input) in enumerate(test_loader,1):
        for batch_idx, (_,_,_,_,image_with_alpha)in enumerate(test_loader,1):
            mask = image_with_alpha[:, 3:4, :, :]
            mask = mask.to(device)
            clipped_recon_mask, mse_loss, bpp,bpp_feature, bpp_z= net(mask)
            """add rounding process"""
            
            clipped_recon_mask = clipped_recon_mask * 255
            clipped_recon_mask = torch.round(clipped_recon_mask)
            clipped_recon_mask = clipped_recon_mask / 255
            clipped_recon_mask = constraint(clipped_recon_mask)
            
            mse_loss = torch.mean((clipped_recon_mask - mask).pow(2))
            torchvision.utils.save_image(clipped_recon_mask, "outputKodak/"+str(cnt)+"mask.png",alpha=True)
            
            
            mse_loss, bpp = \
                torch.mean(mse_loss), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(mask.cpu().detach(), clipped_recon_mask.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, msssim, msssimDB))
            cnt += 1

        logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        if tb_logger !=None:
            logger.info("Add tensorboard---Step:{}".format(step))
            tb_logger.add_scalar("BPP_Test", sumBpp, step)
            tb_logger.add_scalar("PSNR_Test", sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test", sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test", sumMsssimDB, step)
        else:
            logger.info("No need to add tensorboard")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    dd = 1
    save_path = os.path.join('checkpoints', args.name)
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    model = AutoEncoder()
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    net = model.to(device)
    #net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()
    if args.test:
        testKodak(global_step)
        exit(-1)
    optimizer = optim.Adam(parameters, lr=base_lr)
    # save_model(model, 0)
    global train_loader
    tb_logger = SummaryWriter(os.path.join(save_path, 'events'))
    
    train_loader,train_dataset = prepare.prepare_dataset_train_COCOP3M(batch_size=4,height=256,width=256,fill_mix_ratio=0.0)
    #train_loader,train_dataset = prepare.prepare_dataset_train_P3M(batch_size=batch_size, rootpath="./P3M", height=256, width=256)
    steps_epoch = global_step // (len(train_dataset) // (batch_size))# * gpu_num))
    torch.save(net.state_dict(),f'checkpoint.pt')
    for epoch in range(steps_epoch, tot_epoch):
        if global_step >= decay_interval:
            adjust_learning_rate(optimizer, global_step, lr_decay)
        if global_step >= decay_interval2:
            adjust_learning_rate(optimizer, global_step, lr_decay2)
        if global_step > tot_step:
            save_model(model, global_step, save_path)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, save_path)
