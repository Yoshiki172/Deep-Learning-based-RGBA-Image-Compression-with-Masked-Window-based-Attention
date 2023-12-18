import os.path
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import sys
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torchvision.transforms as transforms
import torchvision
import torch
"""
画像が30%以上の完全不透明（255）ピクセルと30%以上の完全透明（0）ピクセルを持つか、
または1から254の値を持つ半透明ピクセルが全体の80%以上を占める場合にのみ画像を保存
"""

def process_image(im_name, ori_data_dir, ori_RGBdata_dir, patch_size, output_path, sample_num_for_each_seq):
    mask_seq_path = os.path.join(ori_data_dir, im_name)
    rgb_seq_path = os.path.join(ori_RGBdata_dir, im_name.replace('.png', '.jpg'))

    # マスク画像とRGB画像を読み込む
    mask_img = Image.open(mask_seq_path)#.convert('L')
    rgb_img = Image.open(rgb_seq_path)#.convert('RGB')

    mask_array = np.array(mask_img)
    rgb_array = np.array(rgb_img)

    mask_array = mask_array[:,:,0]
    rgb_array = rgb_array[:,:,:3]

    rgba_array = np.dstack((rgb_array, mask_array))
    rgba_img = Image.fromarray(rgba_array, 'RGBA')

    pixel_ratio = 0.3

    # RGB画像にアルファチャンネルを追加
    #rgba_img.putalpha(mask_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(patch_size),
        
    ])

    for sample_id in range(sample_num_for_each_seq):
        #print(f"Processing: {im_name}")  # 処理中の画像名を表示
        
        while True:
            # ランダムなリサイズ係数
            
            random_resize_factor = np.random.uniform(0.6, 1.0)
            crop_size = [int(patch_size[0] / random_resize_factor), int(patch_size[1] / random_resize_factor)]

            # ランダムなクロップ位置
            random_crop_x = np.random.randint(0, rgba_img.width - crop_size[1])
            random_crop_y = np.random.randint(0, rgba_img.height - crop_size[0])
            random_box = (random_crop_x, random_crop_y, random_crop_x + crop_size[1], random_crop_y + crop_size[0])
            
            # クロップとリサイズ
            cropped_img = rgba_img.crop(random_box)
            tensor_img = transform(cropped_img)
            
            alpha_channel = tensor_img[3, :, :]  # アルファチャンネルを取得

            # 255と0のピクセル数をカウント
            opaque_count = torch.sum(alpha_channel == 1)
            transparent_count = torch.sum(alpha_channel == 0)
            semi_transparent_count = torch.sum((alpha_channel > 0) & (alpha_channel < 1))
            total_count = alpha_channel.numel()

            # 両方の条件を確認
            if opaque_count / total_count >= pixel_ratio and transparent_count / total_count >= pixel_ratio or (semi_transparent_count / total_count >= 0.8):
                # テンソルをPIL画像に変換して保存
                #save_img = transforms.ToPILImage()(tensor_img)
                save_path = os.path.join(output_path, f"{im_name[:-4]}_{sample_id:04d}.png")
                #save_img.save(save_path)
                torchvision.utils.save_image(tensor_img, save_path,alpha=True)
                break
            
            
            break



# メイン処理
if __name__ == '__main__':
    ori_data_dir = 'P3M/mask/'
    ori_RGBdata_dir = 'P3M/blurred_image'
    output_path = 'P3Mdata/MASKpatches/'
    imgNames = os.listdir(ori_data_dir)
    patch_size = [256, 256]

    sample_num_for_each_seq = 30
    process_func = partial(process_image, ori_data_dir=ori_data_dir, ori_RGBdata_dir=ori_RGBdata_dir, patch_size=patch_size, output_path=output_path, sample_num_for_each_seq=sample_num_for_each_seq)

    with Pool() as p:
        res = list(tqdm(p.imap(process_func, imgNames), total=len(imgNames)))
