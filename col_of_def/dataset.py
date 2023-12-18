import os.path as osp
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch
import random
from pycocotools.coco import COCO
import numpy as np
import os
import glob
def make_datapath_list(rootpath):
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/trainval.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

def make_datapath_list_coco(rootpath='./COCO/fast-ai-coco/annotations_trainval2017/', phase ='train'):
    coco = COCO(osp.join(rootpath, f"annotations/instances_{phase}2017.json"))
    img_ids = coco.getImgIds()
    img_list = [coco.loadImgs(img_id)[0]['file_name'] for img_id in img_ids]
    anno_list = img_ids
    
    return img_list, anno_list

def make_datapath_list_for_Kodak(rootpath):
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'PNGImages', '%s.png')
    annopath_template = osp.join(rootpath, 'MaskImages', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    val_id_names = osp.join(rootpath + 'ImageSets/mask.txt')

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return val_img_list, val_anno_list

def make_datapath_list_for_P3M(rootpath):
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'mask', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    val_id_names = osp.join(rootpath + '/train_list.txt')

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        val_img_list.append(img_path)

    return val_img_list

class COCOP3MDataset(data.Dataset):
    def __init__(self, coco_path='P3Mdata/COCOdata', p3m_path='P3Mdata/MASKpatches', height=256, width=256,fill_mix_ratio=0.25):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.random_crop = transforms.Compose([
            transforms.RandomResizedCrop((height, width))
        ])
        self.images = glob.glob(os.path.join(coco_path, '*.png')) + glob.glob(os.path.join(p3m_path, '*.png'))
        self.height = height
        self.width = width
        self.fill_mix_ratio = fill_mix_ratio
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)#.convert('RGBA')
        
        img_array = np.array(img)

        rgba = Image.fromarray(img_array, 'RGBA')
        
        rgb_array = img_array[:,:,:3]
        alpha_array = img_array[:,:,3]
        
        img = Image.fromarray(rgb_array, 'RGB')
        alpha = Image.fromarray(alpha_array, 'L')
        
        img = self.transform(img)
        alpha = self.transform(alpha)
        RGBA = torch.cat([img,alpha],dim=0)

        RGBA = self.random_crop(RGBA)

        flip_horizontal = random.random() < 0.5
        flip_vertical = random.random() < 0.5
        
        def apply_transforms(img):
            if flip_horizontal:
                img = transforms.functional.hflip(img)
            if flip_vertical:
                img = transforms.functional.vflip(img)
            
            return img
        fill_transform = transforms.RandomApply([
            FillImage()
        ], p=self.fill_mix_ratio)
        
        RGBA = apply_transforms(RGBA)
        
        img = RGBA[:3,:,:]
        alpha = RGBA[3:4,:,:]

        alpha = fill_transform(alpha)
        rgba_data = torch.cat([img,alpha],dim=0)

        masked_image = torch.where((alpha > 0), img, alpha)
        images_with_alpha = torch.cat([img, alpha], dim=0)
        return masked_image,alpha,img,alpha,images_with_alpha

    
class FillImage(object):
    def __call__(self, img):
        return torch.ones_like(img)

class KodakDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase="test"):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        #anno_class_img = anno_class_img / 255.0#追加
        masked_image = torch.where((anno_class_img > 0), img, anno_class_img)
        maskdata = anno_class_img[0:1,:,:]


        images_with_alpha = torch.cat([masked_image, maskdata], dim=0)

        return masked_image, maskdata, img, anno_class_img, images_with_alpha
       

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''
        transform = transforms.Compose([
        ])
        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]
        img = torchvision.transforms.functional.to_tensor(img)
        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
        anno_class_img = anno_class_img.convert("L").convert("RGB")
        anno_class_img = torchvision.transforms.functional.to_tensor(anno_class_img)
        img = transform(img)
        anno_class_img = transform(anno_class_img)
        return img, anno_class_img

class P3MDataset(data.Dataset):
    
    def __init__(self, img_list,height,width):
        self.img_list = img_list
        self.height = height
        self.width = width
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        
        

        resize = transforms.Resize(size=(512,512))
        crop = transforms.RandomCrop(size=(1024,1024))   
        x_flip = transforms.RandomHorizontalFlip(p=0.5)
        y_flip = transforms.RandomVerticalFlip(p=0.5)
        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]
        
        #img = resize(img)
        img = crop(img)
        img = x_flip(img)
        img = y_flip(img)
        img = img.convert("L")
        img = torchvision.transforms.functional.to_tensor(img)
        
        #img = apply_transforms(img)

        masked_image, maskdata, anno_class_img, images_with_alpha = img,img,img,img.expand(4,-1,-1)

        return masked_image, maskdata, img, anno_class_img, images_with_alpha
        

    def random_crop(self, img):
        scale_range = (1.0, 2.0)
            # リサイズとクロップを同時に行うRandomResizedCropを使用する
        resize_crop = transforms.RandomResizedCrop(
            (self.height, self.width), scale=scale_range, ratio=(1.0, 1.0))
        img = resize_crop(img)

        return img
    
    def pull_item(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        img = img.convert("L")
        # 2. アノテーション画像読み込み
        img = transform(img)

        return img

from torchvision.transforms import functional as F
from random import randint
class COCODataset(data.Dataset):
    def __init__(self, img_list, anno_list, rootpath='./COCO/fast-ai-coco/', phase='train', height=256, width=256):
        self.img_list = img_list
        self.anno_list = anno_list
        self.rootpath = rootpath
        self.phase = phase
        self.height = height
        self.width = width
        self.coco = COCO(osp.join(rootpath, f"annotations_trainval2017/annotations/instances_{phase}2017.json"))
    
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        width, height = img.shape[1],img.shape[2]
        crop_h, crop_w = self.height,self.width
        anno_class_img = anno_class_img / 255.0#追加
        
        zeros = torch.zeros(anno_class_img.shape)
        ones = torch.ones(anno_class_img.shape)
        
        maskdata = torch.where((anno_class_img > 0) & (anno_class_img < 1), ones, zeros)
        
        maskdata = maskdata[0:1, :, :]

        #追加
        flip_horizontal = random.random() < 0.5
        flip_vertical = random.random() < 0.5
        i = randint(0, height - crop_h)
        j = randint(0, width - crop_w)
        #rotation_angle = random.uniform(-15, 15)
        #scale_range = (1, 1)
        def apply_transforms(img):
            if flip_horizontal:
                img = transforms.functional.hflip(img)
            if flip_vertical:
                img = transforms.functional.vflip(img)
            
            return img
        """
        blur_transform = transforms.RandomApply([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=15),
            transforms.ToTensor()
        ], p=0.5)
        """
        blur_transform = transforms.RandomApply([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=15),
            transforms.ToTensor()
        ], p=0.5)
        fill_transform = transforms.RandomApply([
            FillImage()
        ], p=0.0)
        img = apply_transforms(img)
        maskdata = apply_transforms(maskdata)
        maskdata = blur_transform(maskdata)
        #maskdata = fill_transform(maskdata)
        masked_image = torch.where((maskdata > 0), img, maskdata)
        images_with_alpha = torch.cat([img, maskdata], dim=0)

        return masked_image, maskdata, img, anno_class_img, images_with_alpha

    def random_crop(self, img):
        # リサイズとクロップを同時に行うRandomResizedCropを使用する
        resize_crop = transforms.RandomResizedCrop((self.height, self.width))
        img = resize_crop(img)

        return img
    
    def pull_item(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 1. 画像読み込み
        image_file_path = osp.join(self.rootpath, f"{self.phase}2017/{self.phase}2017", self.img_list[index])
        img = Image.open(image_file_path).convert("RGB")

        # 2. アノテーション画像生成
        img_id = self.anno_list[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 全てのアノテーションを統合したマスク画像を作成
        mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        for ann in anns:
            mask += self.coco.annToMask(ann) * ann['category_id']

        anno_class_img = Image.fromarray(mask, mode="L")

        
        img = transform(img)
        
        anno_class_img = transform(anno_class_img)
        RGBA = torch.cat([img,anno_class_img],dim=0)
        RGBA = self.random_crop(RGBA)
        img = RGBA[0:3,:,:]
        anno_class_img = RGBA[3:4,:,:]
        return img, anno_class_img
    
if __name__ == "__main__":
    imglist,anolist = make_datapath_list_for_Kodak("../Kodak/")
    dataset = KodakDataset(imglist,anolist)
    img,ano,_,_,_ = dataset[3]
    img = torch.cat([img,ano],dim=0)
    torchvision.utils.save_image(img, "img.png")
    print(img.shape)