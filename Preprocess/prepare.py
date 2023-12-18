import argparse
import glob

import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F

import pickle
from PIL import Image

from torch.autograd import Function
import time
import os

class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self, train_glob, transform):
        super(PrepareDataset, self).__init__()
        self.transform = transform
        self.images = list(sorted(glob.glob(train_glob)))

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.images)

class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        img /= 127.5
        img -= 1.0
        return img.transpose((2, 0, 1))
    
def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
