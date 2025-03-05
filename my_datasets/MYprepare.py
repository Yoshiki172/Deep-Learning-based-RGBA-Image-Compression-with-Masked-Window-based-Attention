import my_datasets.MYdataset as dataset
import torch.utils.data as data
import torch
import torchvision
import os

def prepare_dataset_train_COCOP3M(batch_size=1,COCOrootpath = '../P3Mdata/COCOdata',P3Mrootpath = '../P3Mdata/MASKpatches',height=256,width=256,fill_mix_ratio=0.25):
    train_dataset = dataset.COCOP3MDataset(coco_path=COCOrootpath, p3m_path=P3Mrootpath,height=height, width=width,fill_mix_ratio=fill_mix_ratio)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return train_dataloader,train_dataset

def prepare_dataset_train_COCO(batch_size=1,COCOrootpath = '../P3Mdata/COCOdata',height=256,width=256,fill_mix_ratio=0.25):
    train_dataset = dataset.COCODataset(coco_path=COCOrootpath, height=height, width=width,fill_mix_ratio=fill_mix_ratio)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return train_dataloader,train_dataset

def prepare_dataset_Kodak(batch_size=1,rootpath = "../Kodak"):
    val_img_list, val_anno_list = dataset.make_datapath_list_for_Kodak(rootpath=rootpath)
    val_dataset = dataset.KodakDataset(val_img_list, val_anno_list, phase="test")
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return val_dataloader,val_img_list

def prepare_dataset_P3Meval(batch_size=1,rootpath = "../P3M-500-NP"):#P3M-500-NP
    val_img_list, val_anno_list = dataset.make_datapath_list_for_P3Meval(rootpath=rootpath)
    val_dataset = dataset.KodakDataset(val_img_list, val_anno_list, phase="test")
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return val_dataloader,val_img_list
