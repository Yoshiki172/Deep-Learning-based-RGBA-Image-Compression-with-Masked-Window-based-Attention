import col_of_def.dataset as dataset
import col_of_def.imagedataset as imagedataset
import torch.utils.data as data
import torch
import torchvision
#def prepare_dataset_train(batch_size=1,rootpath = "./VOCdevkit/VOC2012/",height=256,width=256):
def prepare_dataset_train(batch_size=1,rootpath = "./VOCdevkit/VOC2012/",height=256,width=256):
    #train_img_list, train_anno_list, val_img_list, val_anno_list = dataset.make_datapath_list(rootpath=rootpath)
    train_img_list, train_anno_list= dataset.make_datapath_list_coco()
    #train_dataset = dataset.VOCDataset(train_img_list, train_anno_list, phase="train", height=height, width=width)
    train_dataset = dataset.COCODataset(train_img_list, train_anno_list, phase="train", height=height, width=width)
    #val_dataset = dataset.VOCDataset(val_img_list, val_anno_list, phase="val", height=height, width=width)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    #val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, train_img_list

def prepare_dataset_train_P3M(batch_size=1,rootpath = "./P3M",height=256,width=256):
    train_img_list = dataset.make_datapath_list_for_P3M(rootpath)
    train_dataset = dataset.P3MDataset(train_img_list, height=height, width=width)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return train_dataloader, train_img_list

def prepare_dataset_train_COCOP3M(batch_size=1,COCOrootpath = './P3Mdata/COCOdata',P3Mrootpath = './P3Mdata/MASKpatches',height=256,width=256,fill_mix_ratio=0.25):
    train_dataset = dataset.COCOP3MDataset(coco_path=COCOrootpath, p3m_path=P3Mrootpath,height=height, width=width,fill_mix_ratio=fill_mix_ratio)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return train_dataloader

def prepare_dataset_Kodak(batch_size=1,rootpath = "./Kodak"):
    val_img_list, val_anno_list = dataset.make_datapath_list_for_Kodak(rootpath=rootpath)
    val_dataset = dataset.KodakDataset(val_img_list, val_anno_list, phase="test")
    #val_dataset = dataset.VOCDataset(val_img_list, val_anno_list, phase="val", height=height, width=width)

    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    #val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_dataloader,val_img_list
