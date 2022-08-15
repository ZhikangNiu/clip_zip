# -*- coding: utf-8 -*-
# @Time    : 2022-07-12 19:01
# @Author  : Zhikang Niu
# @FileName: ClipDataset.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import os


class JsonDataset(Dataset):
    """
    使用的是Place365 json读取方式
    file_path: 读取标注文件的位置
    img_path : 图片数据集的位置
    transform: 对文件的变换操作
    """
    def __init__(self,file_path:str,
                      img_path:str,
                      transform=None,
                      phase='train'):
        # 读取文件
        self.df = pd.read_json(file_path)
        # 传入图片路径
        self.img_path = img_path
        self.transform = transform
        self.phase = phase


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.phase == 'train':
            img_name = self.df['image_dict'][index]['img_path']
            img_path = os.path.join(self.img_path,img_name)
            img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            #img = Image.open(img_path)
            #img = img.resize((224,224))
            label = int(self.df['image_dict'][index]['level_2'])
            if self.transform is not None:
                img = self.transform(img)
            else:
                transform =transforms.Compose([transforms.ToTensor()])
                img = transform(img)
            label = torch.tensor(label)
            return img,label

class single_file_dataset(Dataset):
    def __init__(self,file_path,label,transform=None):
        self.file_path = file_path
        self.img_list = []
        for img_path in os.listdir(self.file_path):
            img_path = os.path.join(self.file_path,img_path)
            img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
            if img is not None:
                img = Image.fromarray(img)
                self.img_list.append(img)
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        img = self.img_list[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform =transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(self.label)
        return img,label

