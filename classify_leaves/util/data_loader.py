# -*- encoding: utf-8 -*-
'''
@File    :   data_loader.py
@Time    :   2023/09/16 17:42:24
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib


import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage
from albumentations.pytorch import ToTensorV2
import albumentations as albu
from tqdm import tqdm

import os


class BaseDataset(Dataset):
    def __init__(self, data_csv: str, transform, isTest = False):

        super(BaseDataset, self).__init__()
        self.data_csv = data_csv
        self.transform = transform
        self.isTest = isTest
        self.img_path = [os.path.join("/home/nrc/classify_leaves/data", image) for image in data_csv["image"]]
        if not isTest:
            self.label = list(data_csv["number"])

    def __getitem__(self, index):
        raw_img = cv.imread(self.img_path[index], cv.IMREAD_COLOR)[:, :, ::-1]
        raw_img = self.transform(image=raw_img)["image"]

        if not self.isTest:
            label = self.label[index]
            return raw_img, label
        else:
            return raw_img

    def __len__(self):
        return self.data_csv.shape[0]

def read_csv(path, isTest = False):
    data = pd.read_csv(path)

    if isTest:
        return data
    
    labels = list(data["label"])
    # unique_labels = list(set(list(labels)))
    unique_labels = list(data["label"].unique())
    label_nums = []
    for label in labels:
        label_nums.append(unique_labels.index(label))
    
    data["number"] = label_nums

    return data, unique_labels

# def __normal_image(self):
#     # dataset = ImageFolder("./", transform=ToTensor())
#     # num_imgs = self.train_data.shape[0]
#     num_imgs = self.data.shape[0]
#     bar = tqdm(range(num_imgs))
#     for img_seq in bar:#遍历数据集的张量和标签
#         raw_img = cv.imread(self.img_path + self.data["image"][img_seq], cv.IMREAD_COLOR)
#         raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)
#         # raw_img = cv.imread(self.train_data["image"][img_seq], cv.IMREAD_COLOR)
#         raw_img = raw_img.astype(np.float32) / 255.
#         for i in range(3):#遍历图片的RGB三通道
#             # 计算每一个通道的均值和标准差
#             self.means[i] += raw_img[:, :, i].mean()
#             self.std[i] += raw_img[:, :, i].std()
#     self.means = np.array(self.means) / num_imgs
#     self.std = np.array(self.std) / num_imgs#要使数据集归一化，均值和方差需除以总图片数量


# if __name__ == "__main__":
        
    # data = BaseDataset("/home/nrc/classify_leaves/data/train.csv")
    # print (data.means)
    # print (data.std)
    # means = [0.7586179, 0.7776906, 0.75756389]
    # std = [0.18266079, 0.14999901, 0.15760997]
    # raw_img = cv.imread(r"C:\Users\wyyaa123\Desktop\classify_leaves\data\images\1.jpg", cv.IMREAD_COLOR)
    # # raw_img = raw_img.astype(np.float32) / 255.

    # # b_img = raw_img[:, :, 0]
    # # g_img = raw_img[:, :, 1]
    # # r_img = raw_img[:, :, 2]

    # # b_img = (b_img - means[0]) / std[0]
    # # g_img = (g_img - means[1]) / std[1]
    # # r_img = (r_img - means[2]) / std[2]

    # # raw_img = np.dstack((b_img, g_img, r_img))

    # aug = albu.Compose([albu.Resize(224, 224),
    #                     albu.Normalize(mean=means, std=std, max_pixel_value=255),
    #                     ToTensorV2()], additional_targets={"image_2": "image"})

    # raw_img = aug(image=raw_img, image_2=raw_img)

    # print (raw_img["image"].shape)
    pass

