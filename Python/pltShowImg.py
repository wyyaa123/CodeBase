# import torch
# from torch import nn
import cv2 as cv
import PIL
# from torch.utils.data import Dataset
import albumentations as albu
import matplotlib.pyplot as plt
import time
        
aug = albu.OneOf([
                  albu.HorizontalFlip(always_apply=True), #水平翻转
                  albu.ShiftScaleRotate(always_apply=True), #仿射变换（平移，缩放，旋转）
                  albu.Transpose(always_apply=True), #图像转置
                  albu.OpticalDistortion(always_apply=True), # 应用光学失真效果
                  albu.ElasticTransform(always_apply=True), # 应用弹性变换
                  ])

# aug = albu.Compose([albu.Resize(512, 512), albu.PadIfNeeded(512, 512)])

beg = time.time()
# raw_img = cv.imread("./assets/3.png", cv.IMREAD_COLOR)[:, :, (2, 1, 0)] # 0.163468599319458 seconds
raw_img = cv.imread("1.png", cv.IMREAD_COLOR)[:, :, ::-1] #  0.13281989097595215 seconds
during_time = time.time() - beg

img = aug(image = raw_img)['image']

plt.subplot(1, 2, 1)
plt.title('raw_img')
plt.imshow(raw_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('aug_img')
plt.imshow(img)
plt.axis('off')

plt.show()

print (during_time)
