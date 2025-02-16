# import torch
# from torch import nn
import cv2 as cv
import PIL
# from torch.utils.data import Dataset
import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import time

def cal_sharpness(image: cv.Mat):

    # cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_sharpness = np.mean(sobel)

    return sobel_sharpness

def cal_clarity(image: cv.Mat):
    laplacian = cv.Laplacian(image, cv.CV_64F)
    laplacian_clarity = np.var(laplacian)

    return laplacian_clarity

blur = cv.imread("blur.png", cv.IMREAD_GRAYSCALE)
sharp = cv.imread("reg.png", cv.IMREAD_GRAYSCALE)

blur_sharpness = cal_sharpness(blur)
sharp_sharpness = cal_sharpness(sharp)

blur_clarity = cal_clarity(blur)
sharp_clarity = cal_clarity(sharp)

print("blur_sharpness: , sharp_sharpness: ", blur_sharpness, sharp_sharpness)
print("blur_clarity: , sharp_clarity: ", blur_clarity, sharp_clarity)

print("锐度提高了%f" % (sharp_sharpness - blur_sharpness))
print("清晰度提高了%f" % (sharp_clarity - blur_clarity))

# aug = albu.OneOf([
#                   albu.HorizontalFlip(always_apply=True), #水平翻转
#                   albu.ShiftScaleRotate(always_apply=True), #仿射变换（平移，缩放，旋转）
#                   albu.Transpose(always_apply=True), #图像转置
#                   albu.OpticalDistortion(always_apply=True), # 应用光学失真效果
#                   albu.ElasticTransform(always_apply=True), # 应用弹性变换
#                   ])

# # aug = albu.Compose([albu.Resize(512, 512), albu.PadIfNeeded(512, 512)])

# beg = time.time()
# # raw_img = cv.imread("./assets/3.png", cv.IMREAD_COLOR)[:, :, (2, 1, 0)] # 0.163468599319458 seconds
# raw_img = cv.imread("1.png", cv.IMREAD_COLOR)[:, :, ::-1] #  0.13281989097595215 seconds
# during_time = time.time() - beg

# img = aug(image = raw_img)['image']

# plt.subplot(1, 2, 1)
# plt.title('raw_img')
# plt.imshow(raw_img)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('aug_img')
# plt.imshow(img)
# plt.axis('off')

# plt.show()

# print (during_time)
