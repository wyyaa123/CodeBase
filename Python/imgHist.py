import cv2 as cv
import numpy as np
from PIL import Image
import skimage.exposure as exposure
import matplotlib.pyplot as plt

raw_img = cv.imread("./1413394997605760512.png", cv.IMREAD_COLOR)

raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

# plt.hist(raw_img.ravel(), 256, [0, 256])
# plt.show()

hist = cv.calcHist([raw_img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

img = cv.equalizeHist(raw_img)
img_AHE = exposure.equalize_adapthist(img)
# img_AHE = Image.fromarray(np.uint8(img1 * 255))
# img_AHE = np.array(img_AHE)
clahe = cv.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
clahe_img = clahe.apply(raw_img)


## 直方图均衡化处理（HE）
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.show()

hist = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

## 自适应直方图均衡化处理(AHE)
plt.imshow(img_AHE, cmap="gray")
plt.axis('off')
plt.show()

hist = cv.calcHist([clahe_img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

## 限制对比度自适应直方图均衡化处理（CLAHE），也叫局部直方图均衡化处理
plt.imshow(clahe_img, cmap="gray")
plt.axis('off')
plt.show()

hist = cv.calcHist([clahe_img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()
