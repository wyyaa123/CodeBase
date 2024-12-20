import cv2 as cv
import numpy as np
from pywt import dwt2, idwt2
import matplotlib.pyplot as plt
 
# 读取灰度图
img = cv.imread('./w700d1q75cms.jpg', cv.IMREAD_GRAYSCALE)

# 对img进行haar小波变换：
cA1, (cH1, cV1, cD1) = dwt2(img, 'haar')
cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar')
cA3, (cH3, cV3, cD3) = dwt2(cA2, 'haar')

# AH3 = np.concatenate([cA3, cH3], axis=1)
# VD3 = np.concatenate([cV3, cD3], axis=1)
# cA2 = np.concatenate([AH3, VD3], axis=0)
# cv.line(cA2, (cA2.shape[1] // 2 - 2, 0), (cA2.shape[1] // 2 - 2, cA2.shape[0]), (255, 255, 255), 4, 4)
# cv.line(cA2, (0, cA2.shape[0] // 2 - 2), (cA2.shape[1], cA2.shape[0] // 2 - 2), (255, 255, 255), 4, 4)

# AH2 = np.concatenate([cA2, cH2], axis=1)
# VD2 = np.concatenate([cV2, cD2], axis=1)
# cA1 = np.concatenate([AH2, VD2], axis=0)
# cv.line(cA1, (cA1.shape[1] // 2 - 2, 0), (cA1.shape[1] // 2 - 2, cA1.shape[0]), (255, 255, 255), 4, 4)
# cv.line(cA1, (0, cA1.shape[0] // 2 - 2), (cA1.shape[1], cA1.shape[0] // 2 - 2), (255, 255, 255), 4, 4)

# AH1 = np.concatenate([cA1, cH1], axis=1)
# VD1 = np.concatenate([cV1, cD1], axis=1)
# dwt_img = np.concatenate([AH1, VD1], axis=0)
# cv.line(dwt_img, (dwt_img.shape[1] // 2 - 2, 0), (dwt_img.shape[1] // 2 - 2, dwt_img.shape[0]), (255, 255, 255), 4, 4)
# cv.line(dwt_img, (0, dwt_img.shape[0] // 2 - 2), (dwt_img.shape[1], dwt_img.shape[0] // 2 - 2), (255, 255, 255), 4, 4)

# high = cH1 + cV1 + cD1 - cA1
fig, axes0 = plt.subplots(1, 1)
axes0.imshow(cA1, cmap='gray')
axes0.set_title('raw')
axes0.axis('off')


fig, axes1 = plt.subplots(1, 1)
axes1.imshow(cH1, cmap='gray')
axes1.set_title('LH')
axes1.axis('off')

fig, axes2 = plt.subplots(1, 1)
axes2.imshow(cV1, cmap='gray')
axes2.set_title('HL')
axes2.axis('off')

fig, axes3 = plt.subplots(1, 1)
axes3.imshow(cD1, cmap='gray')
axes3.set_title('HH')
axes3.axis('off')

plt.show()
