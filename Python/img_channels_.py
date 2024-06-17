import cv2 as cv
import numpy as np

img = cv.imread("./1717314772141232491.png", cv.IMREAD_COLOR)
img_bchannel = img[:, :, 0]
img_gchannel = img[:, :, 1]
img_rchannel = img[:, :, 2]

cv_image = np.dstack((img_rchannel, img_gchannel, img_bchannel))

cv.imshow("img", cv_image)
cv.waitKey()

