import cv2 as cv
import numpy as np

img = cv.imread("./images/3.png", cv.IMREAD_COLOR)
img_bchannel = img[:, :, 0]
img_gchannel = img[:, :, 1]
img_rchannel = img[:, :, 2]

cv_image = np.dstack((img_bchannel, img_gchannel, img_rchannel))

cv.imshow("img", cv_image)
cv.waitKey()

