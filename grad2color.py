import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./1413394989805760512.png", cv.IMREAD_GRAYSCALE)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

plt.imshow(img)
plt.show()

