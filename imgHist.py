import cv2 as cv
import matplotlib.pyplot as plt

raw_img = cv.imread("./1413394989805760512.png", cv.IMREAD_COLOR)

raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

# plt.hist(raw_img.ravel(), 256, [0, 256])
# plt.show()

hist = cv.calcHist([raw_img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

img = cv.equalizeHist(raw_img)
clahe = cv.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
clahe_img = clahe.apply(raw_img)

plt.imshow(img, cmap="gray")
plt.axis('off')
plt.show()

plt.imshow(clahe_img, cmap="gray")
plt.axis('off')
plt.show()

hist = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()
