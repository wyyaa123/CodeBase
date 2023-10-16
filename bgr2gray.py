import cv2 as cv
import matplotlib.pyplot as plt

def hist(pic_path):
	img=cv.imread(pic_path,0);
	hist = cv.calcHist([img],[0],None,[256],[0,256])
	plt.subplot(121)
	plt.imshow(img,'gray')
	plt.xticks([])
	plt.yticks([])
	plt.title("Original")
	plt.subplot(122)
	plt.hist(img.ravel(),256,[0,256])
	plt.show()

img = cv.imread("./000047.png", cv.IMREAD_COLOR)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # plt.title('aug_img')
# plt.imshow(img, cmap='gray')
# plt.axis('off')

# plt.show()

# plt.subplot(121)
# plt.imshow(img,'gray')
# plt.xticks([])
# plt.yticks([])
# plt.title("Original")
# plt.subplot(122)
plt.hist(img.ravel(),256,[0,256])
plt.show()
