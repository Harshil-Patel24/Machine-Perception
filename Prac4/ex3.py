from matplotlib import pyplot as plt
import cv2 as cv

img = cv.imread('Images/prac04ex02img01.png', cv.IMREAD_GRAYSCALE)

# hist = cv.calcHist(img, [0], None, [256], [0, 256])

plt.hist(img.ravel(), 200, [0, 256])
plt.show()











