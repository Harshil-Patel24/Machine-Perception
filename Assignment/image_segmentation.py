import cv2 as cv
import numpy as np

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
# MAYBE https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

# Removing the green colour channel makes the dugong stand out more from the background
dugong[:,:,1] = np.zeros([dugong.shape[0], dugong.shape[1]])

kernel = np.ones((2,2))

dugong = cv.morphologyEx(dugong, cv.MORPH_OPEN, kernel)

# cv.imshow("RED", dugong)
# cv.waitKey()

# diamond = cv.cvtColor(diamond, cv.COLOR_BGR2HSV)
# dugong = cv.cvtColor(dugong, cv.COLOR_BGR2HSV)

Z_diamond = diamond.reshape((-1, 3))
Z_diamond = np.float32(Z_diamond)
Z_dugong = dugong.reshape((-1, 3))
Z_dugong = np.float32(Z_dugong)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.2)
K = 3
ret_diamond, labels_diamond, center_diamond = cv.kmeans(Z_diamond, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
ret_dugong, labels_dugong, center_dugong = cv.kmeans(Z_dugong, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

center_diamond = np.uint8(center_diamond)
center_dugong = np.uint8(center_dugong)

labels_diamond = labels_diamond.flatten()
labels_dugong = labels_dugong.flatten()

segmented_diamond = center_diamond[labels_diamond.flatten()]
segmented_dugong = center_dugong[labels_dugong.flatten()]

# Make the white segment black
color_diamond = 1
segmented_diamond[labels_diamond == color_diamond] = [0, 0, 0]

# Make the water around the dugong segment black
color_dugong_1 = 1
segmented_dugong[labels_dugong == color_dugong_1] = [0, 0, 0]

color_dugong_2 = 2
segmented_dugong[labels_dugong == color_dugong_2] = [0, 0, 0]

segmented_diamond = segmented_diamond.reshape(diamond.shape)
segmented_dugong = segmented_dugong.reshape(dugong.shape)

cv.imshow("Diamond: ", segmented_diamond)
cv.imshow("Dugong: ", segmented_dugong)
cv.waitKey()