import cv2 as cv
import numpy as np

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

Z_diamond = diamond.reshape((-1, 3))
Z_diamond = np.float32(Z_diamond)
Z_dugong = dugong.reshape((-1, 3))
Z_dugong = np.float32(Z_dugong)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.2)
K = 2
ret_diamond, label_diamond, center_diamond = cv.kmeans(Z_diamond, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
ret_dugong, label_dugong, center_dugong = cv.kmeans(Z_dugong, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

center_diamond = np.uint8(center_diamond)
center_dugong = np.uint8(center_dugong)
res_diamond = center_diamond[label_diamond.flatten()]
res_dugong = center_dugong[label_dugong.flatten()]
res2_diamond = res_diamond.reshape((diamond.shape))
res2_dugong = res_dugong.reshape((dugong.shape))

# res2_dugong[label_dugong == 2] = [0, 0, 0]
# res2_dugong[:,:,2] = np.zeros([res2_dugong.shape[0], res2_dugong.shape[1]])

cv.imshow("Diamond k-means: ", res2_diamond)
cv.imshow("Dugong k-means: ", res2_dugong)
cv.waitKey()
