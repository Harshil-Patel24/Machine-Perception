import cv2 as cv
import numpy as np
from tools import *

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

# We'll probably need the grays later
diamond_gray = cv.cvtColor(diamond, cv.COLOR_BGR2GRAY)
dugong_gray = cv.cvtColor(dugong, cv.COLOR_BGR2GRAY)

# ---------------- SCALE AND ROTATE ----------------

# The idea to translate before rotating was taken from -
# https://subscription.packtpub.com/book/application_development/9781785283932/1/ch01lvl1sec12/image-rotation

# Lets create a scaled diamond
scaled_diamond = cv.resize(diamond, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
rows, cols = scaled_diamond.shape[:2]

# Some boring ol' maths here
hypotenuse = np.sqrt((rows ** 2) + (cols ** 2))
theta = np.arctan(rows / cols)
w = hypotenuse * np.sin(theta + (np.pi / 6))
h = hypotenuse * np.cos(theta - (np.pi / 6))

print("Rows = " + str(rows / h))
print("Cols = " + str(cols / w))
print("Hypotenuse = " + str(hypotenuse))
print("Theta = " + str(theta))
# print("Width Confirm = " + str(np.sin(theta)*hypotenuse))
print("Width = " + str(w))
print("Height = " + str(h))


# Now to rotate... correctly -- so translate it first so we don't cut the corners
translation_matrix = np.float32([[1, 0, int(0.5*cols)], [0, 1, int(0.5*rows)]])
# translation_matrix = np.float32([[1, 0, int((cols / (2*w))*cols)], [0, 1, int((rows / (2*h))*rows)]])

# Create a bigger area so the rotation doesn't truncate the corners
img_translation = cv.warpAffine(scaled_diamond, translation_matrix, (cols*2, rows*2))
# img_translation = cv.warpAffine(scaled_diamond, translation_matrix, (int(w*2), int(h*2)))

# Do the rotatey stuff
M = cv.getRotationMatrix2D((cols, rows), 30, 1)
rotated_scaled_diamond = cv.warpAffine(img_translation, M, (cols * 2, rows * 2))
# rotated_scaled_diamond = cv.warpAffine(img_translation, M, (int(w*2), int(h*2)))

# cv.imshow("Scaled and Rotated", rotated_scaled_diamond)
# cv.waitKey()

# ---------------- MAKE HISTOGRAMS ----------------

rotated_scaled_diamond_hist_image = histogramCalc(rotated_scaled_diamond)
scaled_hist_image = histogramCalc(scaled_diamond)
diamond_hist_image = histogramCalc(diamond)

# cv.imshow("Rotated dugong", rot)
# cv.imshow("Rotated histogram", rot_hist)
# cv.imshow("Dugong histogram", dugong_hist)
cv.imshow("Rotated Scaled Diamond", rotated_scaled_diamond)
cv.imshow("Histogram of Diamond", diamond_hist_image)
cv.imshow("Histogram of Rotated Scaled Diamond", rotated_scaled_diamond_hist_image)
cv.imshow("Histogram of Scaled Diamond", scaled_hist_image)
cv.waitKey()

# ---------------- HARRIS CORNER DETECTION ----------------

diamong_harris = cornerHarris(diamond)
diamond_scaled_rotated_harris = cornerHarris(rotated_scaled_diamond)


# cv.imshow("Diamond corner detection", diamong_harris)
# cv.imshow("Diamond scaled and rotated corner detection", diamond_scaled_rotated_harris)
#
# if cv.waitKey(0) & 0xff == 27:
# 	cv.destroyAllWindows()


# ---------------- SIFT ----------------
diamond_sift = siftKeyPoints(diamond)
rotated_scaled_diamond_sift = siftKeyPoints(rotated_scaled_diamond)

cv.imshow("Diamond Sift", diamond_sift)
cv.imshow("Diamond Rotated and Scaled Sift", rotated_scaled_diamond_sift)

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()


