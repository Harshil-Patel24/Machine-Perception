import cv2 as cv
import numpy as np

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

diamond_gray = cv.cvtColor(diamond, cv.COLOR_BGR2GRAY)
dugong_gray = cv.cvtColor(dugong, cv.COLOR_BGR2GRAY)

# Lets create a scaled diamond
scaled_diamond = cv.resize(diamond_gray, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
rows, cols = scaled_diamond.shape[:2]

# Now to rotate... correctly -- so translate it first so we don't cut the corners
translation_matrix = np.float32([[1, 0, int(0.5*cols)], [0, 1, int(0.5*rows)]])
# Create a bigger area so the rotation doesn't truncate the corners
img_translation = cv.warpAffine(scaled_diamond, translation_matrix, (cols*2, rows*2))

# Do the rotatey stuff
M = cv.getRotationMatrix2D((cols, rows), 30, 1)
rotated_diamond = cv.warpAffine(img_translation, M, (cols * 2, rows * 2))



cv.imshow("Scaled and Rotated", rotated_diamond)
cv.waitKey()

# RESIZING -- Let's create two new images to isolate the subject













