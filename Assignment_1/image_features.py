import cv2 as cv
import numpy as np
from tools import *

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

imgs = [diamond, dugong]
scaled = rows = cols = translations = img_translations = M = rotated = [None] * 2

# This just extracts the top right diamond
cropped_diamond = diamond[45:70, 5:30]

# This extracts the mother and her calf
cropped_dugong = dugong[205:280, 383:433]

# ---------------- SCALE AND ROTATE ----------------

# Lets create a scaled version of the diamonds and dugongs
scaled_diamond = cv.resize(diamond, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)
rows_diamond, cols_diamond = scaled_diamond.shape[:2]

cropped_scaled_diamond = scaled_diamond[68:105, 8:45]

scaled_dugong = cv.resize(dugong, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)
rows_dugong, cols_dugong = scaled_dugong.shape[:2]

cropped_scaled_dugong = scaled_dugong[310:420, 575:650]

# Create 2 rotations, as SIFT is scale invarient up to 2.5x and 40 degree rotation invarient
rotated_90_diamond = cv.rotate(diamond, cv.ROTATE_90_CLOCKWISE)
rotated_90_dugong = cv.rotate(dugong, cv.ROTATE_90_CLOCKWISE)

cropped_90_diamond = rotated_90_diamond[8:30, 210:235]
cropped_90_dugong = rotated_90_dugong[380:430, 200:275]

rotated_30_diamond = rotate(diamond, 30)
rotated_30_dugong = rotate(dugong, 30)

print(rotated_30_dugong.shape)

cropped_30_diamond = rotated_30_diamond[122:142, 35:55]
cropped_30_dugong = rotated_30_dugong[390:475, 450:500]

# Hit 'em with both

images = [diamond, scaled_diamond, rotated_30_diamond, rotated_90_diamond, \
    dugong, scaled_dugong,  rotated_30_dugong,  rotated_90_dugong]

cropped = [cropped_diamond, cropped_scaled_diamond, cropped_30_diamond, cropped_90_diamond,\
        cropped_dugong, cropped_scaled_dugong, cropped_30_dugong, cropped_90_dugong]

# ---------------- SIFT ----------------

sifts_cropped = []

filenames_diamond = ["Results\Features\SIFT\cropped_diamond.png", "Results\Features\SIFT\cropped_scaled_diamond.png", \
    "Results\Features\SIFT\cropped_30_diamond.png", "Results\Features\SIFT\cropped_90_diamond.png"]
filenames_dugong = ["Results\Features\SIFT\cropped_dugong.png", "Results\Features\SIFT\cropped_scaled_dugong.png", \
    "Results\Features\SIFT\cropped_30_dugong.png", "Results\Features\SIFT\cropped_90_dugong.png"]

filenames_single = ["Results\Features\SIFT\cropped_single_diamond.png", "Results\Features\SIFT\cropped_scaled_single_diamond.png", \
    "Results\Features\SIFT\cropped_30_single_diamond.png", "Results\Features\SIFT\cropped_90_single_diamond.png", \
    "Results\Features\SIFT\cropped_single_dugong.png", "Results\Features\SIFT\cropped_scaled_single_dugong.png", \
    "Results\Features\SIFT\cropped_30_single_dugong.png", "Results\Features\SIFT\cropped_90_single_dugong.png"]

for ii, img in enumerate(images):
    sifts_cropped.append(siftKeyPoints(cropped[ii]))


for ii in range(len(cropped) - 4):
    siftKeyPointMatcher(cropped[ii], images[0], filenames_diamond[ii])

for ii in range(4, len(cropped)):
    siftKeyPointMatcher(cropped[ii], images[4], filenames_dugong[ii - 4])

saveImages(sifts_cropped, filenames_single)

# ---------------- HOG ---------------


