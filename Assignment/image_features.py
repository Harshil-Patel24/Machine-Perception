import cv2 as cv
import numpy as np
from tools import *

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

imgs = [diamond, dugong]
scaled = rows = cols = translations = img_translations = M = rotated = [None] * 2

# ---------------- SCALE AND ROTATE ----------------

# Lets create a scaled version of the diamonds and dugongs
scaled_diamond = cv.resize(diamond, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)
rows_diamond, cols_diamond = scaled_diamond.shape[:2]

scaled_dugong = cv.resize(dugong, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)
rows_dugong, cols_dugong = scaled_dugong.shape[:2]

# Create 2 rotations, as SIFT is scale invarient up to 2.5x and 40 degree rotation invarient
rotated_90_diamond = cv.rotate(diamond, cv.ROTATE_90_CLOCKWISE)
rotated_90_dugong = cv.rotate(dugong, cv.ROTATE_90_CLOCKWISE)

rotated_30_diamond = rotate(diamond, 30)
rotated_30_dugong = rotate(dugong, 30)

# Hit 'em with both
rotated_scaled_diamond = cv.rotate(scaled_diamond, cv.ROTATE_90_CLOCKWISE)
rotated_scaled_dugong = cv.rotate(scaled_dugong, cv.ROTATE_90_CLOCKWISE)

images = [diamond, scaled_diamond, rotated_30_diamond, rotated_90_diamond, rotated_scaled_diamond, \
    dugong, scaled_dugong,  rotated_30_dugong,  rotated_90_dugong, rotated_scaled_dugong]

# ---------------- SIFT ----------------

sifts = []

for ii, img in enumerate(images):
    sifts.append(siftKeyPoints(images[ii]))





