import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
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

# ---------------- Histograms ----------------

# Now we use these to check the variance/invariance of certain key-point detections
# We now have 10 images:
# diamond, dugong
# scaled_diamond, scaled_dugong
# rotated_30_diamond, rotated_30_dugong
# rotated_90_diamond, rotated_90_dugong
# rotated_scaled_diamond, rotated_scaled_dugong

images = [diamond, scaled_diamond, rotated_30_diamond, rotated_90_diamond, rotated_scaled_diamond, \
    dugong, scaled_dugong,  rotated_30_dugong,  rotated_90_dugong, rotated_scaled_dugong]

# https://matplotlib.org/3.1.1/gallery/statistics/histogram_multihist.html

bins = 50

figure, axes = plt.subplots(nrows=2, ncols=5)
axes = axes.flatten()

for ii in range(len(images)):
    axes[ii].hist(images[ii].ravel(), bins, [0, 256])

axes[0].set_title("Diamond")
axes[1].set_title("Diamond Scaled")
axes[2].set_title("Diamond Rotated 30")
axes[3].set_title("Diamond Rotated 90")
axes[4].set_title("Diamond Scaled and Rotated")
axes[5].set_title("Dugong")
axes[6].set_title("Dugong Scaled")
axes[7].set_title("Dugong Rotated 30")
axes[8].set_title("Dugong Rotated 90")
axes[9].set_title("Dugong Scaled and Rotated")


figure.tight_layout()
figure.subplots_adjust(left=0.05, wspace=0.5)
plt.show()

# ---------------- Harris Corner Detection ----------------

# diamong_harris = cornerHarris(diamond)
# dugong_harris = cornerHarris(dugong)
# diamond_30_harris = cornerHarris(rotated_30_diamond)
# dugong_30_harris = cornerHarris(rotated_30_dugong)
# diamond_90_harris = cornerHarris(rotated_90_diamond)
# dugong_90_harris = cornerHarris(rotated_90_dugong)
# diamond_scaled_harris = cornerHarris(scaled_diamond)
# dugong_scaled_harris = cornerHarris(scaled_dugong)
# diamond_scaled_rotated_harris = cornerHarris(rotated_scaled_diamond)
# dugong_scaled_rotated_harris = cornerHarris(rotated_scaled_dugong)

# corners = [cornerHarris(diamond), cornerHarris(scaled_diamond), cornerHarris(rotated_30_diamond), cornerHarris(rotated_90_diamond), cornerHarris(rotated_scaled_diamond), \
    # cornerHarris(dugong), cornerHarris(scaled_dugong), cornerHarris(rotated_30_dugong), cornerHarris(rotated_90_dugong), cornerHarris(rotated_scaled_dugong)]

corners = []

for ii, img in enumerate(images):
    corners.append(cornerHarris(images[ii]))

# https://www.delftstack.com/howto/matplotlib/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/

# width = 5
# height = 5 
rows = 2
cols = 5
axes = []
figure = plt.figure()

for ii in range(rows * cols):
    axes.append(figure.add_subplot(rows, cols, ii + 1))
    plt.imshow(corners[ii])

axes[0].set_title("Diamond")
axes[1].set_title("Diamond Scaled")
axes[2].set_title("Diamond Rotated 30")
axes[3].set_title("Diamond Rotated 90")
axes[4].set_title("Diamond Scaled and Rotated")
axes[5].set_title("Dugong")
axes[6].set_title("Dugong Scaled")
axes[7].set_title("Dugong Rotated 30")
axes[8].set_title("Dugong Rotated 90")
axes[9].set_title("Dugong Scaled and Rotated")

figure.tight_layout()
plt.show()

# ---------------- SIFT ----------------

sifts = []

for ii, img in enumerate(images):
    sifts.append(siftKeyPoints(images[ii]))

displayImages(sifts)

