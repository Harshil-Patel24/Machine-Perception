import numpy as np
import cv2 as cv

img = cv.imread('Images/prac04ex01img01.png', cv.IMREAD_GRAYSCALE)
scaled = cv.resize(img, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
rows,cols = scaled.shape

M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 30, 1)
rot = cv.warpAffine(scaled, M, (cols, rows))

# cv.imshow("Rotated and Scaled", rot)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(rot, None)

print("Number of descriptors: " + str(len(des)))
print("Size of the descriptors: " + str(len(des[0])))

# img = cv.drawKeypoints(img,kp,img)

rot = cv.drawKeypoints(rot, kp, rot, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("Keypoints:", rot)

res = cv.resize(des, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)

cv.imshow("Descriptors:", res)
cv.waitKey()







