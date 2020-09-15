import numpy as np
import cv2 as cv

img = cv.imread('Images/prac04ex01img01.png', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(img, None)

print("Number of descriptors: " + str(len(des)))
print("Size of the descriptors: " + str(len(des[0])))

# img = cv.drawKeypoints(img,kp,img)

img = cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("Keypoints:", img)

res = cv.resize(des, None, fx=3, fy=3, interpolation= cv.INTER_LINEAR)

cv.imshow("Descriptors:", res)
cv.waitKey()







