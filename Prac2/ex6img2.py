import cv2 as cv
import numpy as np

img = cv.imread('Images/prac02ex06img02.png',0)

kernel = np.ones((5,5),np.uint8)

erode = cv.erode(img,kernel,iterations = 1)
dilate = cv.dilate(img,kernel,iterations = 1)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

cv.imshow("Erosion:", erode)
cv.imshow("Dilation:", dilate)
cv.imshow("Opening:", opening)
cv.imshow("Closing:", closing)
cv.imshow("Gradient:", gradient)
cv.imshow("Tophat:", tophat)
cv.imshow("Blackhat:", blackhat)

cv.waitKey()


