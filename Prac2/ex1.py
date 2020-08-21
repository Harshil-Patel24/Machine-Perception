import cv2 as cv

img = cv.imread('Images/prac02ex01img01.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
luv = cv.cvtColor(img, cv.COLOR_BGR2LUV)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

cv.imshow("Gray:",gray)
cv.imshow("HSV:",hsv)
cv.imshow("LUV:",luv)
cv.imshow("LAB:",lab)

cv.waitKey(0)

