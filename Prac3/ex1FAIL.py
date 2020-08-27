import cv2 as cv
import numpy as np

img = [cv.imread('Images/prac03ex01img01.png'),cv.imread('Images/prac03ex01img02.png'),cv.imread('Images/prac03ex01img03.png')]

gray = [cv.cvtColor(img[0], cv.COLOR_BGR2GRAY),cv.cvtColor(img[1], cv.COLOR_BGR2GRAY),cv.cvtColor(img[2], cv.COLOR_BGR2GRAY)]

dst = [None]*3

for i in range(3):
	gray[i] = np.float32(gray[i])
	dst[i] = cv.cornerHarris(gray[i],2,3,0.04)
	dst[i] = cv.dilate(dst[i],None)
	img[i][dst>0.01*dst.max()]=[0,0,255]
	cv.imshow("Harris Corner Detected",img[i])

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()
