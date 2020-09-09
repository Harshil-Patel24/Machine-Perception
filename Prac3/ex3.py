import numpy as np
import cv2 as cv
import math
import sys


img = cv.imread('Images/prac03ex02img01.jpg')

#cv.imshow("img", img)
#cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

if img is None: 
	print('Error opening image')

#cv.imshow("Img", gray)
#cv.waitKey(0)

dst = cv.Canny(gray, 50, 200, None, 3)

#cv.imshow("dst", dst)
#cv.waitKey(0)

cdst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

cdstP = np.copy(cdst)


lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
	for i in range(0, len(lines)):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		a = math.cos(theta)
		b = math.sin(theta)
		x0 = a * rho
		y0 = b * rho
		pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
		pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
		cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, 10)

if linesP is not None:
	for i in range(0, len(linesP)):
		l = linesP[i][0]
		cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

cv.imshow("Source", img)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.imshow("Detected Lines (in red) - Probablilistic Line Transform", cdstP)

cv.waitKey()






