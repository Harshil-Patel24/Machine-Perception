import numpy as np
import cv2 as cv
import math
import sys

img = gray = dst = cdst = cdstP = lines = linesP = [None]*3

filenames = ['Images/prac03ex03img02.jpg','Images/prac03ex03img01.png','Images/prac03ex01img01.png']

for ii in range(3):
	#cv.imshow("img", img)
	#cv.waitKey(0)

	img[ii] = cv.imread(filenames[ii])

	gray[ii] = cv.cvtColor(img[ii], cv.COLOR_BGR2GRAY)

	if img[ii] is None:
		print('Error opening image')

	# cv.imshow("Img", gray[ii])
	# cv.waitKey(0)

	dst[ii] = cv.Canny(gray[ii], 50, 200, None, 3)

	# cv.imshow("dst", dst[ii])
	# cv.waitKey(0)

	cdst[ii] = cv.cvtColor(dst[ii], cv.COLOR_GRAY2BGR)

	# cv.imshow("dst", cdst[ii])
	# cv.waitKey(0)

	#cdst = np.copy(dst)
	cdstP[ii] = np.copy(cdst[ii])

	cv.imshow("dst", dst[ii])
	cv.waitKey(0)

	lines[ii] = cv.HoughLines(dst[ii], 1, (np.pi / 180), 150, None, 0, 0)

	if lines[ii] is not None:
		for i in range(0, len(lines[ii])):
			rho = lines[ii][i][0][0]
			theta = lines[ii][i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
			cv.line(cdst[ii], pt1, pt2, (0,0,255), 1, cv.LINE_AA)

	linesP[ii] = cv.HoughLinesP(dst[ii], 1, np.pi / 180, 50, 10)

	if linesP[ii] is not None:
		for i in range(0, len(linesP[ii])):
			l = linesP[ii][i][0]
			cv.line(cdstP[ii], (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)

	cv.imshow("Source", img[ii])
	cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst[ii])
	cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP[ii])

cv.waitKey()






