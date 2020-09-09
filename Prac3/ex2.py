import cv2 as cv
import numpy as np

img = gray = prewit = sobel = [None]*2

img[0] = cv.imread('Images/prac03ex02img01.jpg')
gray[0] = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

img[1] = cv.imread('Images/prac03ex01img01.png')
gray[1] = cv.cvtColor(img[1], cv.COLOR_BGR2GRAY)

kernelPrewit = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernelSobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

for ii in range(2):
	prewit[ii] = cv.filter2D(gray[ii],-1,kernelPrewit)
	sobel[ii] = cv.filter2D(gray[ii],-1,kernelSobel)

	cv.imshow("Prewit: "+str(ii),prewit[ii])
	cv.imshow("Sobel: "+str(ii),sobel[ii])

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()






