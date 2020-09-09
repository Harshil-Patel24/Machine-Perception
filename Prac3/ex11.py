import numpy as np
import cv2 as cv

filename = 'Images/prac03ex01img0'

img = gray = dst = [None]*3

for ii in range(3):

	img[ii] = cv.imread(filename + str(ii+1) + '.png')
	#cv.imshow('test',img[ii])
	#cv.waitKey(0)
	gray[ii] = cv.cvtColor(img[ii], cv.COLOR_BGR2GRAY)

	gray[ii] = np.float32(gray[ii])
	
	dst[ii] = cv.cornerHarris(gray[ii],2,3,0.04)
	
	dst[ii] = cv.dilate(dst[ii],None)
	
	#cv.imshow('test',gray[ii])
	#cv.waitKey(0)
	
	img.pop(ii)[dst[ii]>0.01*dst[ii].max()]=[0,0,255]

	cv.imshow('dst',img[ii])

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()


