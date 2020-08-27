import cv2 as cv
import numpy as np


#Initialise an empty array of size 3
img = [None]*3 
gray = [None]*3
dst = [None]*3

#Read in each image
for ii in range(3):
	filename = 'Images/prac03ex01img0'+str(ii+1)+'.png'
	#decode = cv.imread(filename)
	#cv.imshow("Decode:",decode)
	#cv.waitKey()
	print( ii )
	img[ii] = cv.imread( filename )
	gray[ii] = cv.cvtColor(img[ii], cv.COLOR_BGR2GRAY) 
	gray[ii] = np.float32( gray[ii] )
	
	dst[ii] = cv.cornerHarris(gray[ii],2,3,0.04)

	#Result is dilated for marking the corners, not important
	dst[ii] = cv.dilate(dst[ii],None)	
	
	#Threshold for an optimal value, can vary this
	img[ii][dst[ii]>0.01*dst[ii].max()]=[0,0,255]
	
	cv.imshow("Image "+str(ii+1),img[ii])

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()

	



