import cv2 as cv
import numpy as np


#Initialise an empty array of size 3
img = gray = dst = corners = [None]*3 

#Read in each image
for ii in range(3):
	filename = 'Images/prac03ex01img0'+str(ii+1)+'.png'

	img[ii] = cv.imread( filename )
	#img2[ii] = img[ii].copy()
	gray[ii] = cv.cvtColor(img[ii], cv.COLOR_BGR2GRAY) 
	gray[ii] = np.float32( gray[ii] )

	#print(type(gray[ii]))
	#cv.imshow("test", gray[ii])
	#cv.waitKey(0)

	#Harris Corner detector stuff
	dst[ii] = cv.cornerHarris(gray[ii],2,3,0.04)
	#Result is dilated for marking the corners, not important
	dst[ii] = cv.dilate(dst[ii],None)
	#Threshold for an optimal value, can vary this
	
	transform = np.asarray([dst[ii]>0.01*dst[ii].max()])
	transform = transform.astype(np.int)

	transform = transform*[0,0,255]

	for pixel in img[ii]:
		
		
	
	#print(transform)
	
	#[pixel = [0,0,255] for pixel in transform]

	print(transform)
	#img[ii][dst[ii]>0.01*dst[ii].max()]=[0,0,255]
	
	#[img[ii][pixel] for pixel in  expression 	
	
	#Shi-Tomasi detector stuff
	#corners[ii] = cv.goodFeaturesToTrack(gray,25,0.01,10)
	#corners[ii] = np.int0(corners[ii])

	#for jj in corners[ii]:
	#	x,y = jj.ravel()
	#	cv.circle(img2[ii])	

	cv.imshow("Harris - Image: "+str(ii+1),img[ii])
	#cv.imshow("Shi-Tomasi - Image "+str(ii+1),img2[ii])

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()

	



