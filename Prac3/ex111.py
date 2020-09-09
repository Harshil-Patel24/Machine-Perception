import cv2 as cv
import numpy as np

filename = 'Images/prac03ex01img01.png'

img = cv.imread(filename)

#cv.imshow("Image:",img)
#cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)

img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow("Harris: ", img)

if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()






