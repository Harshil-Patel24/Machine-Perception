import numpy as np
import cv2 as cv

img = cv.imread('Images/prac01ex02img01.png')

if img is None:
	print('Could not open image')
	exit(0)

f = open("prac01ex02crop.txt", "r")
a = f.readline().split()

xl = int(a[0])
yl = int(a[1])
xr = int(a[2])
yr = int(a[3])
 
cropped = img[yl:yr, xl:xr]

cv.imwrite("croppedImage.png", cropped)
cv.imshow('Cropped', cropped)
cv.waitKey(0)
