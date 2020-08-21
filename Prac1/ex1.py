import numpy as np
import cv2 as cv

img = cv.imread('../../Downloads/Lenna.png')

if img is None:
	print( 'Could not open or read the image!' )
	exit(0)
 
print( "Image name: prac01ex01img01.png" )
print( "Image dimensions: ", img.shape )

res = cv.resize( img, None, fx=0.5, fy=0.5, interpolation = cv.INTER_AREA)

bgrPlanes = cv.split( img )

histSize = 10
histRange = (0, 256) #Upper bound is exclusive
accumulate = False

bHist = cv.calcHist(bgrPlanes, [0], None, [histSize], histRange, accumulate=accumulate)
gHist = cv.calcHist(bgrPlanes, [1], None, [histSize], histRange, accumulate=accumulate)
rHist = cv.calcHist(bgrPlanes, [2], None, [histSize], histRange, accumulate=accumulate)

histW = 1024
histH = 800
binW = int( round( histW / histSize ) )

histImage = np.zeros( ( histH, histW, 3 ), dtype=np.uint8 )

cv.normalize( bHist, bHist, alpha=0, beta=histH, norm_type=cv.NORM_MINMAX)
cv.normalize( gHist, gHist, alpha=0, beta=histH, norm_type=cv.NORM_MINMAX)
cv.normalize( rHist, rHist, alpha=0, beta=histH, norm_type=cv.NORM_MINMAX)

for i in range( 1, histSize ):
	cv.line( histImage, ( binW * ( i - 1 ), histH - int( bHist[i-1] ) ), ( binW * (i), histH - int( bHist[i] ) ), ( 255, 0, 0 ), thickness=2)
	cv.line( histImage, ( binW * ( i - 1 ), histH - int( gHist[i-1] ) ), ( binW * (i), histH - int( gHist[i] ) ), ( 0, 255, 0 ), thickness=2)
	cv.line( histImage, ( binW * ( i - 1 ), histH - int( rHist[i-1] ) ), ( binW * (i), histH - int( rHist[i] ) ), ( 0, 0, 255 ), thickness=2)

cv.imshow('Source image', img)
cv.imshow('Histogram', histImage)
cv.imshow('Resized', res)
cv.waitKey()
