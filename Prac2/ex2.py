import cv2 as cv
import numpy as np

img = cv.imread('Images/prac02ex02img01.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernelPrewit = np.array([[-1.0,0.0,1.0],[-1.0,0.0,1.0],[-1.0,0.0,1.0]])
kernelSobel = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
kernelLaplacain = np.array([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])
kernelGaussian = np.array([[1.0,4.0,7.0,4.0,1.0],[4.0,16.0,26.0,16.0,4.0],[7.0,26.0,41.0,26.0,7.0],[4.0,16.0,26.0,16.0,4.0],[1.0,4.0,7.0,4.0,1.0]])

gaussianBlur = cv.GaussianBlur(gray,(5,5),0)

prewit = cv.filter2D(gray,-1,kernelPrewit)
sobel = cv.filter2D(gray,-1,kernelSobel)
laplacian = cv.filter2D(gray,-1,kernelLaplacain)
gaussian = cv.filter2D(gray,-1,kernelGaussian)

cv.imshow("Prewit:",prewit)
cv.imshow("Sobel:",sobel)
cv.imshow("Laplacian:",laplacian)
cv.imshow("Gaussian:",gaussian)
cv.imshow("Gaussian Blur:",gaussianBlur)

cv.waitKey(0)

