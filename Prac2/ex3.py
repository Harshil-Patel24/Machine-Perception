import cv2 as cv
import numpy as np

img = cv.imread('Images/prac02ex03img01.jpg')

median = cv.medianBlur(img,5)

cv.imshow('Median Filter:', median)
cv.waitKey(0)
