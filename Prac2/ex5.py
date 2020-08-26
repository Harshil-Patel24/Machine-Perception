import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Images/prac02ex05img01.png',0)

equ = cv.equalizeHist(img)

cv.imshow("Equalized:", equ)
cv.waitKey()

