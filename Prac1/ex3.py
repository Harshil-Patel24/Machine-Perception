import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

RED = [255,0,0]

img = cv.imread('Images/prac01ex02img01.png')

f = open("prac01ex02crop.txt", "r")
line = f.readline().split()

left = int(line[0])
right = int(line[2])
top = int(line[1])
bottom = int(line[3])

cv.rectangle(img, (top,left), (right,bottom), (0,0,255), 5)

cv.circle(img, (top,left), 10, (0,255,0), -1)
cv.circle(img, (right, top), 10, (0,255,0), -1)
cv.circle(img, (left, bottom), 10, (0,255,0), -1)
cv.circle(img, (right,bottom), 10, (0,255,0), -1)

cv.imshow("Drawn", img)
cv.waitKey(0)

