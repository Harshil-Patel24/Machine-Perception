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

roi = img[top:bottom, left:right]

border = cv.copyMakeBorder(roi, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=RED)

plt.subplot(231),plt.imshow(border,'gray'),plt.title('BORDER')

plt.show()
