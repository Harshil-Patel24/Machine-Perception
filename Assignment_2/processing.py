import cv2 as cv
import numpy as np
import os
from tools import *
from matplotlib import pyplot as plt

knn = trainKNN()

for ii, fname in enumerate(os.listdir('train')):    
    if fname.endswith('.jpg') or fname.endswith('.png'):
        img = cv.imread('train/' + fname)
        # img = cv.imread('val/' + fname)
  
        img = cv.resize(img, (300, 330))
        # Our kernel for morphological transformations
        # kernel = np.ones((3, 3), np.uint8)

        # Some images have glare, so use CLAHE to reduce glare
        clahe = CLAHE(img,  clipLimit=1.0)

        # Apply a gaussian blur to reduce noise in the images
        gauss = cv.GaussianBlur(img, (3, 3), 0)
        # gauss = cv.blur(clahe, (3, 3))
 
        # close = cv.morphologyEx(gauss, cv.MORPH_OPEN, (5, 5))
        close = cv.dilate(gauss, (5, 5))      

        # Find the connected components
        stats, thresh = CCL(close) 

        # Find the predicted regions for "numbers"
        # detections = extractNumbers(stats, thresh)
        detections = extractNumbers(stats, thresh)
        
        result = detectNum(detections, knn) 

        print(result)
        # show(img)

   

        # show(mser, fname)