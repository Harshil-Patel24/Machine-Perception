import cv2 as cv
import numpy as np
import os
import sys
from tools import *
from matplotlib import pyplot as plt

def main(argv):

    if len(argv) > 1:
        print('usage: python3 assignment.py <image-directory>')
        exit(0)
    elif len(argv) == 1:
        directory = argv[0]
    else:
        directory = 'train'

    print(directory)

    knn = trainKNN()

    for ii, fname in enumerate(os.listdir(directory)):    
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img = cv.imread(directory + '/' + fname)
            # img = cv.imread('val/' + fname)
    
            img = cv.resize(img, (300, 330))
            
            # Some images have glare, so use CLAHE to reduce glare
            clahe = CLAHE(img,  clipLimit=0.3)



            # Apply a gaussian blur to reduce noise in the images
            gauss = cv.GaussianBlur(clahe, (3, 3), 0)
    
            close = cv.dilate(gauss, (5, 5))      

            # Find the connected components
            stats, thresh = CCL(close) 

            # Find the predicted regions for "numbers"
            detections, det_area, bounding = extractNumbers(stats, thresh)
            
            if(len(detections) != 0):

                result = detectNum(detections, knn) 

                imName = fname[2:-4]

                det_area_name = directory + '-DetectedArea' + imName + '.jpg'
                bounding_box_name = directory + '-BoundingBox' + imName + '.txt'
                house_name = directory + '-House' + imName + '.txt'

                bounding_box = 'X: ' + str(bounding[0]) + '\nY: ' + str(bounding[1]) +\
                    '\nW: ' + str(bounding[2]) + '\nH: ' + str(bounding[3])

                print(imName)

                cv.imwrite('output/' + det_area_name, det_area)
                writeFile('output/' + bounding_box_name, bounding_box)
                writeFile('output/' + house_name, result)
            else:
                print('No numbers detected')

if __name__ == '__main__':
    main(sys.argv[1:])
