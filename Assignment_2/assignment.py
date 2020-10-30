import cv2 as cv
import numpy as np
import os
import sys
from tools import *
from matplotlib import pyplot as plt

def main(argv):
    # Controller to allow for different input directories 
    if len(argv) > 1:
        print('usage: python3 assignment.py <image-directory>')
        exit(0)
    elif len(argv) == 1:
        directory = argv[0]
    else:
        directory = '/home/student/test'

    # Train our SVM model
    svm = trainSVM()

    # Iterate through directory and find all images
    for ii, fname in enumerate(os.listdir(directory)):    
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img = cv.imread(directory + '/' + fname)
    
            img = cv.resize(img, (300, 330))
            
            # Some images have glare, so use CLAHE to reduce glare
            clahe = CLAHE(img,  clipLimit=0.4, tileGridSize=(3,3))

            # Apply a gaussian blur to reduce noise in the images
            gauss = cv.GaussianBlur(clahe, (5, 7), 0)
    
            close = cv.dilate(gauss, (5, 5))      

            # Find the connected components
            stats, thresh = CCL(close) 

            # Find the predicted regions for "numbers"
            detections, det_area, bounding = extractNumbers(stats, thresh, img)
            
            # Detect images if there are detections
            if(len(detections) != 0):

                result = detectNum(detections, svm) 
                # This just takes the number from the file name
                imName = fname[2:-4]    

                det_area_name = 'DetectedArea' + imName + '.jpg'
                bounding_box_name = 'BoundingBox' + imName + '.txt'
                house_name = 'House' + imName + '.txt'

                bounding_box = 'X: ' + str(bounding[0]) + '\nY: ' + str(bounding[1]) +\
                    '\nW: ' + str(bounding[2]) + '\nH: ' + str(bounding[3])
                # Write results to file
                cv.imwrite('/home/student/output/' + det_area_name, det_area)
                writeFile('/home/student/output/' + bounding_box_name, bounding_box)
                writeFile('/home/student/output/' + house_name, result)
            else:
                print('No numbers detected')

if __name__ == '__main__':
    main(sys.argv[1:])
