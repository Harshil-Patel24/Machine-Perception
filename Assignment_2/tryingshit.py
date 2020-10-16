import cv2 as cv
from tools import *
import os

# image = cv.imread('train/tr23.jpg', 0)
# image = cv.imread('train/tr22.jpg', 0)

# _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

for ii, fname in enumerate(os.listdir('train')):    
    if fname.endswith('.jpg') or fname.endswith('.png'):
        og_img = cv.imread('train/' + fname, 0)

        ccl = CCL(og_img)
        show(ccl)

    


# cv.imshow("Thresholded", thresh)
# cv.waitKey()



