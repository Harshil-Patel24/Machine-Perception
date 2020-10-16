import cv2 as cv
import os
import numpy as np
from processing import *
from tools import *

def main():
	
    training_data = []

# Our process
# So far what has worked - 
# Read in an image - gray
# Morph close it
# Gaussian Blur it
# Adaptive threshold
# Find contours
# Find the second largest rectangle and use it
# (Largest rectagle is usually the outline of the image)

# What to try next
# Use edge detection instead of thresholding

	# Read in our training dataset
    for ii, fname in enumerate(os.listdir('train')):    
        if fname.endswith('.jpg') or fname.endswith('.png'):
            og_img = cv.imread('train/' + fname)

            img = CLAHE(og_img, 2.0, (5, 5))

            og_img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
            thresh = cv.GaussianBlur(og_img, (5, 5), 0)

            # thresh = knnSegmentation(thresh, k=4, block_segments=[])
            _, thresh = cv.threshold(thresh, 170, 255, cv.THRESH_BINARY_INV)
            # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, (8, 8))
            # thresh = cv.dilate(thresh, (1, 1), iterations=5)


            # rects = findRect(thresh, og_img, noRect=2)

            # show(rects, fname)

            # thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

            # contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # max_w = 0
            # max_h = 0
            # sec_w = 0
            # sec_h = 0
            # second_rect = None
            # largest_rect = None
            # house_num = og_img.copy()
            # for cnt in contours:
            #     area = cv.contourArea(cnt)
            #     if area < 400:
            #         cv.drawContours(thresh, [cnt], 0, (255, 255, 255), -1)
            #     else: 
            #         rect = cv.boundingRect(cnt)
            #         x, y, w, h = rect

            #         mask = np.zeros(og_img.shape, np.uint8)
            #         mask[y:y+h, x:x+w] = og_img[y:y+h, x:x+w]
                
            #         if max_w * max_h < w * h:
            #             sec_w = max_w
            #             sec_h = max_h
            #             max_w = w
            #             max_h = h
            #             second_rect = largest_rect
            #             largest_rect = rect
            #         elif sec_w * sec_h < w * h:
            #             sec_w = w
            #             sec_h = h
            #             second_rect = rect

            # if second_rect is not None:
            #     x, y, w, h = second_rect
            # cv.rectangle(house_num, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # # cv.imshow(fname, house_num)
            cv.imshow("Thresh: " + fname, thresh)

            # # https://stackoverflow.com/questions/42721213/python-opencv-extrapolating-the-largest-rectangle-off-of-a-set-of-contour-poin
            # max_w = 0
            # max_h = 0
            # # sec_w = 0
            # # sec_h = 0
            # # second_rect = None
            # largest_rect = None
            # canvas = img.copy()
            # for cnt in contours:
            #     rect = cv.boundingRect(cnt)
            #     x, y, w, h = rect

            #     mask = np.zeros(img.shape, np.uint8)
            #     mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]

            #     if max_w * max_h < w * h:
            #         # sec_w = max_w
            #         # sec_h = max_h
            #         max_w = w
            #         max_h = h
            #         # second_rect = largest_rect
            #         largest_rect = rect
            #     # elif sec_w * sec_h < w * h:
            #     #     sec_w = w
            #     #     sec_h = h
            #     #     second_rect = rect

            # x, y, w, h = largest_rect # second_rect
            # cv.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # cv.imshow(fname, canvas)
            # cv.imshow("Edges: " + fname, thresh)
            # cv.waitKey()
            # cv.destroyAllWindows()

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()