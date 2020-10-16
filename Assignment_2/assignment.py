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
            img = cv.imread('train/' + fname)
            
            # lab = cv.cvtColor(og_img, cv.COLOR_BGR2LAB)
            # clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
            # planes = cv.split(lab)
            # planes[0] = clahe.apply(planes[0])
            # lab = cv.merge(planes)

            # img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
            # img = CLAHE(og_img, 2.0, (5, 5))
            # cv.imshow(fname, bgr)
            # cv.waitKey()

            # img = og_img[:,:,:] * 2
            # show(img, fname) 
            # img = og_img

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (5, 5), 0)
            gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, (10, 10))

            gray = cv.equalizeHist(gray)
            gray = cv.morphologyEx(gray, cv.MORPH_OPEN, (5, 5))
            # gray = cv.morphologyEx(gray, cv.MORPH_DILATE, (30, 30), iterations=20)
            thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 5)
    
            training_data.append(thresh)

            # rectangles = findRect(thresh, img, noRect=2)

            # show(rectangles, "Rectangles of " + fname)
            # cv.imshow("Rectangles: " + fname, rectangles)
            # cv.waitKey()

            contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            max_w = 0
            max_h = 0
            sec_w = 0
            sec_h = 0
            second_rect = None
            largest_rect = None
            house_num = img.copy()
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < 400:
                    cv.drawContours(thresh, [cnt], 0, (255, 255, 255), -1)
                else: 
                    rect = cv.boundingRect(cnt)
                    x, y, w, h = rect

                    mask = np.zeros(img.shape, np.uint8)
                    mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
                
                    if max_w * max_h < w * h:
                        sec_w = max_w
                        sec_h = max_h
                        max_w = w
                        max_h = h
                        second_rect = largest_rect
                        largest_rect = rect
                    elif sec_w * sec_h < w * h:
                        sec_w = w
                        sec_h = h
                        second_rect = rect

            x, y, w, h = second_rect
            cv.rectangle(house_num, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv.imshow(fname, house_num)
            cv.imshow("Thresh: " + fname, thresh)
            cv.waitKey()
            cv.destroyAllWindows()

            # # https://stackoverflow.com/questions/42721213/python-opencv-extrapolating-the-largest-rectangle-off-of-a-set-of-contour-poin
            # max_w = 0
            # max_h = 0
            # sec_w = 0
            # sec_h = 0
            # second_rect = None
            # largest_rect = None
            # canvas = img.copy()
            # for cnt in contours:
            #     rect = cv.boundingRect(cnt)
            #     x, y, w, h = rect

            #     mask = np.zeros(img.shape, np.uint8)
            #     mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]

            #     if max_w * max_h < w * h:
            #         sec_w = max_w
            #         sec_h = max_h
            #         max_w = w
            #         max_h = h
            #         second_rect = largest_rect
            #         largest_rect = rect
            #     elif sec_w * sec_h < w * h:
            #         sec_w = w
            #         sec_h = h
            #         second_rect = rect

            # x, y, w, h = second_rect # largest_rect # 
            # cv.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # cv.imshow(fname, canvas)
            # cv.imshow("Edges: " + fname, thresh)
            # cv.waitKey()
            # cv.destroyAllWindows()

    # cv.waitKey()

if __name__ == "__main__":
    main()