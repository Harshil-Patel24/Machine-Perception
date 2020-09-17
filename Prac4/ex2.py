import numpy as np
import cv2 as cv

img = cv.imread('Images/prac04ex02img01.png', cv.IMREAD_GRAYSCALE)

th,otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

otsu = cv.bitwise_not(otsu)

# cv.imshow("Binarised", otsu)
# cv.waitKey()

num_labels, labels = cv.connectedComponents(otsu)

# Map component labels to hue val, 0-179 is the hue range in OpenCV
label_hue = np.uint8(179 * labels/np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

# Converting cvt to BGR
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

cv.imshow("CCL", labeled_img)
cv.waitKey()








