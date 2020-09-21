import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

img = cv.imread('Images/prac04ex02img01.png', cv.IMREAD_GRAYSCALE)

th,otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

otsu = cv.bitwise_not(otsu)

# cv.imshow("Binarised", otsu)
# cv.waitKey()

connectivity = 8
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(otsu, connectivity, cv.CV_32S)

# Map component labels to hue val, 0-179 is the hue range in OpenCV
label_hue = np.uint8(179 * labels/np.max(labels))

# Makes a white image the shape of the original
blank_ch = 255 * np.ones_like(label_hue)
# Merges the "colored components" to the blank shape to create the image
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

# Converting to BGR
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

# Set bg to black
labeled_img[label_hue==0] = 0

# for ii, blob in enumerate(labels):
for ii in range(1, num_labels):
    area = stats[ii, cv.CC_STAT_AREA]
    width = stats[ii, cv.CC_STAT_WIDTH]
    height = stats[ii, cv.CC_STAT_HEIGHT]
    fracFG = area / (width * height)
    distFGX = stats[ii, cv.CC_STAT_LEFT]
    distFGY = stats[ii, cv.CC_STAT_TOP]

    string = "Label: " + str(ii) + \
             "\n    Area: " + str(area) + \
             "\n    Height: " + str(height) + \
             "\n    Fraction of Foreground Pixels: %.3f" + \
             "\n    Distribution of FG Pixels in X: " + str(distFGX) + \
             "\n    Distribution of FG Pixels in Y: " + str(distFGY) + \
             "\n ---------------------------------------- \n" % fracFG



    # Take the positions of the blobs so we can label them
    # x = label_hue[ii][0]
    # print("X: " + str(x))
    # y = max(0, label_hue[ii][1] - 10)
    # # y = label_hue[ii][1]
    # print("Y: " + str(y))
    # cv.putText(labeled_img, str(ii), (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    #
    print(string)




cv.imshow("CCL -- Full Image:", labeled_img)
cv.waitKey()








