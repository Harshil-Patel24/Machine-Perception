import cv2 as cv
import numpy as np

# https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python

diamond = cv.imread('Images/diamond.png', cv.IMREAD_GRAYSCALE)
dugong = cv.imread('Images/dugong.jpg')

# This was used to remove the green channel as an attempt to separate background and foreground further
# dugong[:,:,1] = np.zeros([dugong.shape[0], dugong.shape[1]])

# We wont gaussian blur the diamond as this is not noisy
diamond_thresh, diamond_binary = cv.threshold(diamond, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# We'll threshold the dugong twice, once as a HSV image, as this seems to isolate the
# dugong from the background really well, and then once to convert it to a gray binary image
dugong = cv.cvtColor(dugong, cv.COLOR_BGR2HSV)
dugong_blur = cv.GaussianBlur(dugong, (5, 5), 0)

# A threshold of 130 works nicely with this image
dugong_thresh, dugong_binary = cv.threshold(dugong_blur, 130, 255, cv.THRESH_BINARY)

# Now we convert back to gray
# dugong_binary = cv.cvtColor(dugong_binary, cv.COLOR_HSV2BGR)
dugong_binary = cv.cvtColor(dugong_binary, cv.COLOR_BGR2GRAY)

# Apply a closing to ensure objects that are very close to eachother form one
kernel = np.ones((3, 3), np.uint8)
dugong_binary = cv.morphologyEx(dugong_binary, cv.MORPH_CLOSE, kernel)

# Final threshold to make the image a black and white binary image
dugong_thresh, dugong_binary = cv.threshold(dugong_binary, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# We will use the built in CCL in opencv with 8 way connectivity
connectivity = 8
num_labels_diamond, labels_diamond, stats_diamond, centroids_diamond = cv.connectedComponentsWithStats(diamond_binary, connectivity, cv.CV_32S)
num_labels_dugong, labels_dugong, stats_dugong, centroids_dugong = cv.connectedComponentsWithStats(dugong_binary, connectivity, cv.CV_32S)

# Map component labels to hue val, 0-179 is the hue range in OpenCV
label_hue_diamond = np.uint8(179 * labels_diamond/np.max(labels_diamond))
label_hue_dugong = np.uint8(179 * labels_dugong/np.max(labels_dugong))

blank_ch_diamond = 255 * np.ones_like(label_hue_diamond)
blank_ch_dugong = 255 * np.ones_like(label_hue_dugong)

# Merges the "colored components" to the blank shape to create the image
labeled_img_diamond = cv.merge([label_hue_diamond, blank_ch_diamond, blank_ch_diamond])
labeled_img_dugong = cv.merge([label_hue_dugong, blank_ch_dugong, blank_ch_dugong])

# Converting to BGR
labeled_img_diamond = cv.cvtColor(labeled_img_diamond, cv.COLOR_HSV2BGR)
labeled_img_dugong = cv.cvtColor(labeled_img_dugong, cv.COLOR_HSV2BGR)

# Set the background to be black
labeled_img_diamond[label_hue_diamond==0] = 0
labeled_img_dugong[label_hue_dugong==0] = 0


# https://www.programcreek.com/python/example/89340/cv2.connectedComponentsWithStats


print("Number of labels for diamond: " + str(num_labels_diamond))
for ii in range(num_labels_diamond):
    area = stats_diamond[ii, cv.CC_STAT_AREA]
    width = stats_diamond[ii, cv.CC_STAT_WIDTH]
    height = stats_diamond[ii, cv.CC_STAT_HEIGHT]

    string = "Label: " + str(ii + 1) + \
             "\n    Area: " + str(area) + \
             "\n    Height: " + str(height) + \
             "\n    Width: " + str(width) + \
             "\n ---------------------------------------- \n"

    print(string)

print("Number of labels for dugong: " + str(num_labels_dugong))
for ii in range(num_labels_dugong):
    area = stats_dugong[ii, cv.CC_STAT_AREA]
    width = stats_dugong[ii, cv.CC_STAT_WIDTH]
    height = stats_dugong[ii, cv.CC_STAT_HEIGHT]

    string = "Label: " + str(ii + 1) + \
             "\n    Area: " + str(area) + \
             "\n    Height: " + str(height) + \
             "\n    Width: " + str(width) + \
             "\n ---------------------------------------- \n"

    print(string)

cv.imshow("Labeled diamond", labeled_img_diamond)
cv.imshow("Labeled dugong", labeled_img_dugong)
cv.waitKey()