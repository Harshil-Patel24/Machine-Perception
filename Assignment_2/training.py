import os
import cv2 as cv
import numpy as np

# Create a list of arrays to store all of the images
# Each index in the list represents a "digit" (ie. contents of one directory)
# Each element in the corresponding array holds a test image from the directory
num_dir = len(os.listdir('digits'))

# Holds all of the images
images = [None] * num_dir
# This will be our training set
train = [None] * num_dir
# This will be used to test our model
test = [None] * num_dir

# Loop through all directories in 'digits/' (ie. all 10 directories for individual digits)
for ii, dname in enumerate(os.listdir('digits')):    
    # Create an array with the same size as the number of images in the directory
    path = 'digits/' + dname
    num_img = len(os.listdir(path)) 
    images[ii] = np.zeros(num_img, dtype=object)

    # Loop through and read in all images
    for jj, fname in enumerate(os.listdir(path)):
        img = cv.imread(path + '/' + fname, 0)
        images[ii][jj] = img
    
    # We will use 80% of the data (the first 80% in each set of digits) as training data
    train[ii] = images[ii][int(0.8 * jj):]
    # We will use the remaining 20% to test our model
    test[ii] = images[ii][:int(0.2 * jj)]








