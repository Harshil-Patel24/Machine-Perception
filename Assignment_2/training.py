import os
import cv2 as cv
import numpy as np

# Create a list of arrays to store all of the images
# Each index in the list represents a "digit" (ie. contents of one directory)
# Each element in the corresponding array holds a test image from the directory
num_dir = len(os.listdir('digits'))


# # Holds all of the images
# images = [None] * num_dir
# # This will be our training set
# train = [None] * num_dir
# # This will be used to test our model
# test = [None] * num_dir
# # Training and testing labels
# train_labels = [None] * num_dir 
# test_labels = [None] * num_dir


# Holds all of the images
images = np.zeros(num_dir, dtype=object)
# This will be our training set
train = np.zeros(num_dir, dtype=object)
# This will be used to test our model
test = np.zeros(num_dir, dtype=object)
# Training and testing labels
train_labels = np.zeros(num_dir, dtype=object) 
test_labels = np.zeros(num_dir, dtype=object)

# Create training labels
k = np.arange(10)

# Loop through all directories in 'digits/' (ie. all 10 directories for individual digits)
for ii, dname in enumerate(os.listdir('digits')):    
    # Create an array with the same size as the number of images in the directory
    path = 'digits/' + dname
    num_img = len(os.listdir(path)) 
    images[ii] = np.zeros(num_img, dtype=object)

    # Loop through and read in all images
    for jj, fname in enumerate(os.listdir(path)):
        img = cv.imread(path + '/' + fname, cv.IMREAD_GRAYSCALE)
        images[ii][jj] = img
        # print(img.shape)
    # All training images are 40 x 28
    # We will use 80% of the data (the first 80% in each set of digits) as training data
    # train[ii] = images[ii][int(0.8 * jj):].reshape(-1, 1120).astype(np.float32)
    # # # We will use the remaining 20% to test our model
    # test[ii] = images[ii][:int(0.2 * jj)].reshape(-1, 1120).astype(np.float32)
    
    # arr = np.array(images[ii])

    # train[ii] = arr[ii][int(0.8 * jj):].reshape(-1, 1120).astype(np.float32)
    # test[ii] = arr[ii][:int(0.2 * jj)].reshape(-1, 1120).astype(np.float32)
    
    percent = int(0.8 * len(images[ii]))
    # print("Percent " + str(percent))

    # print(len(images[ii][percent:]))
    # print(len(images[ii][:len(images[ii]) - percent - 1]))
    # print(len(images[ii])) 

    train[ii] = images[ii][:percent] #.reshape(-1, 1120).astype(np.float32)
    test[ii] = images[ii][percent:] #.reshape(-1, 1120).astype(np.float32)

    for jj in range(len(train[ii])):
        train[ii][jj] = train[ii][jj].reshape(-1, 1120).astype(np.float32)

    for jj in range(len(test[ii])):
        test[ii][jj] = test[ii][jj].reshape(-1, 1120).astype(np.float32)

    # print(train[ii].shape)
    # print(test[ii].shape)

    # train_labels[ii] = (k[ii] * len(train[ii]))[:, np.newaxis]
    # test_labels[ii] = (k[ii] * len(test[ii]))[:, np.newaxis]

    train_labels[ii] = np.repeat(ii, len(train[ii]))
    test_labels[ii] = np.repeat(ii, len(test[ii]))

    # knn = cv.ml.KNearest_create()
    # knn.train(train[ii], cv.ml.ROW_SAMPLE, train_labels[ii])

    # ret, results, neighbours, dist = knn.findNearest(test[ii], k=5)

    # matches = results==test_labels[ii]
    # correct = np.count_nonzero(matches)
    # accuracy = correct * 100.0 / results.size
    # print("Num correct: " + str(correct))
    # print("Results size: " + str(results.size))
    # print("Accuracy: " + str(accuracy))
    # print(train[ii].shape)

    # np.savez('knn_data.npz', train=train[ii], train_labels=train_labels[ii])

# train = train.reshape(-1, 105)
# test = test.reshape(-1, 27)

# print(train.shape)
# print("Test: " + str(test[0][0].shape))

# train_labels = train_labels[:, np.newaxis]
# test_labels = test_labels[:, np.newaxis]

# knn = cv.ml.KNearest_create()
# knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

# ret, results, neighbours, dist = knn.findNearest(test, k=5)

# matches = results==test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * 100.0 / results.size
# print("Num correct: " + str(correct))
# print("Results size: " + str(results.size))
# print("Accuracy: " + str(accuracy))

# np.savez('knn_data.npz', train=train, train_labels=train_labels)

    # print(images[ii][:int(0.2 * jj)])
    # print(k)










