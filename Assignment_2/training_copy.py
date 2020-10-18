import os
import cv2 as cv
import numpy as np

# Create a list of arrays to store all of the images
# Each index in the list represents a "digit" (ie. contents of one directory)
# Each element in the corresponding array holds a test image from the directory
num_dir = len(os.listdir('digits'))

m = -1 
num_images = 0
percentages = np.zeros(10, dtype=int)
dir_images = np.zeros(10, dtype=int)
cum_dir = np.zeros(10, dtype=int)
training = []
testing = []
training_labels = []
testing_labels = []

for root, dirs, files in os.walk('digits'):
    if m < 0:
        m += 1
        continue
    if m > 9:
        break
    num_images += len(files)
    dir_images[m] = len(files)
    cum_dir[m] = num_images
    m += 1

images = np.zeros(num_images, dtype=object)
m = 0

cum_dir_i = np.insert(cum_dir, 0, 0)

percentages = [int(0.8 * dir_images[j]) for j in range(10)]

for ii, dname in enumerate(os.listdir('digits')):
    path = 'digits/' + dname
    
    for jj, fname in enumerate(os.listdir(path)):
        img = cv.imread(path + '/' + fname, cv.IMREAD_GRAYSCALE)
m)
        images[m] = img

        if(m in range(cum_dir_i[ii], cum_dir_i[ii] + percentages[ii])):
            training.append(img)
            training_labels.append(ii)
        else:
            testing.append(img)
            testing_labels.append(ii)
        m += 1
    
training = np.asarray(training).reshape(-1, 1120).astype(np.float32)
testing = np.asarray(testing).reshape(-1, 1120).astype(np.float32)
training_labels = np.asarray(training_labels).astype(np.int64)
testing_labels = np.asarray(testing_labels).astype(np.int64)

knn = cv.ml.KNearest_create()
knn.train(training, cv.ml.ROW_SAMPLE, training_labels)

ret, results, neighbours, dist = knn.findNearest(testing, k=5)

results = results.flatten().astype(np.int64)

matches = (results==testing_labels)
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / results.size

print("Num correct: " + str(correct))
print("Results size: " + str(results.size))
print("Accuracy: " + str(accuracy))