import cv2 as cv
import numpy as np
import os
from tools import *

labels = []
hogs = []
# Iterate through all training data
for ii, dname in enumerate(os.listdir('/home/student/train')):
	path = '/home/student/train/' + dname
	print(dname)
	for jj, fname in enumerate(os.listdir(path)):
		print(fname)
		img = cv.imread(path + '/' + fname, cv.IMREAD_GRAYSCALE)
		# Create the training labels
		labels.append(int(dname))
        # Create the training dataset
		hogs.append(hog(img))


hogs = np.array(hogs).astype(np.float32)

responses = np.array(labels)

# Create, train and save SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.trainAuto(hogs, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

