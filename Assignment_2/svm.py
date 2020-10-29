import cv2 as cv
import numpy as np
import os
from tools import *

# training = []
labels = []
hogs = []

for ii, dname in enumerate(os.listdir('digits')):
    path = 'digits/' + dname

    for jj, fname in enumerate(os.listdir(path)):
        img = cv.imread(path + '/' + fname, cv.IMREAD_GRAYSCALE)

        # training.append(img)
        labels.append(ii)

        hogs.append(hog(img))



# for tr in training:
#     hogs.append(hog(tr))

hogs = np.array(hogs).astype(np.float32)

responses = np.array(labels)

svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.trainAuto(hogs, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

