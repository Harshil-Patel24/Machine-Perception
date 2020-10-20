import cv2 as cv
import numpy as np
import os
from tools import *
from matplotlib import pyplot as plt

for ii, fname in enumerate(os.listdir('train')):    
    if fname.endswith('.jpg') or fname.endswith('.png '):
        img = cv.imread('train/' + fname)

        mser = MSER(img)

        # show(mser, fname)