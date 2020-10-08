import cv2 as cv
import os
from processing import *

def main():
	
	training = []

	# Read in our training dataset
	for fname in os.listdir('train'):
		if fname.endswith('.jpg') or fname.endswith('.png'):
			training.append(cv.imread('train/' + fname))
	
	
	
if __name__ == "__main__":
    main()
