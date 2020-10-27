import cv2 as cv
from tools import *
import os

# image = cv.imread('train/tr23.jpg', 0)
# image = cv.imread('train/tr22.jpg', 0)

# _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

with np.load('knn_data.npz') as data:
    # print(data.files)
    train = data['train'] 
    train_labels = data['train_labels']

knn = cv.ml.KNearest_create() 
knn.train(train, cv.ml.ROW_SAMPLE, train_labels) 

for ii, fname in enumerate(os.listdir('train')):    
    if fname.endswith('.jpg') or fname.endswith('.png '):
        og_img = cv.imread('train/' + fname)

        og_img = cv.resize(og_img, (300, 350))


        og_img = CLAHE(og_img)
        og_img = cv.convertScaleAbs(og_img, alpha=1.5, beta=10)
        og_img = cv.GaussianBlur(og_img, (5, 5), 0)
        


        # Find the connected components
        stats, thresh = CCL(og_img) 
        show(og_img)

        # Find the predicted regions for "numbers"
        detections = extractNumbers(stats, thresh)
        
        result = detectNum(detections, knn)

        print(result)


    


# cv.imshow("Thresholded", thresh)
# cv.waitKey()



