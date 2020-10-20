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
        og_img = cv.imread('train/' + fname, 0)
 
        # cv.imshow(fname, og_img)
        detections = CCL(og_img)

        # for det in detections:
        #     show(det)

        for det in detections:
            res = cv.resize(det, (28, 40))
            arr = np.array(res)
            test = np.reshape(arr, (-1, 1120)).astype(np.float32)

            ret, result, neighbours, dist = knn.findNearest(test, k=1)
            cv.destroyAllWindows()
            print("--------------\n")
        #     cv.namedWindow(str(dist), cv.WINDOW_NORMAL)
        #     cv.resizeWindow(str(dist), 500, 200)
        #     cv.imshow(str(dist), det)
        # cv.waitKey()
        # cv.destroyAllWindows()
            # print(int(result))
        # show(ccl) 

        # show(detections)



    


# cv.imshow("Thresholded", thresh)
# cv.waitKey()



