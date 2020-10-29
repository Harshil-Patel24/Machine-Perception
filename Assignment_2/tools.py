import cv2 as cv
import numpy as np
import math

bin_n = 16

def CLAHE(image, clipLimit=1.0, tileGridSize=(5, 5)):
    img = image.copy()
    if len(image.shape) != 3:
        img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
 
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    planes = cv.split(lab)
    planes[0] = clahe.apply(planes[0])
    lab = cv.merge(planes)

    img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return img

def CCL(image):
# ------------------------------------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)

    image = toGray(image)

    _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    thresh = cv.dilate(thresh, kernel, iterations=1)    

    connectivity = 8

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S) 

    label_hue = np.uint8(179 * labels/np.max(labels))

    blank = 255 * np.ones_like(label_hue)

    labeled_image = cv.merge([label_hue, blank, blank])

    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

    labeled_image[label_hue==0] = 0

    return stats, thresh

def extractNumbers(stats, thresh, img):

    test_img = thresh.copy()


    count = 0
    detections = []
    position = []

    selected_stats = []
    sum_areas = 0
    sum_heights = 0

    centres = []
    for ii, stat in enumerate(stats):
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        rect_area = width * height
        left = stat[cv.CC_STAT_LEFT]    
        top = stat[cv.CC_STAT_TOP] 
        fracFG = area / (rect_area)

        if(((fracFG < 0.95) & (fracFG > 0.15)) & (rect_area > 200) & (rect_area < 12000)):


            if (abs(height / width - 2.0) < 0.85) or \
                (abs(height / width - 3.5) < 1.2):
                
                selected_stats.append(stat)
                sum_areas += rect_area
                sum_heights	+= height

                centres.append(((left + (width // 2)), (top + (height // 2))))

                count += 1

    if not len(centres):
        return centres

    selected_stats = np.asarray(selected_stats)

    centres = np.asarray(centres)
    centres_sorted = centres[centres[:,1].argsort(kind='mergesort')]

    y = centres_sorted[:,1]

    y_tol = 0.5

    y_fin = []

    for n in range(len(y) - 1):
        if abs(y[n] - y[n + 1]) < y[n] * y_tol:
            if y[n] in y_fin:
                y_fin.append(y[n + 1])
            else:
                y_fin.append(y[n])  
                y_fin.append(y[n + 1])

    if(len(y) <= 2):
        y_fin = y

    y_fin = np.asarray(y_fin)

    if count != 0:       
        mean_area = sum_areas / count
        area_thresh = 0.3 * mean_area
        
        mean_height = sum_heights / count
        height_thresh = 0.8 * mean_height
    else:
        mean_area = 0
        area_thresh = 0

        mean_height = 0
        height_thresh = 0

    final_top = math.inf
    final_bottom = 0
    final_right = 0
    final_left = math.inf

    further_selected = []

    for ii, stat in enumerate(selected_stats):
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        rect_area = width * height
        left = stat[cv.CC_STAT_LEFT]    
        top = stat[cv.CC_STAT_TOP] 
        right = width + left
        bottom = height + top
        fracFG = area / (rect_area)
        curr_x = left + (width // 2)
        curr_y = top + (height // 2)

        if (height > height_thresh) & (rect_area > area_thresh) & (curr_y in y_fin):
                  
            further_selected.append(stat)
    
    if len(further_selected) == 0:
        return further_selected, further_selected, [0, 0, 0, 0]

    further_selected = np.asarray(further_selected)
    stats_sorted =  further_selected[further_selected[:,cv.CC_STAT_TOP].argsort(kind='mergesort')]
    tops = stats_sorted[:,cv.CC_STAT_TOP]
    areas = stats_sorted[:, cv.CC_STAT_AREA]

    y_var = np.diff(tops)

    y_keep = [0] * len(tops)

    y_tol = 15

    y_count = 0
    
    if(len(tops) == 1):
        y_keep[0] = 1

    for ii in range(len(tops) - 1):
        
        if ii > 0:
            if (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) >= y_tol):
                y_count += 1
                y_keep[ii] = y_count
                y_keep[ii + 1] = y_count
            elif (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) < y_tol):
                y_keep[ii] = y_count
                y_keep[ii + 1] = y_count      
        else:
            if (abs(tops[ii] - tops[ii + 1]) < y_tol):
                y_count += 1
                y_keep[ii] = y_count
                y_keep[ii + 1] = y_count
 
    for ii, stat in enumerate(further_selected):
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        rect_area = width * height
        left = stat[cv.CC_STAT_LEFT]    
        top = stat[cv.CC_STAT_TOP] 
        right = width + left
        bottom = height + top
        fracFG = area / (rect_area)
        curr_x = left + (width // 2)
        curr_y = top + (height // 2)
        
        if (y_keep[ii] != 0):
            
            roi = img[(top-3):(bottom+3), (left-3):(right+3)]
            
            detections.append(roi)
            position.append(left)

            if(top < final_top):
                final_top = top

            if(bottom > final_bottom):
                final_bottom = bottom

            if(left < final_left):
                final_left = left

            if(right > final_right):
                final_right = right

    if (final_top * final_left != math.inf) & (final_bottom * final_right != 0):
        detected_area = img[final_top:final_bottom, final_left:final_right]
    
        y = final_top
        x = final_left
        w = final_right - final_left
        h = final_bottom - final_top

        bounding = [x, y, w, h]
    else:
        detected_area = []
        bounding = []
    try:
        final_det = [det for _, det in sorted(zip(position, detections))]
    except ValueError:
        final_det = detections
    
    return final_det, detected_area, bounding

def writeFile(name, contents):
    f = open(name, 'w')
    f.write(contents)
    f.close()

def hog(img):
    img = cv.resize(img, (64, 128))

    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9

    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hogged = hog.compute(img)

    return hogged

def trainSVM():
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm = svm.load('svm_data.dat')
    return svm


def detectNum(detections, svm):

    detected = ""

    for det in detections:
        if det.size:

            test = toGray(det)
        
            test = np.array(hog(test)).astype(np.float32).reshape(1, -1)

            result = svm.predict(test)

            result = str(result[1].ravel()[0].astype(np.uint8))

            detected += result

    return detected

def trainKNN():
    with np.load('knn_data.npz') as data:
        train = data['train'] 
        train_labels = data['train_labels']

    knn = cv.ml.KNearest_create() 
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels) 

    return knn


def MSER(image):
    cp = image.copy()
    mser = cv.MSER_create(_min_area=100, _max_area=6000)
    gray = toGray(image)

    regions, _ = mser.detectRegions(gray)

    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)

        height = ymax - ymin
        width = xmax - xmin

        if((abs(height / width - 2.0) < 0.75) or (abs(height / width - 3.0) < 0.75)):
            print(height * width)
            roi = cp[ymin:ymax, xmin:xmax]
            roi = toGray(roi)
            _, roi= cv.threshold(roi, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            
            cv.rectangle(cp, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)

    return cp

def toGray(image):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray
    else:
        return image

def show(image, message="", delay=0):
    cv.imshow(message, image)
    cv.waitKey(delay)