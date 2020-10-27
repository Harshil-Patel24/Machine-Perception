import cv2 as cv
import numpy as np
import math

def calcHough(image):
    canvas = image.copy()

    if len(image.shape) == 3:
        canny = cv.cvtColor(image, cv.COLOR_BGR2GRAY)    
    
    canny = cv.Canny(image, 50, 200, None, 3)

    lines = cv.HoughLines(canny, 1, np.pi / 180, 200, None, 0, 0)

    if lines is not None:
        for ii in range(0, len(lines)):
            rho = lines[ii][0][0]
            theta = lines[ii][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            cv.line(canvas, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    return canvas

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

# Adapted from https://stackoverflow.com/questions/42721213/python-opencv-extrapolating-the-largest-rectangle-off-of-a-set-of-contour-poin
def findRect(thresh, image, noRect=1, startRect=0):
    if len(thresh.shape) != 2:
        thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    rank_w = []
    rank_h = []
    rank_rect = []

    rank_size = startRect + noRect

    target = image.copy()

    print(len(contours))

    for ii, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        
        if area > 500:
            rect = cv.boundingRect(cnt)
            x, y, w, h = rect

            mask = np.zeros(image.shape, np.uint8)
            mask = image[y:y+h, x:x+w]

            # elif ii < noRect:
            #     for jj in range(noRect):
            #         if rank_w[jj] * rank_h[jj] < w * h:
            #             rank_w.insert(jj, w)
            #             rank_h.insert(jj, h)
            #             rank_rect.insert(jj, rect)
            #         else:
            #             rank_w.append(w)
            #             rank_h.append(h)
            #             rank_rect.append(rect)

# THE PROBLEM IS THAT I NEED TO INSERT IN THE FIRST "NORECT" ITERATIONS REGARDLESS OF SIZE
# THIS SHOULD BE DONE BEFORE STARTING THE FOLLOWING "IF" BLOCK
            # If the rank'd lists aren't full (ie. at size noRect), then we add to them
            # If it's the first item, then just append
            if len(rank_rect) == 0:
                rank_w.append(w)
                rank_h.append(h)
                rank_rect.append(rect)
            # Otherwise we need to check if the area of the rectangles are higher than the previous
            else:
                # Loop through size of rank_size as this is the maximum size of the lists
                for jj in range(rank_size):
                    # If we are still in the first "rank_size" iterations, then we need to add the rectagle
                    # to the list regardless
                    if ii < rank_size:
                        # If area is smaller than current position in ranked list
                        # and there are still entries carry on
                        if (rank_w[jj] * rank_h[jj] > w * h) & (jj < len(rank_rect)):
                            continue
                        # If we are at the end of the list and we still havn't found a bigger rectangle
                        # Then append it to the list
                        elif (rank_w[jj] * rank_h[jj] > w * h) & (jj == len(rank_rect)):
                            rank_w.append(w)
                            rank_h.append(h)
                            rank_rect.append(rect)

                    # General clause of "if current contours rectagle area is larger than the listing at jj"
                    # Insert this before
                    if (jj < len(rank_rect)) & rank_w[jj] * rank_h[jj] < w * h:
                        rank_w.insert(jj, w)
                        rank_h.insert(jj, h)
                        rank_rect.insert(jj, rect)
                         
                # A quick truncate of the list to ensure we only have the largest "noRect" rectangles
                if len(rank_rect) > rank_size:
                    del rank_w[noRect:]
                    del rank_h[noRect:]
                    del rank_rect[noRect:]

    # Drawing the rectangles!
    if len(rank_rect) != 0:
        for ii in range(startRect, rank_size):
            print("II is :" + str(ii))
            print("Length of rectangle list: " + str(len(rank_rect)))
            if len(rank_rect) >= rank_size:
                x, y, w, h = rank_rect[ii]
                target = cv.rectangle(target, (x, y), (x+w, y+h), (0, 255, 0), 1)
            else:
                raise IndexError
        
    return target

def knnSegmentation(image, k=2, block_segments=[]):
    Z_image = image.reshape((-1, 3))
    Z_image = np.float32(Z_image)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, center = cv.kmeans(Z_image, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    labels = labels.flatten()

    segmented_image = center[labels.flatten()]


    for seg in block_segments:
        if seg is not None and type(seg) is int:
            segmented_image[labels == seg] = [0, 0, 0]

    segmented_image = segmented_image.reshape((image.shape))
    return segmented_image

def CCL(image):

    # x, y = image.shape[2:]

    # centre = (x // 2, y // 2)

    detections = []

# ------------------------------------------------------------------------------
    # kernel = np.ones((3, 3), np.uint8)

    # image = cv.GaussianBlur(image, (5, 5), 0)
    # # image = cv.blur(image, (3, 3))
    # # image = cv.medianBlur(image, 5)

    # image = CLAHE(image, clipLimit=1.0)

    image = toGray(image)

    # image = cv.equalizeHist(image)

    # show(image)

    _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # _, thresh = cv.threshold(image, 160, 255, cv.THRESH_BINARY)

    # cv.imshow("", thresh)
    # cv.waitKey()

    # show(thresh)

    thresh = cv.dilate(thresh, kernel, iterations=1)    
    # thresh = cv.erode(thresh, kernel, iterations=1)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
# ------------------------------------------------------------------------------
    # thresh = image.copy()

    connectivity = 8

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S) 

    label_hue = np.uint8(179 * labels/np.max(labels))

    blank = 255 * np.ones_like(label_hue)

    labeled_image = cv.merge([label_hue, blank, blank])

    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

    labeled_image[label_hue==0] = 0

    # print(num_labels)
    count = 0

    selected_stats = []
    sum_areas = 0
    sum_heights = 0
    # tops = []
    # lefts = []
    centres = []
    for ii, stat in enumerate(stats):
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        rect_area = width * height

        # print("Area: " + str(type(rect_area)))
        # print("Width: " + str(type(width)))
        # print("Height: " + str(type(height)))

        left = stat[cv.CC_STAT_LEFT]    
        top = stat[cv.CC_STAT_TOP] 
        fracFG = area / (rect_area)

        if(((fracFG < 0.75) & (fracFG > 0.20)) & (rect_area > 150) & (rect_area < 12000)):
            # print(stat[cv.CC_STAT_HEIGHT] / stat[cv.CC_STAT_WIDTH])
            # First condition is for "Ones" the other is for all other numbers 
            if (abs(height / width - 2.0) < 0.75) or \
                (abs(height / width - 3.0) < 0.75):
                # right = width + left
                # bottom = height + top
                # cv.rectangle(labeled_image, (left, top), (right, bottom), (0, 255, 0))
                # roi = image[(top):(bottom), (left):(right)]
                # detections.append(roi)
                selected_stats.append(stat)
                sum_areas += rect_area
                sum_heights	+= height
                # tops.append(top)
                # lefts.append(left)

                # Centres will have (x, y)
                centres.append(((left + (width // 2)), (top + (height // 2))))

                # print("Area: " + str(rect_area))
                # cv.imshow(str(rect_area + ii), labeled_image[(top):(bottom), (left):(right)])
        
                # show(image[(top - 3):(stat[cv.CC_STAT_TOP] + stat[cv.CC_STAT_HEIGHT] + 3), (left - 3):(stat[cv.CC_STAT_LEFT] + stat[cv.CC_STAT_WIDTH] + 3)])
                # labeled_image = labeled_image[top:bottom, left:right]
                # breaK
                count += 1

    selected_stats = np.asarray(selected_stats)
    # stats_sorted = selected_stats[selected_stats[:,cv.CC_STAT_TOP].argsort(kind='mergesort')]

    centres = np.asarray(centres)
    centres_sorted = centres[centres[:,1].argsort(kind='mergesort')]



    # x, y = centres
    # x = centres_sorted[:,0]
    y = centres_sorted[:,1]

    # x_diff = [abs(x[i] - x[i+1]) for i in range(len(x) - 1)]
    
    # # print(x)
    # # print(y)

    # # x_mean = np.mean(x)
    # y_mean = np.mean(y)

    # # x_diff = abs(x - x_mean)
    # y_diff = abs(y - y_mean)

    # # x_std = np.std(x)
    # y_std = np.std(y)

    # # x_max_var = 10.0
    # y_max_var = 1.0

    y_tol = 0.08
    # if x_std > 50.0:
    #     x_non_outliers = x_diff < x_std * x_max_var
    #     x_fin = x[x_non_outliers]
    # else:
    #     x_fin = x



    # if y_std > 10.0:
    #     y_non_outliers = y_diff < y_std * y_max_var
    #     y_fin = y[y_non_outliers]
    # else:
    #     y_fin = y

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

    # print("X OG: " + str(x))
    print("Y OG: "+ str(y))
    # print("X Final: " + str(x_fin))
    print("Y Final: " + str(y_fin))

    
    # Sort by top
    # selected_stats_sorted_top = selected_stats[selected_stats[:,cv.CC_STAT_TOP].argsort(kind='mergesort')]

    # tops = selected_stats_sorted_top[:,cv.CC_STAT_TOP]

    # diff_tops = [abs(tops[i] - tops[i+1]) if i < len(tops) for i, top in enumerate(tops)]
    # diff_tops = [abs(tops[i] - tops[i+1]) for i in range(len(tops) - 1)]

    # print(diff_tops)

    # tops = np.asarray(tops).astype(np.float32)
    # lefts = np.asarray(lefts).astype(np.float32)

    # med_top = np.median(tops)
    # med_left = np.median(lefts)

    # tops_dist = abs(tops - med_top)
    # lefts_dist = abs(lefts - med_left)

    # tops_std_dist = np.std(tops_dist)
    # lefts_std_dist = np.std(lefts_dist)

    # tops_max_dev = 1.0
    # lefts_max_dev = 5.0

    # print("Tops: " + str(tops))
    # print("Median Tops: " + str(med_top))
    # print("Distance Tops: " + str(tops_dist))
    # print("STD Tops Distances: " + str(tops_std_dist))
    
    # print("Lefts: " + str(lefts))
    # print("Median lefts: " + str(med_left))
    # print("Distance lefts: " + str(lefts_dist))
    # print("STD lefts Distances: " + str(lefts_std_dist))


    # if tops_std_dist > 50.0:
    #     non_outlier_tops = tops_dist < tops_max_dev * tops_std_dist
    #     tops_no_outliers = tops[non_outlier_tops]
    #     # print("Non Outlier Tops: " + str(non_outlier_tops))
    # else:
    #     tops_no_outliers = tops

    # if lefts_std_dist > 10.0:
    #     non_outlier_lefts = lefts_dist < lefts_max_dev * lefts_std_dist
    #     lefts_no_outliers = lefts[non_outlier_lefts]
    #     print("Non Outlier lefts: " + str(non_outlier_lefts))
    # else:
    #     lefts_no_outliers = lefts

    # print("lefts With No Outliers: " + str(lefts_no_outliers))


    # criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # _, label_tops, center_tops = cv.kmeans(tops, 2, None, criteria, 3, cv.KMEANS_RANDOM_CENTERS)
    # _, label_lefts, center_lefts = cv.kmeans(lefts, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # print(center_tops)

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



    # print("Mean: " + str(mean_height))
    # print("Height Thresh: " + str(height_thresh))
# ---------------------------------------------------------------------------------------------------------
    # tops = np.asarray(tops)

    # mean_tops = np.mean(tops)
    # std_tops = np.std(tops)
    # tops_dist_mean = abs(tops - mean_tops)
    # max_dev_tops = 1

    # # print("Tops OG: " + str(tops))
    # # print("Mean: " + str(mean_tops))

    # if std_tops > 10:
    #     non_outlier_tops = tops_dist_mean < max_dev_tops * std_tops
    #     no_outliers_tops = tops[non_outlier_tops]
    # else:
    #     no_outliers_tops = tops

    # lefts = np.asarray(lefts)

    # mean_lefts = np.mean(lefts)
    # std_lefts = np.std(lefts)
    # lefts_dist_mean = abs(lefts - mean_lefts)
    # max_dev_lefts = 1.3

    # print("Lefts OG: " + str(lefts))
    # print("Mean: " + str(mean_lefts))

    # if std_lefts > 10:
    #     non_outlier_lefts = lefts_dist_mean < max_dev_lefts * std_lefts
    #     no_outliers_lefts = lefts[non_outlier_lefts]
    # else:
    #     no_outliers_lefts = lefts

    # print("STD: " + str(std_lefts))
    # print(no_outliers_lefts)
    # # print(selected_stats[cv.CC_STAT_TOP])


    # selected_stats = np.asarray(selected_stats)
    # # selected_stats = selected_stats[cv.CC_STAT_TOP].intersect1d(no_outliers)
    # # selected_stats_indecies = np.intersect1d(selected_stats, no_outliers, return_indices=True)

    # # print(selected_stats_indecies)
# ---------------------------------------------------------------------------------------------------------

    for ii, stat in enumerate(selected_stats):
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        rect_area = width * height
        left = stat[cv.CC_STAT_LEFT]    
        top = stat[cv.CC_STAT_TOP] 
        fracFG = area / (rect_area)
        curr_x = left + (width // 2)
        curr_y = top + (height // 2)

        # print("Area: " + str(type(rect_area)))
        # print("Top: " + str(top))
        # print("Height: " + str(type(height)))

        print("X Current: " + str(curr_x) + " Y Current: "+ str(curr_y))
        print("Accept? " + str(curr_y in y_fin))
        print("Height: " + str(height) + " H_Thresh: " + str(height_thresh))
        print("Area: " + str(rect_area) + " A_Thresh: " + str(area_thresh))

        if (height > height_thresh) & (rect_area > area_thresh) & (curr_y in y_fin): #& (top in tops_no_outliers) & (left in lefts_no_outliers):
            right = width + left
            bottom = height + top
            cv.rectangle(labeled_image, (left, top), (right, bottom), (0, 255, 0))
            roi = thresh[(top-3):(bottom+3), (left-3):(right+3)]
            detections.append(roi)

            # show(roi)
            # print("Area: " + str(rect_area))
            # print("Left: " + str(left))
            # cv.imshow(str(width), labeled_image[(top):(bottom), (left):(right)])

    cv.imshow(str(ii), labeled_image)
    print("Count: " + str(count))
    cv.waitKey()

    return detections

def MSER(image):
    cp = image.copy()
    mser = cv.MSER_create(_min_area=100, _max_area=6000)
    gray = toGray(image)

    # [regions, rects] = mser.detectRegions(gray)
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
            # show(roi)
            
            cv.rectangle(cp, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
    # hulls = [cv.convexHull(np.asarray(p).reshape(-1, 1, 2)) for p in regions]
    # cv.polylines(cp, hulls, 1, (0, 255, 0))
    return cp

def toGray(image):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray
    else:
        return image

# def findRectList(thresh, image, noRect=1, startRect=0):
#     rank_w = [0]
#     rank_h = [0]
#     rank_rect = [None]

#     target = image.copy()

#     contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     print(len(contours))

#     for cnt in contours:
#         area = cv.contourArea(cnt)
#         if area > 400:
#             rect = cv.boundingRect(cnt)
#             x, y, w, h = rect

#             mask = np.zeros(image.shape, np.uint8)
#             mask = image[y:y+h, x:x+w]

#             if len(rank_rect) != 0:
#                 for ii in range(len(rank_rect)):
#                     if rank_w[ii] * rank_h[ii] < w * h:
#                         rank_w.insert(ii, w)
#                         rank_h.insert(ii, h)
#                         rank_rect.insert(ii, rect)
#                     # else:
#                         # rank_w.insert(ii + 1, w)
#                         # rank_h.insert(ii + 1, h)
#                         # rank_rect.insert(ii, rect)
#             else:
#                 rank_w.insert(len(rank_w) - 1, w)
#                 rank_h.insert(len(rank_h) - 1, h)
#                 rank_rect.insert(len(rank_rect) - 1, rect)
        
#     if len(rank_rect) != 0:
#         for ii in range(startRect, noRect + startRect):
#             # print(ii)
#             # print(len(rank_rect))
#             if len(rank_rect) >= noRect + startRect:
#                 x, y, w, h = rank_rect[ii]
#                 cv.rectangle(target, (x, y), (x+w, y+h), (0, 255, 0), 1)
#             else:
#                 raise IndexError
        
#     return target

def show(image, message="", delay=0):
    cv.imshow(message, image)
    cv.waitKey(delay)