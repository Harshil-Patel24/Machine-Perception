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


# ------------------------------------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)

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
#---------------------------------TESTING----------------------------------------
    # thresh = cv.erode(thresh, (5, 5), iterations=2)
#---------------------------------TESTING----------------------------------------   

    thresh = cv.dilate(thresh, kernel, iterations=1)    
    # thresh = cv.erode(thresh, kernel, iterations=1)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
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

    return stats, thresh

def extractNumbers(stats, thresh):

    test_img = thresh.copy()


    count = 0
    detections = []
    position = []

    selected_stats = []
    sum_areas = 0
    sum_heights = 0

    # show(thresh)

    centres = []
    for ii, stat in enumerate(stats):
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        rect_area = width * height
        left = stat[cv.CC_STAT_LEFT]    
        top = stat[cv.CC_STAT_TOP] 
        fracFG = area / (rect_area)
        # right = width + left
        # bottom = height + top
        # fracFG = area / (rect_area)

        if(((fracFG < 0.95) & (fracFG > 0.15)) & (rect_area > 200) & (rect_area < 12000)):


            if (abs(height / width - 2.0) < 0.85) or \
                (abs(height / width - 3.5) < 1.2):
                
                selected_stats.append(stat)
                sum_areas += rect_area
                sum_heights	+= height

            #   # -----------------------------TESTING-------------------------------------------
            #     thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
            #     cv.rectangle(thresh, (left, top), (right, bottom), (0, 255, 0))
            #     print(height / width)
            #     show(thresh)
            #     thresh = toGray(thresh)
            #     # -----------------------------TESTING-------------------------------------------`
                # Centres will have (x, y)
                centres.append(((left + (width // 2)), (top + (height // 2))))

                count += 1

    if not len(centres):
        return centres


    # # -----------------------------------------TRYING----------------------------------------------------------
    # selected_stats = np.asarray(selected_stats)
    # stats_sorted =  selected_stats[selected_stats[:,cv.CC_STAT_TOP].argsort(kind='mergesort')]
    # tops = stats_sorted[:,cv.CC_STAT_TOP]
    # areas = stats_sorted[:, cv.CC_STAT_AREA]

    # print("Tops: " + str(tops))
    
    # y_var = np.diff(tops)

    # y_keep = [0] * len(tops)
    # a_keep = [0] * len(areas)
    # # x_keep = []

    # y_tol = 5
    # # x_tol = 40
    # a_tol = (sum_areas / count) * 0.6

    # y_count = 0
    # a_count = 0
    # # incs = []
    
    # for ii in range(len(tops) - 1):
        
    #     if ii > 0:
    #         if (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) >= y_tol):
    #             # print("a")
    #             y_count += 1
    #             # incs.append(ii)
    #             y_keep[ii] = y_count
    #             y_keep[ii + 1] = y_count
    #         elif (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) < y_tol):
    #             # print("b")
    #             y_keep[ii] = y_count
    #             y_keep[ii + 1] = y_count      

    #         if (abs(areas[ii] - areas[ii + 1]) < a_tol) & (abs(areas[ii] - areas[ii - 1]) >= a_tol):
    #             # print("a")
    #             a_count += 1
    #             # incs.append(ii)
    #             a_keep[ii] = a_count
    #             a_keep[ii + 1] = a_count
    #         elif (abs(areas[ii] - areas[ii + 1]) < a_tol) & (abs(areas[ii] - areas[ii - 1]) < a_tol):
    #             # print("b")
    #             a_keep[ii] = a_count
    #             a_keep[ii + 1] = a_count     
    #     else:
    #         if (abs(tops[ii] - tops[ii + 1]) < y_tol):
    #             # print("c")
    #             y_count += 1
    #             y_keep[ii] = y_count
    #             y_keep[ii + 1] = y_count
                
    #         if (abs(areas[ii] - areas[ii + 1]) < a_tol):
    #             # print("c")
    #             a_count += 1
    #             a_keep[ii] = a_count
    #             a_keep[ii + 1] = a_count

        
    # print(y_keep)

    # # -----------------------------------------TRYING----------------------------------------------------------

    # -----------------------------------------TEMPCOMMENT----------------------------------------------------------

    selected_stats = np.asarray(selected_stats)

    centres = np.asarray(centres)
    centres_sorted = centres[centres[:,1].argsort(kind='mergesort')]

    y = centres_sorted[:,1]

    # y_tol = 10
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

    # print("Y OG: "+ str(y))
    # print("Y Final: " + str(y_fin))

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

    # # -----------------------------------------TEMPCOMMENT----------------------------------------------------------


    final_top = math.inf
    final_bottom = 0
    final_right = 0
    final_left = math.inf

    # print("a_tol: " + str(a_tol))
    # print("y_keep: " + str(y_keep))
    # print("a_keep: " + str(a_keep))

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


        # final_det = None

        # print("X Current: " + str(curr_x) + " Y Current: "+ str(curr_y))
        # print("Area Threshold: " + str(area_thresh))
        # print("Rect Area: " + str(rect_area))
        # print("Area: " + str(area))
        # print("Area accept?: " + str(rect_area > area_thresh))

        # print("Accept? " + str((height > height_thresh) & (rect_area > area_thresh) & (curr_y in y_fin)))
        # print("Height: " + str(height) + " H_Thresh: " + str(height_thresh))
        # print("Area: " + str(rect_area) + " A_Thresh: " + str(area_thresh))
        # print("\n-------------------------------------------------------------\n")


        # print(y_keep[ii] != 0)
        if (height > height_thresh) & (rect_area > area_thresh) & (curr_y in y_fin):
        # if (y_keep[ii] != 0) & (a_keep[ii] != 0) & (rect_area > area_thresh):
            # print("HERE")
            # -----------------------------TESTING-------------------------------------------
            # test_img = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
            # cv.rectangle(test_img, (left, top), (right, bottom), (0, 255, 0))
            # show(thresh)
            # thresh = toGray(thresh)

            # print("X Current: " + str(curr_x) + " Y Current: "+ str(curr_y))
            # print("Accept? " + str((height > height_thresh) & (rect_area > area_thresh) & (curr_y in y_fin)))
            # print("Height: " + str(height) + " H_Thresh: " + str(height_thresh))
            # print("Area: " + str(rect_area) + " A_Thresh: " + str(area_thresh))
            # print("\n-------------------------------------------------------------\n")
            # -----------------------------TESTING-------------------------------------------
            
            further_selected.append(stat)
            
            # roi = thresh[(top-3):(bottom+3), (left-3):(right+3)]
            
            # detections.append(roi)
            # position.append(left)

            # if(top < final_top):
            #     print("w")
            #     final_top = top

            # if(bottom > final_bottom):
            #     print("x")
            #     final_bottom = bottom

            # if(left < final_left):
            #     print("y")
            #     final_left = left

            # if(right > final_right):
            #     print("z")
            #     final_right = right


    # print("Top " + str(top))
    # print("Bottom " + str(bottom))
    # print("Left " + str(left))
    # print("Right " + str(right))
    # print("----------------------------")
    # print("Final Top " + str(final_top))
    # print("Final Bottom " + str(final_bottom))
    # print("Final Left " + str(final_left))
    # print("Final Right " + str(final_right))
    # print("----------------------------")

 # -----------------------------------------TRYING----------------------------------------------------------
    
    if len(further_selected) is 0:
        return further_selected, further_selected, [0, 0, 0, 0]

    further_selected = np.asarray(further_selected)
    stats_sorted =  further_selected[further_selected[:,cv.CC_STAT_TOP].argsort(kind='mergesort')]
    tops = stats_sorted[:,cv.CC_STAT_TOP]
    areas = stats_sorted[:, cv.CC_STAT_AREA]

    # print("Tops: " + str(tops))
    
    y_var = np.diff(tops)

    y_keep = [0] * len(tops)
    # a_keep = [0] * len(areas)
    # x_keep = []

    y_tol = 15
    # x_tol = 40
    # a_tol = (sum_areas / count) * 0.4
    # a_tol_ones = 0.3
    # a_tol = 10

    y_count = 0
    # a_count = 0
    # incs = []
    
    if(len(tops) == 1):
        y_keep[0] = 1

    for ii in range(len(tops) - 1):
        
        if ii > 0:
            if (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) >= y_tol):
                # print("a")
                y_count += 1
                # incs.append(ii)
                y_keep[ii] = y_count
                y_keep[ii + 1] = y_count
            elif (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) < y_tol):
                # print("b")
                y_keep[ii] = y_count
                y_keep[ii + 1] = y_count      

            # if ((abs(areas[ii] - areas[ii + 1]) < a_tol) or (abs((max(areas[ii], areas[ii + 1]) / min(areas[ii], areas[ii + 1])) - 2.0) < a_tol_ones)) & ((abs(areas[ii] - areas[ii - 1]) >= a_tol) or (abs((max(areas[ii], areas[ii + 1]) / min(areas[ii], areas[ii + 1])) - 2.0) >= a_tol_ones)):
            #     # print("a")
            #     a_count += 1
            #     # incs.append(ii)
            #     a_keep[ii] = a_count
            #     a_keep[ii + 1] = a_count
            # elif ((abs(areas[ii] - areas[ii + 1]) < a_tol) or (abs((max(areas[ii], areas[ii + 1]) / min(areas[ii], areas[ii + 1])) - 2.0) < a_tol_ones)) & ((abs(areas[ii] - areas[ii - 1]) >= a_tol) or (abs((max(areas[ii], areas[ii + 1]) / min(areas[ii], areas[ii + 1])) - 2.0) < a_tol_ones)):
            #     # print("b")
            #     a_keep[ii] = a_count
            #     a_keep[ii + 1] = a_count     
        else:
            if (abs(tops[ii] - tops[ii + 1]) < y_tol):
                # print("c")
                y_count += 1
                y_keep[ii] = y_count
                y_keep[ii + 1] = y_count
                
            # if (abs(areas[ii] - areas[ii + 1]) < a_tol):
            #     # print("c")
            #     a_count += 1
            #     a_keep[ii] = a_count
            #     a_keep[ii + 1] = a_count

        # print("Area thresh ones at " + str(ii) + ": " + str(abs((max(areas[ii], areas[ii + 1]) / min(areas[ii], areas[ii + 1])) - 2.0)))
        # print("Area thresh at " + str(ii) + ": "+ str(abs(areas[ii] - areas[ii + 1])))
    # print(y_keep)
    # print(a_tol)
    # print(a_keep)

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
    # -----------------------------------------TRYING----------------------------------------------------------
        # print("Top: " + str(top))
        # print("Area: " + str(area))
        # print("Current Y: " + str(curr_y))
        
        if (y_keep[ii] != 0):# & (a_keep[ii] != 0):
            
            roi = thresh[(top-3):(bottom+3), (left-3):(right+3)]
            
            detections.append(roi)
            position.append(left)

            if(top < final_top):
                # print("w")
                final_top = top

            if(bottom > final_bottom):
                # print("x")
                final_bottom = bottom

            if(left < final_left):
                # print("y")
                final_left = left

            if(right > final_right):
                # print("z")
                final_right = right

    # cv.destroyAllWindows()

# final_top = math.inf
# final_bottom = 0
# final_right = 0
# final_left = math.inf

    if (final_top * final_left != math.inf) & (final_bottom * final_right != 0):
        detected_area = thresh[final_top:final_bottom, final_left:final_right]
    
        y = final_top
        x = final_left
        w = final_right - final_left
        h = final_bottom - final_top

        bounding = [x, y, w, h]
    else:
        detected_area = []
        bounding = []
    # show(detected_area, str(ii))
        # else:
        #     # -----------------------------TESTING-------------------------------------------
        #     thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        #     cv.rectangle(thresh, (left, top), (right, bottom), (0, 0, 255))
        #     show(thresh)
        #     thresh = toGray(thresh)
        #     # -----------------------------TESTING-------------------------------------------

    # position = np.ndarray(position)
    # detections = np.ndarray(detections)
    # show(thresh)
    try:
        final_det = [det for _, det in sorted(zip(position, detections))]
    except ValueError:
        # print("ELSE")
        final_det = detections
    
            # final_det = [det for _, det in sorted_det]
    # print(final_det)

    # cv.imshow(str(ii), labeled_image)
    # print("Count: " + str(count))
    # cv.waitKey()

    return final_det, detected_area, bounding

# def contrast(image, contr, bright):
#     # for y in range(image.shape[0]):
#     #     for x in range(image.shape[1]):
#     #         for z in range(image.shape[2]):
#     #             contrasted = np.clip(contr * image[x, y, z] + bright, 0, 255)

#     return cv.convertScaleAbs(image, alpha=contr, beta=bright)

def writeFile(name, contents):
    f = open(name, 'w')
    f.write(contents)
    f.close()

def extractNumbers2(stats, thresh):
    # show(thresh)
    selected_stats = []

    stats_sorted =  stats[stats[:,cv.CC_STAT_TOP].argsort(kind='mergesort')]
    tops = stats_sorted[:,cv.CC_STAT_TOP]

    y_var = np.diff(tops)

    y_keep = [0] * len(tops)
    x_keep = []

    y_tol = 10
    x_tol = 40

    count = 0
    incs = []
    
    for ii in range(len(tops) - 1):
        
        if ii > 0:
            if (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) >= y_tol):
                count += 1
                incs.append(ii)
                y_keep[ii] = count
                y_keep[ii + 1] = count
            elif (abs(tops[ii] - tops[ii + 1]) < y_tol) & (abs(tops[ii] - tops[ii - 1]) < y_tol):
                y_keep[ii] = count
                y_keep[ii + 1] = count                                
        else:
            if (abs(tops[ii] - tops[ii + 1]) < y_tol):
                y_keep[ii] = count
                y_keep[ii + 1] = count
    
    incs.append(len(tops) - 1)

    x_left = stats_sorted[:, cv.CC_STAT_LEFT]
    x_right = x_left + stats_sorted[:, cv.CC_STAT_WIDTH]

    # print(x_left)
    # print(x_right)

    # print(incs)

    # counts = np.zeros(len(set(y_keep)))
    # print(y_keep)

    for ii in range(len(incs) - 1):
        for jj in range(incs[ii], incs[ii + 1]):
            for kk in range(incs[ii], incs[ii + 1]):
                # print(jj)
                if(abs(x_left[jj] - x_right[kk]) < x_tol):
                    if(jj != kk):
                        x_keep.append(jj)
                        x_keep.append(kk)
                        # x_keep.append((jj, kk))

    # print(x_keep)

    x_kept = sorted(set(x_keep))

    # print(x_kept)
    # print(counts)

    selected_stats = stats_sorted[x_kept]

    # print(selected_stats)

    detections = []
    positions = []


    for ii, stat in enumerate(selected_stats):
        left = stat[cv.CC_STAT_LEFT]
        top = stat[cv.CC_STAT_TOP]
        width = stat[cv.CC_STAT_WIDTH]
        height = stat[cv.CC_STAT_HEIGHT]
        area = stat[cv.CC_STAT_AREA]
        right = left + width
        bottom = top + height
        rect_area = height * width
        fracFG = area / (rect_area)
        # print(left)

        # thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        if(((fracFG < 0.95) & (fracFG > 0.15)) & (rect_area > 200) & (rect_area < 12000)):
            if (abs(height / width - 2.0) < 0.75) or \
                (abs(height / width - 3.0) < 0.85):
                cv.rectangle(thresh, (left, top), (right, bottom), (0, 0, 255))

                roi = thresh[(top-3):(bottom+3), (left-3):(right+3)]



                if((roi.shape[0] * roi.shape[1]) != 0):                  
                    # show(roi)
                    detections.append(roi)
                    positions.append(left)
    
    try:
        final_det = [det for _, det in sorted(zip(positions, detections))]
    except ValueError:
        final_det = detections
    return final_det

def detectNum(detections, knn):
    detected = ""
    for det in detections:
        # print(det)
        if det.size:
        # det = np.asarray(det)
            res = cv.resize(det, (28, 40))
            arr = np.array(res)
            test = np.reshape(arr, (-1, 1120)).astype(np.float32)

            ret, result, neighbours, dist = knn.findNearest(test, k=2)

            # print(result)

            result = int(result.ravel()[0])

            detected += str(result)
    # if detected == '':
    #     detected = None
    # else:
    #     detected = int(detected)

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