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
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
 
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
    planes = cv.split(lab)
    planes[0] = clahe.apply(planes[0])
    lab = cv.merge(planes)

    img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return img

# Adapted from https://stackoverflow.com/questions/42721213/python-opencv-extrapolating-the-largest-rectangle-off-of-a-set-of-contour-poin
def findRect(thresh, image, noRect=1, startRect=0):
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    rank_w = []
    rank_h = []
    rank_rect = []
    

    target = image.copy()


    print(len(contours))

    for ii, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        
        if area > 400:
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
                # Loop through size of noRect as this is the maximum size of the lists
                for jj in range(noRect):
                    # If we are still in the first "noRect" iterations, then we need to add the rectagle
                    # to the list regardless
                    if ii < noRect:
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
                if len(rank_rect) > noRect:
                    del rank_w[noRect:]
                    del rank_h[noRect:]
                    del rank_rect[noRect:]

    # Drawing the rectangles!
    if len(rank_rect) != 0:
        for ii in range(startRect, noRect + startRect):
            # print("II is :" + str(ii))
            # print("Length of rectangle list: " + str(len(rank_rect)))
            if len(rank_rect) >= noRect + startRect:
                x, y, w, h = rank_rect[ii]
                target = cv.rectangle(target, (x, y), (x+w, y+h), (0, 255, 0), 1)
            else:
                raise IndexError
        
    return target

def findRectList(thresh, image, noRect=1, startRect=0):
    rank_w = [0]
    rank_h = [0]
    rank_rect = [None]

    target = image.copy()

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print(len(contours))

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 400:
            rect = cv.boundingRect(cnt)
            x, y, w, h = rect

            mask = np.zeros(image.shape, np.uint8)
            mask = image[y:y+h, x:x+w]

            if len(rank_rect) != 0:
                for ii in range(len(rank_rect)):
                    if rank_w[ii] * rank_h[ii] < w * h:
                        rank_w.insert(ii, w)
                        rank_h.insert(ii, h)
                        rank_rect.insert(ii, rect)
                    # else:
                        # rank_w.insert(ii + 1, w)
                        # rank_h.insert(ii + 1, h)
                        # rank_rect.insert(ii, rect)
            else:
                rank_w.insert(len(rank_w) - 1, w)
                rank_h.insert(len(rank_h) - 1, h)
                rank_rect.insert(len(rank_rect) - 1, rect)
        
    if len(rank_rect) != 0:
        for ii in range(startRect, noRect + startRect):
            # print(ii)
            # print(len(rank_rect))
            if len(rank_rect) >= noRect + startRect:
                x, y, w, h = rank_rect[ii]
                cv.rectangle(target, (x, y), (x+w, y+h), (0, 255, 0), 1)
            else:
                raise IndexError
        
    return target

def show(image, message="", delay=0):
    cv.imshow(message, image)
    cv.waitKey(delay)