import cv2 as cv
import numpy as np


# Referenced from https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html

# Here we will make the histograms

def histogramCalc(image):
    bgr_planes = cv.split(image)

    # no. bins
    hist_size = 256

    hist_range = (0, 256)

    accumulate = False

    b_hist = cv.calcHist(bgr_planes, [0], None, [hist_size], hist_range, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [hist_size], hist_range, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [hist_size], hist_range, accumulate=accumulate)

    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / hist_size))

    hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for ii in range(1, hist_size):
        cv.line(hist_image, (bin_w * (ii - 1), hist_h - int(round(b_hist[ii - 1][0]))),
                (bin_w * (ii), hist_h - int(round(b_hist[ii][0]))), (255, 0, 0), thickness=2)
        cv.line(hist_image, (bin_w * (ii - 1), hist_h - int(round(g_hist[ii - 1][0]))),
                (bin_w * (ii), hist_h - int(round(g_hist[ii][0]))), (0, 255, 0), thickness=2)
        cv.line(hist_image, (bin_w * (ii - 1), hist_h - int(round(r_hist[ii - 1][0]))),
                (bin_w * (ii), hist_h - int(round(r_hist[ii][0]))), (0, 0, 255), thickness=2)

    return hist_image

# Referenced from https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html

def cornerHarris(image):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = image.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    return img

# Referenced from https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html

def siftKeyPoints(image):
    img = image.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv.drawKeypoints(gray, kp, img)

    return img