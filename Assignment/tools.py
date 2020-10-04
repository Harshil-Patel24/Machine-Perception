import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

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


# Used from https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/33564950#33564950

def rotate(image, angle):
    diagonal = int(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))
    offset_x = (diagonal - image.shape[0])//2
    offset_y = (diagonal - image.shape[1])//2
    dst_image = np.zeros((diagonal, diagonal, 3), dtype='uint8')
    image_center = (diagonal/2, diagonal/2)

    R = cv.getRotationMatrix2D(image_center, angle, 1.0)
    dst_image[offset_x:(offset_x + image.shape[0]), \
    offset_y:(offset_y + image.shape[1]), \
    :] = image
    dst_image = cv.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv.INTER_LINEAR)

    # Calculate the rotated bounding rect
    x0 = offset_x
    x1 = offset_x + image.shape[0]
    x2 = offset_x
    x3 = offset_x + image.shape[0]

    y0 = offset_y
    y1 = offset_y
    y2 = offset_y + image.shape[1]
    y3 = offset_y + image.shape[1]

    corners = np.zeros((3,4))
    corners[0,0] = x0
    corners[0,1] = x1
    corners[0,2] = x2
    corners[0,3] = x3
    corners[1,0] = y0
    corners[1,1] = y1
    corners[1,2] = y2
    corners[1,3] = y3
    corners[2:] = 1

    c = np.dot(R, corners)

    x = int(c[0,0])
    y = int(c[1,0])
    left = x
    right = x
    up = y
    down = y

    for i in range(4):
        x = int(c[0,i])
        y = int(c[1,i])
        if (x < left): left = x
        if (x > right): right = x
        if (y < up): up = y
        if (y > down): down = y
    h = down - up
    w = right - left

    cropped = np.zeros((w, h, 3), dtype='uint8')
    cropped[:, :, :] = dst_image[left:(left+w), up:(up+h), :]
    return cropped


def displayImages(images):
    rows = 2
    cols = 5
    axes = []
    figure = plt.figure()

    for ii in range(rows * cols):
        axes.append(figure.add_subplot(rows, cols, ii + 1))
        plt.imshow(images[ii])

    axes[0].set_title("Diamond")
    axes[1].set_title("Diamond Scaled")
    axes[2].set_title("Diamond Rotated 30")
    axes[3].set_title("Diamond Rotated 90")
    axes[4].set_title("Diamond Scaled and Rotated")
    axes[5].set_title("Dugong")
    axes[6].set_title("Dugong Scaled")
    axes[7].set_title("Dugong Rotated 30")
    axes[8].set_title("Dugong Rotated 90")
    axes[9].set_title("Dugong Scaled and Rotated")

    figure.tight_layout()
    plt.show()