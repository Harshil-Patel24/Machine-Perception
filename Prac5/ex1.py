import cv2 as cv
import numpy as np

GRID_SIZE = 20

img = cv.imread('Images/digits.png')

height = img.shape[0]
width = img.shape[1]

num_rows = int(height / GRID_SIZE)
num_cols = int(width / GRID_SIZE)

# num_sections = int((width * height) / (GRID_SIZE ** 2))

print("Number of Rows: " + str(num_rows))
print("Number of Columns: " + str(num_cols))
# print("Number of Sections: " + str(num_sections))

digits = [[0 for x in range(0, num_cols)] for y in range(0, num_rows)]
digits_flat = digits.copy()
labels = seeds_flat = seeds = [None]*num_rows


for ii in range(0, num_rows, 5):
    for jj in range(0, num_cols):
        # digits[ii][jj] = img[int(ii * height / num_rows):int(ii * height / num_rows + height / num_rows),
        #                  int(jj * height / num_cols):int(jj * width / num_cols + width / num_cols)]

        digits[int(ii / 5)][jj] = img[ii*GRID_SIZE:ii*GRID_SIZE + GRID_SIZE, jj*GRID_SIZE:jj*GRID_SIZE + GRID_SIZE]

        digits_flat[int(ii / 5)][jj] = np.asarray(digits[int(ii / 5)][jj]).flatten()

    seeds[int(ii / 5)] = np.mean(digits[int(ii / 5)], axis=0).astype(np.uint8)
    seeds_flat[ii] = np.float32(np.asarray(seeds[int(ii / 5)]).flatten())

    cv.imshow("Average for: " + str(int(ii / 5)), seeds[int(ii / 5)])

    labels[ii] = ii
cv.waitKey(0)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
compactness, labels_k, centers = cv.kmeans(seeds_flat, 4, None, criteria, 10, flags)

    # cv.imshow("Seed number: " + str(int(ii / 5)), seeds[int(ii / 5)])
    # cv.waitKey()
# for ii in range(0, len(digits)):
#     for jj in range(0, len(digits[ii])):
#         cv.imshow("Digit: " + str(ii) + " " + str(jj), digits[ii][jj])
#         cv.waitKey()