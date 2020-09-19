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
for ii in range(0, num_rows, 5):
    for jj in range(0, num_cols):
        # digits[ii][jj] = img[int(ii * height / num_rows):int(ii * height / num_rows + height / num_rows),
        #                  int(jj * height / num_cols):int(jj * width / num_cols + width / num_cols)]

        digits[int(ii / 5)][jj] = img[ii*GRID_SIZE:ii*GRID_SIZE + GRID_SIZE, jj*GRID_SIZE:jj*GRID_SIZE + GRID_SIZE]
        digits_flat[int(ii / 5)][jj] = np.asarray(digits[int(ii / 5)][jj]).flatten()


# for ii in range(0, len(digits)):
#     for jj in range(0, len(digits[ii])):
#         cv.imshow("Digit: " + str(ii) + " " + str(jj), digits[ii][jj])
#         cv.waitKey()