import cv2 as cv
import sys

# img = cv.imread(cv.samples.findFile("Images/prac01ex01img01.png"))
img = cv.imread("Images/prac01ex01img01.png")


if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("prac01ex01img01.png", img)



