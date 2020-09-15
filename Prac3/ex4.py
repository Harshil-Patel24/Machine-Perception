import cv2 as cv
import numpy as np

coins = cv.imread('Images/prac03ex04img01.png', cv.IMREAD_GRAYSCALE)
card = cv.imread('Images/prac03ex04img02.png',cv.IMREAD_GRAYSCALE)
cardpet = cv.imread('Images/prac03ex04img03.png',cv.IMREAD_GRAYSCALE)
car = cv.imread('Images/prac03ex02img01.jpg',cv.IMREAD_GRAYSCALE)

mser = cv.MSER_create()

vis = coins.copy()
regions, _ = mser.detectRegions(coins)
print(type(regions[0]))
hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv.polylines(vis, hulls, 1, (0, 255, 0))
cv.imshow('img', vis)
cv.waitKey(0)
cv.destroyAllWindows()



# cv.imshow("Coins:", mserCoins)
# cv.waitKey(0)





