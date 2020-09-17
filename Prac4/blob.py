import cv2 as cv

FOREGROUND = 255
BACKGROUND = 0

def ccl(img):
    for ii, pixel in img:
        # checks if pixel is foreground or background (0 is background - 255 is foreground)
        if pixel != FOREGROUND:
            pass
        dimensions = img.shape
        # height = dimensions[0]
        width = dimensions[1]

        left = img[ii - 1]
        above = img[ii - width]

        if FOREGROUND in {left, above}:
            pass









