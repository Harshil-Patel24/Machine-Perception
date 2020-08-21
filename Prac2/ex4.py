import cv2 as cv
import numpy as np

img = cv.imread('Images/prac02ex04img01.png')
if img is None:
	print('Could not read image')
	exit(0)

# Brightness and contrast -- g(i,j) = alpha * f(i,j) + beta

new_img = np.zeros(img.shape, img.dtype)

alpha = 1.0 #Simple contrast
beta = 0 #Simple brightness


# Initialise values
print(' Basic Linear Transforms ')
print('-------------------------')
try:
	alpha = float(input('* Enter the alpha value (contrast) [1.0 - 3.0]: '))
	beta = int(input('* Enter the beta value (brightness) [0 - 100]: ' ))
except ValueError:
	print('Error, not a number')

# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)

for y in range(img.shape[0]):
	for x in range(img.shape[1]):
		for c in range(img.shape[2]):
			new_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)


cv.imshow('Original Image', img)
cv.imshow('New Image', new_img)

cv.waitKey()

