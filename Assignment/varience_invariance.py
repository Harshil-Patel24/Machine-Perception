import cv2 as cv
import numpy as np

diamond = cv.imread('Images/diamond.png')
dugong = cv.imread('Images/dugong.jpg')

imgs = [diamond, dugong]
scaled = rows = cols = translations = img_translations = M = rotated = [None] * 2

# ---------------- SCALE AND ROTATE ----------------

# Lets create a scaled version
scaled_diamond = cv.resize(diamond, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
rows_diamond, cols_diamond = scaled_diamond.shape[:2]

scaled_dugong = cv.resize(dugong, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
rows_dugong, cols_dugong = scaled_dugong.shape[:2]

# # Now to rotate
# translation_matrix_diamond = np.float32([[1, 0, int(0.5*cols_diamond)], [0, 1, int(0.5*rows_diamond)]])
# translation_matrix_dugong = np.float32([[1, 0, int(0.5*cols_dugong)], [0, 1, int(0.5*rows_dugong)]])
#
# # Create a bigger area so the rotation doesn't truncate the corners
# img_translation_diamond = cv.warpAffine(diamond, translation_matrix_diamond, (cols_diamond*2, rows_diamond*2))
# img_translation_dugong = cv.warpAffine(dugong, translation_matrix_dugong, (cols_dugong*2, rows_dugong*2))
# img_translation_scaled_diamond = cv.warpAffine(scaled_diamond, translation_matrix_diamond, (cols_diamond*2, rows_diamond*2))
# img_translation_scaled_dugong = cv.warpAffine(scaled_dugong, translation_matrix_dugong, (cols_dugong*2, rows_dugong*2))
#
# # Do the rotatey stuff
# M_diamond = cv.getRotationMatrix2D((cols_diamond, rows_diamond), 30, 1)
# rotated_scaled_diamond = cv.warpAffine(img_translation_diamond, M_diamond, (cols_diamond * 2, rows_diamond * 2))
# M_dugong = cv.getRotationMatrix2D((cols_dugong, rows_dugong), 30, 1)
# rotated_scaled_dugong = cv.warpAffine(img_translation_dugong, M_dugong, (cols_dugong * 2, rows_dugong * 2))

# We'll use a 90 degree rotation to check if there is any invariance in histograms
# We wont use a "non square" rotation as this would introduce black corners, or truncate edges
rotated_diamond = cv.rotate(diamond, cv.ROTATE_90_CLOCKWISE)
rotated_dugong = cv.rotate(dugong, cv.ROTATE_90_CLOCKWISE)

rotated_scaled_diamond = cv.rotate(scaled_diamond, cv.ROTATE_90_CLOCKWISE)
rotated_scaled_dugong = cv.rotate(scaled_dugong, cv.ROTATE_90_CLOCKWISE)

# cv.imshow("Diamond", diamond)
# cv.imshow("Dugong", dugong)
# cv.imshow("Scaled diamond", scaled_diamond)
# cv.imshow("Scaled dugong", scaled_dugong)
# cv.imshow("Scaled rotated diamond", rotated_scaled_diamond)
# cv.imshow("Scaled rotated dugong", rotated_scaled_dugong)
# cv.imshow("Rotated 90 diamond", rotated_diamond)
# cv.imshow("Rotated 90 dugong", rotated_dugong)
# cv.waitKey()

# ---------------- Histograms ----------------

# Now we use these to check the variance/invariance of certain key-point detections
# We now have 8 images:
# diamond [0], dugong [1]
# scaled_diamond [2], scaled_dugong [3]
# rotated_diamond [4], rotated_dugong [5]
# rotated_scaled_diamond [6], rotated_scaled_dugong [7]

images = [diamond, dugong, scaled_diamond, scaled_dugong, rotated_diamond, rotated_dugong,
          rotated_scaled_diamond, rotated_scaled_dugong]

# hist = rgb_images = [None] * 8

# for ii, image in enumerate(images):
#     rgb_images[ii] = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     hist[ii] = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist[ii] = cv.normalize(hist[ii], hist[ii]).flatten()








