import cv2
import numpy

image = cv2.imread('./Lenna.png')

# Step 01, 02
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
boxes = face_cascade.detectMultiScale(image_gray, 1.1, 8)

image = image[
          boxes[0][1]:boxes[0][1] + boxes[0][3],
          boxes[0][0]:boxes[0][0] + boxes[0][2]
          ]
face_image = numpy.copy(image)

# Step 03
image = cv2.Canny(image, 125, 200)

# Step 04
_, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if (stats[labels[x, y], cv2.CC_STAT_WIDTH] <= 10) and (stats[labels[x, y], cv2.CC_STAT_HEIGHT] <= 10):
            image[x, y] = 0

# Step 05
dilatation_kernel = numpy.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=numpy.uint8)
image = cv2.dilate(image, dilatation_kernel)

# Step 06
image = cv2.GaussianBlur(image, (5, 5), 0, 0)
normalized = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Step 07
bilateral = cv2.bilateralFilter(face_image, 5, 40, 40)

# Step 08
sharpness_kernel = numpy.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])
sharpened = cv2.filter2D(face_image, -1, sharpness_kernel)

# Step 09
result = numpy.zeros(face_image.shape, dtype=numpy.uint8)
for x in range(result.shape[0]):
    for y in range(result.shape[1]):
        for channel in range(result.shape[2]):
            result[x, y, channel] = normalized[x, y] * sharpened[x, y, channel] + \
                                    ((1 - normalized[x, y]) * bilateral[x, y, channel])

cv2.imshow('Result', result)
cv2.waitKey(0)
