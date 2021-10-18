import cv2
import numpy

image = cv2.imread('./Lenna.png')

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
boxes = face_cascade.detectMultiScale(image_gray, 1.1, 8)

image = image[
          boxes[0][1]:boxes[0][1] + boxes[0][3],
          boxes[0][0]:boxes[0][0] + boxes[0][2]
          ]

image = cv2.Canny(image, 100, 150)

KERNEL = numpy.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=numpy.uint8)
image = cv2.dilate(image, KERNEL)

cv2.imshow('TEST', image)
cv2.waitKey(0)
