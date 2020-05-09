import cv2, os
from MTCNN.mtcnn_face_detector import mtcnnCropper

image = cv2.imread('player3.jpg')

face_images = mtcnnCropper(image)

for index, images in enumerate(face_images):
    cv2.imwrite('{}test.jpg'.format(str(index)), images)