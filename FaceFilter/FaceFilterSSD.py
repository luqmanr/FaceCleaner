import cv2
from src.FaceDetector import detector

# Read Image
image = cv2.imread('in.jpg')
# print(image.shape)
image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT)
# print(image.shape)

f_detector = detector.SSD()

bounding_boxes, nrof_faces, scores = f_detector.detect_face_bounding_boxes_from(
    image)
print(scores)
if nrof_faces > 0:
    for i in range(nrof_faces):
        box = bounding_boxes[i]
        bottom = box.top_left[0]
        top = box.top_left[0] + box.height
        left = box.top_left[1]
        right = box.top_left[1] + box.width

        if left < 0:
            left = 0

        if top < 0:
            top = 0

        area = abs((top - bottom) * (left - right))

        image = cv2.rectangle(
            image, (left, top),
            (right, bottom),
            (255, 0, 0),
            2)

        # if i == 0:
        #     maxArea = area
        #     maxTop = top
        #     maxBottom = bottom
        #     maxRight = right
        #     maxLeft = left
        # else :
        #     if area > maxArea:
        #         image = cv2.rectangle(image, (maxLeft, maxTop), (maxRight, maxBottom), (255, 0, 0), -1)
        #         area = maxArea
        #         maxArea = area
        #         maxTop = top
        #         maxBottom = bottom
        #         maxRight = right
        #         maxLeft = left
        #     else :
        #         image = cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), -1)

    image = image[20:(image.shape[0] - 20), 20:(image.shape[1] - 20)]
    # print(image.shape)

# #Saving filtered image to new file
cv2.imwrite("out.jpg", image)
