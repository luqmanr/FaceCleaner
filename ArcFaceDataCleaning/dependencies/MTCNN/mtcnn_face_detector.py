import cv2, math
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

## Initialize mtcnn arguments
detector = MTCNN(min_face_size=50)

def mtcnnCropper(image):    
    ## call mtcnn detect_faces to get the detections
    results = detector.detect_faces(image)
    print('number of faces found=', len(results))
    print('detection results=',results)

    ## face_images is a list, if there are multiple faces detected in one image
    face_images = []
    face_sizes = []
    iterator = 1
    for result in results:
        ## FORMAT DATA result[box] = [left-x, top-y, width, height]
        bbox = result['box']

        left_eye = result['keypoints']['left_eye']
        right_eye = result['keypoints']['right_eye']
        nose = result['keypoints']['nose']

        left_width = abs(left_eye[0] - nose[0])
        left_height = abs(left_eye[1] - nose[1])
        left_degree = math.degrees(math.atan(left_width / left_height))
        print(left_degree)

        right_width = abs(right_eye[0] - nose[0])
        right_height = abs(right_eye[1] - nose[1])
        right_degree = math.degrees(math.atan(right_width / right_height))
        print(right_degree)

        total_degree = (left_degree + right_degree) / 2
        print(total_degree)

        rotation_degree = (right_degree - total_degree)

        bottomY = bbox[1] + bbox[3]
        rightX = bbox[0] + bbox[2]
        topY = bbox[1]
        leftX = bbox[0]

        face_width = abs(leftX - rightX)
        face_height = abs(topY - bottomY)
        face_center = ((face_width / 2), (face_height / 2))

        ## crop the image to the faces only
        image_face = image[
            int(topY):int(bottomY), 
            int(leftX):int(rightX)
            ]
        
        M = cv2.getRotationMatrix2D(face_center, rotation_degree, 1)
        image_face = cv2.warpAffine(image_face, M, (face_width, face_height))

        ## resize image_face to 160x160 pixels
        image_face = cv2.resize(image_face, (112, 112))

        face_images.append(image_face)
        face_sizes.append(face_width*face_height)

    ## returns a list of image(s)
    largest_face_index = face_sizes.index(max(face_sizes))

    face_image = face_images[largest_face_index]
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(face_image, (2,0,1))

    print(aligned)
    return aligned