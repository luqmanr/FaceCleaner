import cv2
import tensorflow as tf
import math  
from FaceDetector.mtcnn import detect_face
from src.FaceDetector import detector

##  MTCNN Config  ##
##################
# Minimum size of face / pixel
minsize = 1
# Three steps threshold
threshold = [0.6, 0.7, 0.7]
# Scale factor
factor = 0.709
margin = 44
image_size = 1

# load MTCNN model
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess)

border = 20

# Read image
image = cv2.imread('in.jpg')

# Add Border
image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT) 

# Detect Face SSD
f_detector = detector.SSD()
bounding_boxes, nrof_faces, scores = f_detector.detect_face_bounding_boxes_from(image)
list_boxSSD = []
if nrof_faces > 0:
    for i in range(nrof_faces):
        box = bounding_boxes[i]
        top = box.top_left[0] - border 
        bottom = box.top_left[0] + box.height - border
        right = box.top_left[1] - border
        left = box.top_left[1] + box.width - border

        x = int((right - left) / 2) + left
        y = int((bottom - top) / 2) + top

        if left < 0: 
             left = 0

        if top < 0: 
            top = 0
      
        boxSSD = [left, top, right, bottom, x, y]
        list_boxSSD.append(boxSSD)

# Remove Border
image = image[border:(image.shape[0]-border), border:(image.shape[1]-border)]

# Detect Face MTCNN
bounding_boxes, keypoints = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]
list_boxMTCNN = []
if nrof_faces > 0:
    for i in range(nrof_faces):
        box = bounding_boxes[i]
        right = int(box[0])
        top = int(box[1])
        left = int(box[2])
        bottom = int(box[3])

        x = int((right - left) / 2) + left
        y = int((bottom - top) / 2) + top
        
        if left < 0: 
             left = 0

        if top < 0: 
            top = 0
        
                
        boxMTCNN = [left, top, right, bottom, x, y]
        list_boxMTCNN.append(boxMTCNN)
        # Assume only get one face from MTCNN
        # image = cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        

# Draw Stroke Rectangle
# for box in list_boxSSD:
#     image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

# for box in list_boxMTCNN:
#     image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

# Find face MTCNN in faces SSD with nearest location
# Assume only get one face drom MTCNN
# image = cv2.circle(image, (list_boxMTCNN[0][4], list_boxMTCNN[0][5]), 10, (255, 255, 0), 2) 

i = 0
for box in list_boxSSD:
    # image = cv2.circle(image, (box[4], box[5]), 10, (255, 0, 0), 2)
    distance = math.sqrt((list_boxMTCNN[0][4] - box[4]) ** 2 + (list_boxMTCNN[0][5] - box[5]) ** 2)
    if (i == 0):
        nearest_distance = distance
        nearest = box
    else:
        if(distance < nearest_distance):
            nearest_distance = distance
            nearest = box
    i += 1
    
# image = cv2.circle(image, (nearest[0], nearest[1]), 10, (50, 50, 50), 2)

# Draw Bold Rectangle
for box in list_boxSSD:
    
    if (box != nearest):
        # check overlap
        overlap = []    
        overlap.append(-(list_boxMTCNN[0][0] - box[0]))
        overlap.append(list_boxMTCNN[0][1] - box[1])
        overlap.append(list_boxMTCNN[0][2] - box[2])
        overlap.append(-(list_boxMTCNN[0][3]- box[3]))
        print(overlap)
        print(box)
        
        if (min(overlap) < 0):
            minpos = overlap.index(min(overlap))
            box[minpos] = list_boxMTCNN[0][(minpos + 2) % 4]
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)

# Write Image
cv2.imwrite("out.jpg", image)
