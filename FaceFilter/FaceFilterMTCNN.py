import cv2
from FaceDetector.mtcnn import detect_face
import tensorflow as tf


##MTCNN Config  ##
##################
# Minimum size of face / pixel
minsize = 1
# Three steps threshold
threshold = [0.6, 0.7, 0.7]
# Scale factor
factor = 0.709
margin = 44
image_size = 1


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess)

#Read Image
image = cv2.imread('in.jpg')

bounding_boxes, keypoints = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]
print(nrof_faces)
if nrof_faces > 0:
    for i in range(nrof_faces):
        box = bounding_boxes[i]
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        
        if int(left) < 0: 
             left = 0

        if int(top) < 0: 
            top = 0
        
        area = (int(top) - int(bottom)) * (int(left) - int(right))
        image = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        if i == 0:
            maxArea = area
            maxTop = top
            maxBottom = bottom
            maxRight = right
            maxLeft = left
        else :
            if area > maxArea:
                image = cv2.rectangle(image, (int(maxLeft), int(maxTop)), (int(maxRight), int(maxBottom)), (255, 0, 0), -1)
                area = maxArea
                maxArea = area
                maxTop = top
                maxBottom = bottom
                maxRight = right
                maxLeft = left
            else :
                image = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), -1)

        # image_face = image[int(top):int(bottom), int(left):int(right)]
        # image_face = cv2.resize(image_face, (160, 160))
        # filename = "out_" + str(i) + ".jpg"
        # cv2.imwrite(filename, image_face)
    
#Saving filtered image to new file
cv2.imwrite("out.jpg", image)