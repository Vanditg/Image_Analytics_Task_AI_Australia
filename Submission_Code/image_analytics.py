'''
Name: Vandit Gajjar
Task: Image Inalytics 
University: The University of Adelaide 
Submission file for AI Australia Task
File: Main Source File
'''

'''
Importing useful libraries and utils functionality
'''
import sys
import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from utility.datasets import get_labels
from utility.inference import detect_faces
from utility.inference import draw_text
from utility.inference import draw_bounding_box
from utility.inference import apply_offsets
from utility.inference import load_detection_model
from utility.inference import load_image
from utility.preprocessor import preprocess_input

'''
Loading the data and pre-trained models 
'''
image_path = sys.argv[1]
detection_model_path = '../pretrained_models/face_detection/haarcascade_frontalface_default.xml'
emotion_model_path = '../pretrained_models/facial_expression/simpleCNN.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_TRIPLEX

'''
Parameters for shape of the Bounding-Box 
'''
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

'''
Loading the pre-trained models and using model shapes for the inference
'''
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

'''
Reading the images and converting to appropriate color codes for processing 
'''
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

'''
Detecting the face using Haar Cascade model
'''
faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    #print(face_coordinates)
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]

    if emotion_text == emotion_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)
	
print('----------------------------------------------------------------------')
print('Bounding-Box and Facial Experssion Generated')
print('----------------------------------------------------------------------')

'''
Identify images where the face is at least 50-60% of the overall photo. If it’s less than 50%, then score of face quality being too small!  
'''
bounding_box_size = face_coordinates[2] * face_coordinates[3]
image_size = rgb_image.shape[0] * rgb_image.shape[1]

if (bounding_box_size / image_size) * 100 <= 50:
    print('')
    print('')
    print("Face quality being too small!")
    print('')
    print('')
else:
    print('')
    print('')
    print("Face quality is perfect!")
    print('')
    print('')
print('----------------------------------------------------------------------')
print('Face Quality Checkup Completed')
print('----------------------------------------------------------------------')

'''
Facial Expression (Sentiment Analysis) on a Face, saving on local folder
'''
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../samples/output_with_box_expression.png', bgr_image)

'''
Blurring the background from the face.
'''
img = cv2.imread(image_path)
blur_img = cv2.blur(img, (25, 25))
cv2.imwrite('../samples/blur_image.jpg', blur_img)
crop_img = img[face_coordinates[1]:face_coordinates[1]+face_coordinates[2], face_coordinates[0]:face_coordinates[0]+ face_coordinates[3]]
cv2.imwrite('../samples/cropped_image.jpg', crop_img) 

im1 = Image.open('../samples/blur_image.jpg')
im2 = Image.open('../samples/cropped_image.jpg')
back_im = im1.copy()
back_im.paste(im2, (face_coordinates[0], face_coordinates[1]))
back_im.save('../samples/face_deblurred.jpg', quality = 95)

print('----------------------------------------------------------------------')
print('Blurring the Background completion')
print('----------------------------------------------------------------------')