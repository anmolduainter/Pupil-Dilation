import cv2
import numpy as np
from imutils import face_utils
import dlib
import os
import time
import math

def load_pretrained(path_predictor):
    print ("Loading Pretrained!!!")
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_predictor)
    return face_detector,predictor

def getEyeLandmarksIndexes():
    (lstart,lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart,rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    return ((lstart,lend),(rstart,rend))

def detectFaces(img,face_detector,verbose=False,gray=False):
    if (not gray):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray_img,0)
    if (verbose):
        print ("Total Number of Faces Found :: " + str(len(rects)))
    return rects

def euclidian_dst(pt1,pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1] - pt2[1])**2)

def getEyeKeypointsDistance(coords):
    horizontal_1 = euclidian_dst(coords[0],coords[3])
    vertical_1 = euclidian_dst(coords[1],coords[5])
    vertical_2 = euclidian_dst(coords[2],coords[4])
    return horizontal_1,(vertical_1,vertical_2)

def crop_eye_helper(img,coords,new_coords=False):
    x_min = min([i[0] for i in coords]) - 15
    y_min = min([i[1] for i in coords]) - 15
    x_max = max([i[0] for i in coords]) + 15
    y_max = max([i[1] for i in coords]) + 15
    height = y_max - y_min 
    width = x_max - x_min 
    crop_img = img[y_min:y_min + height , x_min:x_min+width]
    if(new_coords):
        new_c = []
        for coord in coords:
            new_c.append((coord[0] - x_min , coord[1] - y_min))
        return crop_img,np.array(new_c)    
    return crop_img,None

def crop_dlib_rectangle(img,coords):
    x = coords.left()
    y = coords.top()
    w = coords.right() - coords.left()
    h = coords.bottom() - coords.top()    
    crop_img = img[y:y+h , x:x+w]
    return crop_img

def crop_eyes(img,left_e_coords,right_e_coords,new_coords=False):
    left_eye_crop,left_eye_new_coords = crop_eye_helper(img,left_e_coords,new_coords=True)
    right_eye_crop,right_eye_new_coords = crop_eye_helper(img,right_e_coords,new_coords=True)
    return left_eye_crop,right_eye_crop,left_eye_new_coords,right_eye_new_coords   

def eyes_coordinates(img,face_rects,key_points_predictor,l_e,r_e,face_verbose=False,gray=False):
    if (gray):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rect = face_rects
    if (rect is None or len(rect) == 0):
        rect = [(0,0),(img.shape[1],img.shape[0])]
        rect = dlib.rectangle(rect[0][0],rect[0][1],rect[1][0],rect[1][1])
    else:
        rect = face_rects[0] # TODO
    if(face_verbose):
        cropped_face = crop_dlib_rectangle(img,rect)
        cv2.imwrite("./face.jpg",cropped_face)
    shape = key_points_predictor(gray_img,rect)
    shape = face_utils.shape_to_np(shape)
    left_e_coords = shape[l_e[0]:l_e[1]]
    right_e_coords = shape[r_e[0]:r_e[1]]
    return left_e_coords,right_e_coords

def eye_aspect_ratio(eye):
	A = euclidian_dst(eye[1], eye[5])
	B = euclidian_dst(eye[2], eye[4])
	C = euclidian_dst(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def check_blink(left_e_coords,right_e_coords):
    leftEAR = eye_aspect_ratio(left_e_coords)
    rightEAR = eye_aspect_ratio(right_e_coords)
    ear = (leftEAR+rightEAR)/2
    print (ear)
    if (ear < 0.30):
        return True
    else:
        return False    


def eye_detect_using_haar(img,xml_path):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    value_img = hsv_img[:,:,2]
    eye_cascade = cv2.CascadeClassifier('./utils/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(value_img) 
    return eyes[0],eyes[1]