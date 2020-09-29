import eye 
import pupilMeasurement as pm
import cv2
import os
import shutil
import seaborn as sns
import time
import numpy as np
import helper as h
import draw_helper as dh
from tqdm import tqdm
import imageio

Face_Type = 1
Video = True
path = "./video1.mp4"

if (Face_Type == 3):
    if (not os.path.exists("./face-type-3-res/")):
        os.mkdir("./face-type-3-res/")
    eye_image,all_images = h.Image_Functionality(path,face_type=3)
    final_image = dh.FusedImage(all_images,face_type=3)
    pm.SaveImage(eye_image,"./face-type-3-res/final_eye.jpg")
    pm.SaveImage(final_image,"./face-type-3-res/final.jpg")

elif (Face_Type==2):
    if (not os.path.exists("./face-type-2-res/")):
        os.mkdir("./face-type-2-res/")
    left_eye_image,right_eye_image,all_images = h.Image_Functionality(path,face_type=2)
    final_image = dh.FusedImage(all_images,face_type=2)
    pm.SaveImage(left_eye_image,"./face-type-2-res/left_final.jpg")
    pm.SaveImage(right_eye_image,"./face-type-2-res/right_final.jpg")
    pm.SaveImage(final_image,"./face-type-2-res/final.jpg")

elif (Face_Type==1):
    if(Video):
        cap = cv2.VideoCapture(path)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('./output.avi',fourcc, 20.0, (1280,640))
        writer = imageio.get_writer('test.mp4', fps=35)
        iris_mvg = (0,0)
        beta = 0.95
        current_frame = 0
        dilation_ratios = []
        while(cap.isOpened()):
            ret,img = cap.read()
            if (ret is not False):
                left_eye_image,right_eye_image,iris_mvg,dilation_ratios,all_images = h.Video_Functionality(img,iris_mvg,beta,current_frame,dilation_ratios)
                # left_eye_image = cv2.resize(left_eye_image,(right_eye_image.shape[1],right_eye_image.shape[0]))
                # combine_img = np.concatenate((right_eye_image,left_eye_image),axis=1)
                final_image = dh.FusedImage(all_images,face_type=1)
                writer.append_data(cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB))
                cv2.imshow('frame',cv2.resize(final_image,(1280,640)))
                current_frame = current_frame+1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break  

    else:
        if (not os.path.exists("./face-type-1-res/")):
            os.mkdir("./face-type-1-res/")
        left_eye_image,right_eye_image,all_images = h.Image_Functionality(path,face_type=1)
        pm.SaveImage(left_eye_image,"./face-type-1-res/left_final.jpg")
        pm.SaveImage(right_eye_image,"./face-type-1-res/right_final.jpg")
        final_image = dh.FusedImage(all_images,face_type=1)
        pm.SaveImage(final_image,"./face-type-1-res/final.jpg")

