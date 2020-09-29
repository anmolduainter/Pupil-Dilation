import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from fractions import Fraction

def historgramEqualization(img,clahe=False):
    if (clahe):
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        equ_img = clahe.apply(img)
    else:    
        equ_img = cv2.equalizeHist(img)
    return equ_img


def sharpenImage(img):
    image = Image.fromarray(img)
    sharpened = image.filter(ImageFilter.SHARPEN)
    sharpened = sharpened.filter(ImageFilter.SHARPEN)
    return np.asarray(sharpened)

def gray_blurred(img,blur_l,gray=False,blur="Median",Lab=False):
    if(gray):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if (Lab):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)[:,:,1]

    if (blur == "Median"):
        return cv2.medianBlur(img,blur_l) 
    elif (blur == "gaussian"):
        return cv2.GaussianBlur(img,blur_l,0)   

def draw_detection(img,x,y,r,color=(0,0,255)):
    cv2.circle(img,(int(x),int(y)),int(r),color,1)
    return img

def SaveImage(img,name):
    cv2.imwrite(name,img)

def getParamsHougheCircle(detections):
    try:
        detections = np.uint16(np.around(detections))
        x,y,r = detections[0,0]
        return x,y,r
    except:
        print ("No Circles Found!")    
        return None,None,None

def detect_iris(cropped_l_e_img,cropped_r_e_img,gray=False,draw_and_save=False,face_type=None,key_point_info=None,new_coords=None,pupil_center=None):

    if (face_type is None):
        print ("Please provind type of face!!")
    elif (face_type == 3):
        img = gray_blurred(cropped_l_e_img,15,Lab=True)
        blurred_img = historgramEqualization(img,clahe=False)
        blurred_img = sharpenImage(blurred_img)
        _,threshold_l = cv2.threshold(blurred_img,11,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(threshold_l,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        center_l,radius_l = cv2.minEnclosingCircle(cnt)
        return ((center_l,radius_l))

    elif(face_type==2):
        # # Iris - good
        # blurred_img_l = gray_blurred(cropped_l_e_img,21,gray)
        # blurred_img_r = gray_blurred(cropped_r_e_img,21,gray)

        # # IRIS GOOD
        # blurred_img_l = gray_blurred(cropped_l_e_img,27,gray)
        # blurred_img_r = gray_blurred(cropped_r_e_img,27,gray)

        blurred_img_l = gray_blurred(cropped_l_e_img,15,Lab=True)
        blurred_img_r = gray_blurred(cropped_r_e_img,15,Lab=True)

        blurred_img_l = historgramEqualization(blurred_img_l,clahe=False)
        blurred_img_r = historgramEqualization(blurred_img_r,clahe=False)

        blurred_img_l = sharpenImage(blurred_img_l)
        blurred_img_r = sharpenImage(blurred_img_r)
        
        # Left Eye
        _,threshold_l = cv2.threshold(blurred_img_l,9,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(threshold_l,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        center_l,radius_l = cv2.minEnclosingCircle(cnt)

        # Right Eye
        _,threshold_r = cv2.threshold(blurred_img_r,9,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(threshold_r,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        center_r,radius_r = cv2.minEnclosingCircle(cnt)

        return ((center_l,radius_l),(center_r,radius_r))

    elif(face_type==1):
        blur_levels = [9,11,13,15,17]
        max_l_r,max_l = -1,None
        max_r_r,max_r = -1,None
        for blur_l in blur_levels:
            blurred_img_l = gray_blurred(cropped_l_e_img,blur_l,gray)
            blurred_img_r = gray_blurred(cropped_r_e_img,blur_l,gray)

            blurred_img_l = historgramEqualization(blurred_img_l,clahe=False)
            blurred_img_r = historgramEqualization(blurred_img_r,clahe=False)

            iris_l_e_circle = cv2.HoughCircles(blurred_img_l, cv2.HOUGH_GRADIENT, 1, cropped_l_e_img.shape[0], param1=50, param2=20, minRadius=0, maxRadius=48)
            iris_r_e_circle = cv2.HoughCircles(blurred_img_r, cv2.HOUGH_GRADIENT, 1, cropped_r_e_img.shape[0], param1=50, param2=20, minRadius=0, maxRadius=48)

            _,_,iris_l_r = getParamsHougheCircle(iris_l_e_circle)
            _,_,iris_r_r = getParamsHougheCircle(iris_r_e_circle)

            if (iris_l_r is not None and iris_l_r > max_l_r):
                max_l_r = iris_l_r
                max_l = iris_l_e_circle
            if (iris_r_r is not None and iris_r_r > max_r_r):
                max_r_r = iris_r_r
                max_r = iris_r_e_circle

        l_x,l_y,l_r = getParamsHougheCircle(iris_l_e_circle)
        r_x,r_y,r_r = getParamsHougheCircle(iris_r_e_circle)
        final_radius = int((l_r+r_r)/2)

        if(key_point_info is not None):
            l_e_info = key_point_info[0]
            l_mid_y = (l_e_info[1][0] + l_e_info[1][1])/4
            r_e_info = key_point_info[1]
            r_mid_y = (r_e_info[1][0] + r_e_info[1][1])/4
            l_x= int(((new_coords[0][1][0]+new_coords[0][5][0])/2) + ((new_coords[0][2][0]+new_coords[0][4][0])/2))//2
            l_y= int(((new_coords[0][1][1]+new_coords[0][5][1])/2) + ((new_coords[0][2][1]+new_coords[0][4][1])/2))//2
            r_x = int(((new_coords[1][1][0]+new_coords[1][5][0])/2) + ((new_coords[1][2][0]+new_coords[1][4][0])/2))//2
            r_y = int(((new_coords[1][1][1]+new_coords[1][5][1])/2) + ((new_coords[1][2][1]+new_coords[1][4][1])/2))//2

            l_x = int(pupil_center[0][0])
            l_y = int(pupil_center[0][1])
            r_x = int(pupil_center[1][0])
            r_y = int(pupil_center[1][1])


        return (((l_x,l_y),final_radius) , ((r_x,r_y),final_radius))


def detect_pupil(cropped_l_e_img,cropped_r_e_img,gray=False,face_type=None):

    all_images_arr = []

    if (face_type is None):
        print ("Face Type cannot be none!")

    elif(face_type==3):
        all_images_arr.append(cropped_l_e_img)
        img = gray_blurred(cropped_l_e_img,19,gray=True)
        all_images_arr.append(img)
        blurred_img = historgramEqualization(img,clahe=False)
        all_images_arr.append(blurred_img)
        # Left Eye
        _,threshold_l = cv2.threshold(blurred_img,5,255,cv2.THRESH_BINARY_INV)
        all_images_arr.append(threshold_l)
        contours,_ = cv2.findContours(threshold_l,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        center_l,radius_l = cv2.minEnclosingCircle(cnt)
        return ((center_l,radius_l)),all_images_arr

    elif(face_type==2 or face_type==1):    
        cropped_l_e_img = cv2.cvtColor(cropped_l_e_img , cv2.COLOR_BGR2GRAY)
        cropped_r_e_img = cv2.cvtColor(cropped_r_e_img , cv2.COLOR_BGR2GRAY)
        blurred_img_l_g = gray_blurred(cropped_l_e_img,19,gray=False)
        blurred_img_r_g = gray_blurred(cropped_r_e_img,19,gray=False)

        blurred_img_l = historgramEqualization(blurred_img_l_g,clahe=False)
        blurred_img_r = historgramEqualization(blurred_img_r_g,clahe=False)

        # Left Eye
        _,threshold_l = cv2.threshold(blurred_img_l,5,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(threshold_l,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        center_l,radius_l = cv2.minEnclosingCircle(cnt)

        # Right Eye
        _,threshold_r = cv2.threshold(blurred_img_r,5,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(threshold_r,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        center_r,radius_r = cv2.minEnclosingCircle(cnt)

        all_images_arr.append(cropped_l_e_img)
        all_images_arr.append(blurred_img_l_g)
        all_images_arr.append(blurred_img_l)
        all_images_arr.append(threshold_l)

        all_images_arr.append(cropped_r_e_img)
        all_images_arr.append(blurred_img_r_g)
        all_images_arr.append(blurred_img_r)
        all_images_arr.append(threshold_r)
        return ((center_l,radius_l),(center_r,radius_r)),all_images_arr

