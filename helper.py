import eye 
import pupilMeasurement as pm
import cv2

f_d,k_p_d = eye.load_pretrained("./utils/shape_predictor_68_face_landmarks.dat")
left_e,right_e = eye.getEyeLandmarksIndexes()

def Image_Functionality(img_path,face_type = None):
    all_images = []
    img = cv2.imread(img_path)
    all_images.append(img)
    if (face_type == 1):
        face_rects = eye.detectFaces(img,f_d,verbose=False)
        left_e_coords,right_e_coords = eye.eyes_coordinates(img.copy(),face_rects,k_p_d,left_e,right_e,gray=True)
        left_e_img,right_e_img,left_e_coords,right_e_coords= eye.crop_eyes(img.copy(),left_e_coords,right_e_coords,new_coords=True)
        
        left_e_ki = eye.getEyeKeypointsDistance(left_e_coords)
        right_e_ki = eye.getEyeKeypointsDistance(right_e_coords)

        (pupil_l,pupil_r),all_pupil_images_arr = pm.detect_pupil(left_e_img.copy(),right_e_img.copy(),gray=True,face_type=1)
        iris_l,iris_r = pm.detect_iris(left_e_img.copy(),right_e_img.copy(),
                                        gray=True,
                                        key_point_info=(left_e_ki,right_e_ki),
                                        new_coords=(left_e_coords,right_e_coords),
                                        pupil_center=(pupil_l[0],pupil_r[0]),
                                        face_type=1)

        all_images += all_pupil_images_arr

        left_eye_image = pm.draw_detection(left_e_img,iris_l[0][0],iris_l[0][1],iris_l[1])
        left_eye_image = pm.draw_detection(left_eye_image,int(pupil_l[0][0]),int(pupil_l[0][1]),int(pupil_l[1]),color=(0,255,0))
        all_images.append(left_eye_image)

        right_eye_image = pm.draw_detection(right_e_img,iris_r[0][0],iris_r[0][1],iris_r[1])
        right_eye_image = pm.draw_detection(right_eye_image,int(pupil_r[0][0]),int(pupil_r[0][1]),int(pupil_r[1]),color=(0,255,0))
        all_images.append(right_eye_image)

        return left_eye_image,right_eye_image,all_images

    elif(face_type == 2):
        left_eye_coords,right_eye_coords = eye.eye_detect_using_haar(img,"")
        left_e_img = img[left_eye_coords[1]:left_eye_coords[1]+left_eye_coords[3],left_eye_coords[0]:left_eye_coords[0]+left_eye_coords[2]]
        right_e_img = img[right_eye_coords[1]:right_eye_coords[1]+right_eye_coords[3],right_eye_coords[0]:right_eye_coords[0]+right_eye_coords[2]]

        # left_e_img = cv2.resize(left_e_img,(182,182))
        # right_e_img = cv2.resize(right_e_img,(182,182))

        (pupil_l,pupil_r),all_pupil_images_arr = pm.detect_pupil(left_e_img.copy(),right_e_img.copy(),gray=True,face_type=2)
        iris_l,iris_r = pm.detect_iris(left_e_img.copy(),right_e_img.copy(),gray=True,face_type=2)
        
        all_images += all_pupil_images_arr

        left_eye_image = pm.draw_detection(left_e_img,int(pupil_l[0][0]),int(pupil_l[0][1]),int(iris_l[1]))
        left_eye_image = pm.draw_detection(left_eye_image,int(pupil_l[0][0]),int(pupil_l[0][1]),int(pupil_l[1]),color=(0,255,0))
        all_images.append(left_eye_image)

        right_eye_image = pm.draw_detection(right_e_img,int(pupil_r[0][0]),int(pupil_r[0][1]),int(iris_r[1]))
        right_eye_image = pm.draw_detection(right_eye_image,int(pupil_r[0][0]),int(pupil_r[0][1]),int(pupil_r[1]),color=(0,255,0))
        all_images.append(right_eye_image)

        return left_eye_image,right_eye_image,all_images

    elif(face_type == 3):
        pupil_l,all_pupil_images_arr= pm.detect_pupil(img.copy(),None,gray=True,face_type=3)
        iris_l= pm.detect_iris(img.copy(),None,gray=True,face_type=3)

        all_images += all_pupil_images_arr
        eye_image = pm.draw_detection(img,int(pupil_l[0][0]),int(pupil_l[0][1]),int(iris_l[1]))
        eye_image = pm.draw_detection(eye_image,int(pupil_l[0][0]),int(pupil_l[0][1]),int(pupil_l[1]),color=(0,255,0))
        all_images.append(all_pupil_images_arr)
        return eye_image,all_images        

def Video_Functionality(img,iris_mvg,beta,current_frame,dilation_ratios):

    all_images = []
    all_images.append(img)
    face_rects = eye.detectFaces(img.copy(),f_d,verbose=False)
    left_e_coords,right_e_coords = eye.eyes_coordinates(img.copy(),face_rects,k_p_d,left_e,right_e,gray=True)
    left_e_img,right_e_img,left_e_coords,right_e_coords= eye.crop_eyes(img.copy(),left_e_coords,right_e_coords,new_coords=True)
    
    blink = eye.check_blink(left_e_coords,right_e_coords)

    if (not blink):        
        left_e_ki = eye.getEyeKeypointsDistance(left_e_coords)
        right_e_ki = eye.getEyeKeypointsDistance(right_e_coords)

        (pupil_l,pupil_r),all_images_pupils_arr = pm.detect_pupil(left_e_img.copy(),right_e_img.copy(),gray=True,face_type=1)
        iris_l,iris_r = pm.detect_iris(left_e_img.copy(),right_e_img.copy(),
                                        gray=True,
                                        # draw_and_save=True,
                                        key_point_info=(left_e_ki,right_e_ki),
                                        new_coords=(left_e_coords,right_e_coords),
                                        pupil_center=(pupil_l[0],pupil_r[0]),
                                        face_type=1)

        iris_mvg_l = int(beta * iris_mvg[0] + (1-beta)*iris_l[1])
        iris_mvg_r = int(beta * iris_mvg[1] + (1-beta)*iris_r[1])
        iris_mvg = (iris_mvg_l,iris_mvg_r)

        if (iris_l[1]!=0 or iris_r[1]!=0):
            dilation_ratios.append(((pupil_l[1]+pupil_r[1])/2)/((iris_l[1]+iris_r[1])/2))

        all_images += all_images_pupils_arr

        left_eye_image = pm.draw_detection(left_e_img,iris_l[0][0],iris_l[0][1],iris_mvg_l)
        left_eye_image = pm.draw_detection(left_eye_image,int(pupil_l[0][0]),int(pupil_l[0][1]),int(pupil_l[1]),color=(0,255,0))

        all_images.append(left_eye_image)

        right_eye_image = pm.draw_detection(right_e_img,iris_r[0][0],iris_r[0][1],iris_mvg_r)
        right_eye_image = pm.draw_detection(right_eye_image,int(pupil_r[0][0]),int(pupil_r[0][1]),int(pupil_r[1]),color=(0,255,0))

        all_images.append(right_eye_image)

    else:
        left_eye_image,right_eye_image = left_e_img,right_e_img

    return left_eye_image,right_eye_image,iris_mvg,dilation_ratios,all_images
