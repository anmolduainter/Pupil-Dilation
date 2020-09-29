import cv2
from PIL import Image,ImageFont,ImageDraw
import numpy

'''
1. Original - 1
2. left + right - 2
2. Gray + Blurred - 2
3. Historgram Equal. - 2
4. Threshold - 2
5. Final - 2
'''
def FusedImage(all_image_arr,face_type=1):
    if (face_type==1 or face_type==2): 
        print(len(all_image_arr))
        if (len(all_image_arr) == 11):    
            input_image = all_image_arr[0]
            input_image_pil = Image.fromarray(cv2.resize(input_image,(1280,1280)))

            max_width = 1280 + 256 + 256
            max_height = 1280
            new_img = Image.new('RGB',(max_width,max_height))
            new_img.paste(input_image_pil,(0,0))

            for i in range(1,5):
                left_eye_image_pil = Image.fromarray(cv2.resize(all_image_arr[i],(256,256)))
                new_img.paste(left_eye_image_pil,(1280,(i-1)*256))
            
            for i in range(5,9):
                right_eye_image_pil = Image.fromarray(cv2.resize(all_image_arr[i],(256,256)))
                new_img.paste(right_eye_image_pil,(1280+256,(i-5)*256))
            
            new_img.paste(Image.fromarray(cv2.resize(all_image_arr[9],(256,256))),(1280,(4*256)))
            new_img.paste(Image.fromarray(cv2.resize(all_image_arr[10],(256,256))),(1280+256,(4*256)))
            return numpy.array(new_img)
        else:
            input_image = all_image_arr[0]
            input_image_pil = Image.fromarray(cv2.resize(input_image,(1280,1280)))

            max_width = 1280 + 256 + 256
            max_height = 1280
            new_img = Image.new('RGB',(max_width,max_height))
            new_img.paste(input_image_pil,(0,0))

            font = ImageFont.truetype("./utils/Chunkfive.otf",30)
            draw = ImageDraw.Draw(new_img)
            draw.text((1280+30,1280/2),"Blink Detected!",font=font,fill=(0,0,255,255))
            return numpy.array(new_img)

    elif(face_type == 3):
        print (len(all_image_arr))
        input_image = all_image_arr[0]
        input_image_pil = Image.fromarray(cv2.resize(input_image,(1280-320,512)))

        max_width = 1280 + 256 - 320
        max_height = 1280
        new_img = Image.new('RGB',(max_width,max_height))
        new_img.paste(input_image_pil,(0,256))

        for i in range(1,5):
            left_eye_image_pil = Image.fromarray(cv2.resize(all_image_arr[i],(320,320)))
            new_img.paste(left_eye_image_pil,(1280-320,(i-1)*320))
        return numpy.array(new_img)
        