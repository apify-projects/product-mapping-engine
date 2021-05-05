import cv2
import os
import numpy as np

input_directory = 'image_data/'
save_directory = 'image_data_cropped/'


try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)
 

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        input_path = os.path.join(input_directory, filename)
        image = cv2.imread(input_path)
        
        # converting to gray scale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # applying canny edge detection
        canny = cv2.Canny(blurred, 10, 100)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)
        
        # finding contours
        contours, hierarchies = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # select max object in the picture
        maxy, maxx, maxw, maxh = 0, 0, 0, 0
        maxarea = 0
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w*h > maxarea:
                maxy, maxx, maxw, maxh = y,x,w,h
                maxarea = w*h
        
        # crop image to the biggest found object
        cropped = image[maxy:maxy+maxh,maxx:maxx+maxw]
        cv2.imwrite(f'{save_directory}{filename}', cropped)
        
