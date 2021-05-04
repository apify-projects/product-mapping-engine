import cv2
import os

inp_directory = 'image_data/'
save_directory = 'image_data_cropped_more2/'


try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)
 

for filename in os.listdir(inp_directory):
    if filename.endswith('.jpg'):
        path_inp = os.path.join(inp_directory, filename)
        image = cv2.imread(path_inp)
        #converting to gray scale
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #applying canny edge detection
        edged = cv2.Canny(gray, 10, 100)
        
        #finding contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maxy, maxx, maxw, maxh = 0, 0, 0, 0
        maxarea = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h > maxarea:
                maxy, maxx, maxw, maxh = x,y,w,h
                maxarea = w*h
                
        new_img=image[maxy:maxy+maxh,maxx:maxx+maxw]
        cv2.imwrite(f'{save_directory}{filename}', new_img)
        