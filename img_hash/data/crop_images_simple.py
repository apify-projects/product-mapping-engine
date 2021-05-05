import cv2
import numpy as np
import os

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

        blurred = cv2.blur(image, (3,3))
        canny = cv2.Canny(blurred, 50, 200)

        # find the non-zero min-max coords of canny
        pts = np.argwhere(canny>0)
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)

        # crop the region
        cropped = image[y1:y2, x1:x2]
        output_path = os.path.join(save_directory, filename)
        cv2.imwrite(output_path, cropped)



