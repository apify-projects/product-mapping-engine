import cv2
import numpy as np
from PIL import Image
import os

inp_directory = 'image_data/'
save_directory = 'image_data_resized/'

try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)
             
for filename in os.listdir(inp_directory):
    if filename.endswith('.jpg'):
        path_inp = os.path.join(inp_directory, filename)
        image = Image.open(path_inp)
        new_image = image.resize((800, 600))
        path_out = os.path.join(save_directory, filename)
        new_image.save(path_out)

