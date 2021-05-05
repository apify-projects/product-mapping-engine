from PIL import Image
import os

input_directory = 'image_data_cropped_more/'
save_directory = 'image_data_cropped_more_resized/'
width = 800
height = 600
try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)
             
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        input_path = os.path.join(input_directory, filename)
        image = Image.open(input_path)
        new_image = image.resize((width, height))
        output_path = os.path.join(save_directory, filename)
        new_image.save(output_path)

