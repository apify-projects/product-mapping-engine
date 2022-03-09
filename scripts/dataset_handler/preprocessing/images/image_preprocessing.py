import imghdr
import json
import os
import shutil
import cv2
import numpy as np
import subprocess
from PIL import Image
from ....configuration import IMAGE_RESIZE_WIDTH, IMAGE_RESIZE_HEIGHT


def unify_image_size(input_folder, output_folder, width, height):
    """
    Unify size of images to given shapes
    @param input_folder: folder with input images
    @param output_folder: folder to store output images
    @param width: required width of images
    @param height: required height of images
    @return:
    """
    for filename in os.listdir(input_folder):
        if imghdr.what(os.path.join(input_folder, filename)) is not None:
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path)
            new_image = image.resize((width, height))
            output_path = os.path.join(output_folder, filename)
            new_image.save(output_path)


def crop_images_simple(input_folder, output_folder):
    """
    Crop images using corner detection of the objects in white background
    @param input_folder: folder with input images
    @param output_folder: folder to store output images
    @return:
    """
    for filename in os.listdir(input_folder):
        if imghdr.what(os.path.join(input_folder, filename)) is not None:
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)

            blurred = cv2.blur(image, (3, 3))
            canny = cv2.Canny(blurred, 50, 200)

            # find the non-zero min-max coordinates of canny
            pts = np.argwhere(canny > 0)
            y1, x1 = pts.min(axis=0)
            y2, x2 = pts.max(axis=0)

            # crop the region
            cropped = image[y1:y2, x1:x2]
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped)


def crop_images_contour_detection(input_folder, filenames, output_folder):
    """
    Crop images using contour objects detection
    @param input_folder: folder with input images
    @param filenames: filenames of the images to be cropped
    @param output_folder: folder to store output images
    @return:
    """
    for filename in filenames:
        if imghdr.what(os.path.join(input_folder, filename)) is not None:
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)

            # resizing the images to max height and width to increase speed and preserve memory
            width_resize_ratio = IMAGE_RESIZE_WIDTH / image.shape[1]
            height_resize_ratio = IMAGE_RESIZE_HEIGHT / image.shape[0]
            resize_ratio = min(width_resize_ratio, height_resize_ratio)
            if resize_ratio < 1:
                image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)

            # add white border around image
            color = [255, 255, 255]
            w = image.shape[1] // 10
            h = image.shape[0] // 10
            top, bottom, left, right = [h, h, w, w]
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

            # converting to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # create black and white mask of object
            (thresh, im_bw) = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = 240
            im_bw = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)[1]

            # applying canny edge detection
            canny = cv2.Canny(im_bw, 10, 100)
            kernel = np.ones((5, 5), np.uint8)
            dilate = cv2.dilate(canny, kernel, iterations=1)

            # finding contours
            contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # select max object in the picture
            max_y, max_x, max_w, max_h = 0, 0, 0, 0
            max_area = 0
            for i, c in enumerate(contours):
                x, y, w, h = cv2.boundingRect(c)
                if w * h > max_area:
                    max_y, max_x, max_w, max_h = y, x, w, h
                    max_area = w * h

            # crop image to the biggest found object
            cropped = image[max_y:max_y + max_h, max_x:max_x + max_w]
            cv2.imwrite(f'{output_folder}/{filename}.jpg', cropped)


def compute_image_hashes(index, dataset_folder, img_dir, assigned_filenames, script_dir):
    index = str(index)
    cropped_img_dir = os.path.join(dataset_folder, 'images_cropped_{}'.format(index))
    create_output_directory(cropped_img_dir)
    crop_images_contour_detection(img_dir, assigned_filenames, cropped_img_dir)
    hashes_path = os.path.join(dataset_folder, 'hashes_cropped_{}.json'.format(index))
    subprocess.call(f'node {script_dir} {cropped_img_dir} {hashes_path}', shell=True)
    return hashes_path


def create_output_directory(output_folder):
    """
    Check whether output directory exists - if yes empty it, if not create it
    @param output_folder: output folder to be checked
    @return:
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)


def load_and_parse_data(input_files):
    """
    Load input file and split name and hash into dictionary
    @param input_files: files with hashes and names
    @return: dictionary with name and has value of the image
    """
    data = {}

    for input_file in input_files:
        with open(input_file) as json_file:
            loaded_data = json.load(json_file)

        for image_name_and_hash_pair in loaded_data:
            data_split = image_name_and_hash_pair.split(';')
            data[data_split[0]] = data_split[1]

    return data
