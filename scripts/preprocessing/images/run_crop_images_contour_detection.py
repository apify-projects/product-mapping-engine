import os

import click
import cv2
import numpy as np


@click.command()
@click.option('--input_file', '-i',
              default='C:/Users/kater/PycharmProjects/product-mapping/data/source/10_products/images',
              required=False,
              help='Folder with input images to be cropped')
@click.option('--output_file', '-o',
              default='C:/Users/kater/PycharmProjects/product-mapping/data/preprocessed/10_products/images/cropped_masked',
              required=False, help='Folder to store cropped images')
# Load folder with images and crop them using contour detection
def main(**kwargs):
    try:
        os.stat(kwargs['output_file'])
    except:
        os.mkdir(kwargs['output_file'])

    max_object = False

    for filename in os.listdir(kwargs['input_file']):
        if filename.endswith('.jpg'):
            input_path = os.path.join(kwargs['input_file'], filename)
            image = cv2.imread(input_path)

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
            contours, hierarchies = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if max_object:
                # select max object in the picture
                maxy, maxx, maxw, maxh = 0, 0, 0, 0
                maxarea = 0
                for i, c in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(c)
                    cropped = image[y:y + h, x:x + w]
                    cv2.imwrite(f'{kwargs["output_file"]}{filename[:-4]}_{i}.jpg', cropped)
                    if w * h > maxarea:
                        maxy, maxx, maxw, maxh = y, x, w, h
                        maxarea = w * h

                # crop image to the biggest found object
                cropped = image[maxy:maxy + maxh, maxx:maxx + maxw]
                cv2.imwrite(f'{kwargs["output_file"]}{filename}', cropped)


if __name__ == '__main__':
    main()
