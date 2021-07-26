import os

import click
import cv2
import numpy as np


@click.command()
@click.option('--input_file', '-i',
              default='data/source/10_products/images',
              required=False,
              help='Folder with input images to be cropped')
@click.option('--output_file', '-o',
              default='data/preprocessed/10_products/images/cropped',
              required=False, help='Folder to store cropped images')
# Load folder with images and crop them by detection of white surrounding
def main(**kwargs):
    try:
        os.stat(kwargs['output_file'])
    except:
        os.mkdir(kwargs['output_file'])

    for filename in os.listdir(kwargs['input_file']):
        if filename.endswith('.jpg'):
            input_path = os.path.join(kwargs['input_file'], filename)
            image = cv2.imread(input_path)

            blurred = cv2.blur(image, (3, 3))
            canny = cv2.Canny(blurred, 50, 200)

            # find the non-zero min-max coords of canny
            pts = np.argwhere(canny > 0)
            y1, x1 = pts.min(axis=0)
            y2, x2 = pts.max(axis=0)

            # crop the region
            cropped = image[y1:y2, x1:x2]
            output_path = os.path.join(kwargs['output_file'], filename)
            cv2.imwrite(output_path, cropped)


if __name__ == '__main__':
    main()
