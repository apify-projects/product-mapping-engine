import os

import click
from PIL import Image


@click.command()
@click.option('--width', '-w',
              default='800',
              required=False,
              type=int,
              help='Required with for resizing image')
@click.option('--height', '-h',
              default='600',
              required=False,
              type=int,
              help='Required height for resizing image')
@click.option('--input_file', '-i',
              default='C:/Users/kater/PycharmProjects/ProductMapping/data/preprocessed/10_products/images/cropped',
              required=False,
              help='Folder with input images to be resized')
@click.option('--output_file', '-o',
              default='C:/Users/kater/PycharmProjects/ProductMapping/data/preprocessed/10_products/images/cropped_resized',
              required=False, help='Folder to store resized images')
# Load folder with images and resize them to required size
def main(**kwargs):
    try:
        os.stat(kwargs['output_file'])
    except:
        os.mkdir(kwargs['output_file'])

    for filename in os.listdir(kwargs['input_file']):
        if filename.endswith('.jpg'):
            input_path = os.path.join(kwargs['input_file'], filename)
            image = Image.open(input_path)
            new_image = image.resize((kwargs['width'], kwargs['height']))
            output_path = os.path.join(kwargs['output_file'], filename)
            new_image.save(output_path)


if __name__ == '__main__':
    main()
