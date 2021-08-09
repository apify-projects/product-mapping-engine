import click

from scripts.preprocessing.images.image_preprocessing import create_output_directory, crop_images_simple


@click.command()
@click.option('--input_file', '-i',
              default='test/data/10_products/dataset/source/images',
              required=False,
              help='Folder with input images to be cropped')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/images_cropped',
              required=False, help='Folder to store cropped images')
# Load folder with images and crop them by detection of white surrounding
def main(**kwargs):
    create_output_directory(kwargs['output_file'])
    crop_images_simple(kwargs['input_file'], kwargs['output_file'])


if __name__ == '__main__':
    main()
