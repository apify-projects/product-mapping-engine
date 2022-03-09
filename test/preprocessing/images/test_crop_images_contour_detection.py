import click

from scripts.dataset_handler.preprocessing.images import create_output_directory, crop_images_contour_detection


@click.command()
@click.option('--input_file', '-i',
              default='test/data/10_products/dataset/source/images',
              required=False,
              help='Folder with input images to be cropped')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/images_cropped',
              required=False, help='Folder to store cropped images')
# Load folder with images and crop them using contour detection
def main(**kwargs):
    create_output_directory(kwargs['output_file'])
    crop_images_contour_detection(kwargs['input_file'], kwargs['output_file'])


if __name__ == '__main__':
    main()
