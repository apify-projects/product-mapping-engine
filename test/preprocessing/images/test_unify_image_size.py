import click

from scripts.dataset_handler.preprocessing.images import create_output_directory, unify_image_size


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
              default='test/data/10_products/dataset/source/images',
              required=False,
              help='Folder with input images to be resized')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/images_resized',
              required=False, help='Folder to store resized images')
# Load folder with images and resize them to required size
def main(**kwargs):
    create_output_directory(kwargs['output_file'])
    unify_image_size(kwargs['input_file'], kwargs['output_file'], kwargs['width'], kwargs['height'])


if __name__ == '__main__':
    main()
