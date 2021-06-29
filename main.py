import sys
import click

@click.option('--input_file', '-i', help='', default='', required=True)
@click.option('--output_file', '-o', help='', default='', required=True)

def main(**kwargs):
    print(sys.argv)


if __name__ == '__main__':
    main()

