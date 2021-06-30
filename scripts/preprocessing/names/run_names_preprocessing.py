import click

from names_preprocessing import load_input_file, detect_ids_brands_and_colors, write_to_file


@click.command()
@click.option('--input_file', '-i',
              default='C:/Users/kater/PycharmProjects/product-mapping/data/source/10_products/names/10_products_b.csv',
              required=False,
              help='Input file with product names dictionary file to preprocess')
@click.option('--output_file', '-o',
              default='C:/Users/kater/PycharmProjects/product-mapping/data/preprocessed/10_products/names/10_products_prepro_b.txt',
              required=False, help='Output preprocessed file with product names')
# Load product names and search for ids, brands, colors and parameters
def main(**kwargs):
    data = load_input_file(kwargs['input_file'])

    compare_words = False
    data, cnt_voc, cnt_lem = detect_ids_brands_and_colors(data, compare_words=compare_words)
    if compare_words:
        print('Number of words in names that were in manually created vocabulary: ' + str(cnt_voc))
        print('Number of words in names that were recognised in Morphoditta: ' + str(cnt_lem))

    write_to_file(data, kwargs['output_file'])


if __name__ == '__main__':
    main()
