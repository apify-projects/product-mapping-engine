import click

from scripts.preprocessing.names.names_preprocessing import to_list, detect_ids_brands_colors_and_params


@click.command()
@click.option('--input_file', '-i',
              default='test/data/10_products/dataset/source/names/names_products_1.csv',
              required=False,
              help='Input file with product names dictionary file to preprocess')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/names/names_products_1_prepro.csv',
              required=False, help='Output results file with product names')
# Load product names and search for ids, brands, colors and parameters and save the preprocessed product names to git output file
def main(**kwargs):
    with open(kwargs['input_file'], encoding='utf-8') as f:
        data = [line.rstrip() for line in f]
    data = to_list(data)

    compare_words = False
    data, cnt_voc, cnt_lem = detect_ids_brands_colors_and_params(data, compare_words=compare_words)

    if compare_words:
        print('Number of words in names that were in manually created vocabulary: ' + str(cnt_voc))
        print('Number of words in names that were recognised in Morphoditta: ' + str(cnt_lem))

    with open(kwargs['output_file'], 'w', encoding='utf-16') as f:
        for d in data:
            f.write(' '.join([str(word) for word in d]))
            f.write('\n')


if __name__ == '__main__':
    main()
