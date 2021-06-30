import click

from vocabulary_cleaner import save_vocabulary, load_and_lemmatize_dictionary


@click.command()
@click.option('--batch_size', '-b',
              default='500',
              required=False,
              type=int,
              help='Batch size to be preprocessed')
@click.option('--language', '-l',
              default='cz',
              required=False,
              help='Language of input vocabulary')
@click.option('--input_file', '-i',
              default='C:/Users/kater/PycharmProjects/product-mapping/data/vocabularies/corpus/preprocessed/cz_dict.csv',
              required=False,
              help='Input vocabulary dictionary file to preprocess')
@click.option('--output_file', '-o',
              default='C:/Users/kater/PycharmProjects/product-mapping/data/vocabularies/corpus/preprocessed/cz_dict_cleaned.txt',
              required=False, help='Output file with valid uniques words.')
# Load vocabulary and create vocabulary with unique valid words
# Switch to load czech or english dictionary to process
def main(**kwargs):
    vocabulary = load_and_lemmatize_dictionary(kwargs['input_file'], kwargs['batch_size'], kwargs['language'])
    save_vocabulary(vocabulary, kwargs['output_file'])
    print()


if __name__ == '__main__':
    main()
