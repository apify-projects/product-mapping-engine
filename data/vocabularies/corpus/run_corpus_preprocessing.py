import click

from corpus_preprocessing import load_and_split_corpus, compute_unique_words_occurrences, save_dictionary


@click.command()
@click.option('--corpus_file', '-i',
              default='data/vocabularies/corpus/source/en-cs.txt',
              required=False, help='Input corpus file to preprocess')
@click.option('--en_vocabulary_file', '-e',
              default='data/vocabularies/corpus/preprocessed/en_dict.csv',
              required=False,
              help='Output English vocabulary file after preprocessing')
@click.option('--cz_vocabulary_file', '-c',
              default='data/vocabularies/corpus/preprocessed/cz_dict.csv',
              required=False,
              help='Output Czech vocabulary wile after preprocessing')
# Load corpus and create English and Czech dictionary of unique words
def main(**kwargs):
    lines_en, lines_cz = load_and_split_corpus(kwargs['corpus_file'])
    en_dict = compute_unique_words_occurrences(lines_en)
    cz_dict = compute_unique_words_occurrences(lines_cz)
    save_dictionary(en_dict, kwargs['en_vocabulary_file'])
    save_dictionary(cz_dict, kwargs['cz_vocabulary_file'])


if __name__ == '__main__':
    main()
