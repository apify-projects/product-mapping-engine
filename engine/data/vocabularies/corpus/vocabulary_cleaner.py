import csv
import json
import time
import click
import requests


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
              default='data/vocabularies/corpus/preprocessed/cz_dict.csv',
              required=False,
              help='Input vocabulary dictionary file to preprocess')
@click.option('--output_file', '-o',
              default='data/vocabularies/corpus/preprocessed/cz_dict_cleaned.txt',
              required=False, help='Output file with valid uniques words.')
def lemmatize_batch(batch_str, language):
    """
    Lemmatize batch of words using Morphoditta
    @param language: language of words in vocabulary
    @param batch_str: number of words to be lemmatized at once
    @return: list of valid words
    """
    words = []
    if language == 'CZ':
        url = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={batch_str}&output=json&guesser=no&model=czech-morfflex-pdt-161115"
    else:
        url = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={batch_str}&output=json&guesser=no&model=english-morphium-wsj-140407"

    r = json.loads(requests.get(url).text)['result']
    for word in r[0]:
        if (not language == 'CZ' and word['tag'] != 'UNK') or (language == 'CZ' and word['tag'] != 'X@-------------'):
            words.append(word['token'])
    return words


def load_and_lemmatize_dictionary(input_file, batch_size, language):
    """
    Load dictionary and check whether the words in manually created vocabulary from corpus are existing words using MORPHODITTA
    @param language: language of words in vocabulary
    @param batch_size: number of words to be preprocessed at once
    @param input_file: Name of the input dictionary
    @return: cleaned vocabulary
    """
    i = 0
    batch = []
    batch_str = ''
    with open(input_file, encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            batch.append(row[0])
            if len(batch) == batch_size:
                print(i)  # print progress
                i += 1
                batch_str = ' '.join(b for b in batch)
                sleep_time = 1
                while True:
                    try:
                        lemmatize_batch(batch_str, language)
                        batch = []
                        sleep_time = 1
                        time.sleep(sleep_time)
                        break
                    except Exception as e:
                        time.sleep(sleep_time)
                        print(f'Lemmatizer failed, sleeping for {sleep_time} secs and trying again.')
                        sleep_time *= 2
    lemmatize_batch(batch_str, language)


def save_vocabulary(words, output_file):
    """
    Save unique words from vocabulary to output file
    @param words: unique words from vocabulary
    @param output_file: file to save uniques words
    @return:
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for w in words:
            f.write(f"{w}\n")


# Load vocabulary and create vocabulary with unique valid words
# Switch to load czech or english dictionary to process
def main(**kwargs):
    vocabulary = load_and_lemmatize_dictionary(kwargs['input_file'], kwargs['batch_size'], kwargs['language'])
    save_vocabulary(vocabulary, kwargs['output_file'])


if __name__ == '__main__':
    main()
