import click


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
              help='Output Czech vocabulary file after preprocessing')
def load_and_split_corpus(corpus_file, separator='\t'):
    """
    Load corpus file and split Czech and English sentences
    @param separator: separator of Czech and English sentences
    @param corpus_file: input file with corpus data
    @return: list of English sentences, list of Czech sentences
    """
    lines_cz = []
    lines_en = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.split(separator)
            lines_en.append(line_split[0])
            lines_cz.append(line_split[1])
    return lines_en, lines_cz


def compute_unique_words_occurrences(list_of_sentences):
    """
    Split sentences according to spaces and compute words occurrences
    @param list_of_sentences: input string sentences
    @return: dictionary of unique words and their occurrences
    """
    words_dict = []
    for sentence in list_of_sentences:
        for word in sentence.split(' '):
            word = word.lower()
            if word.isalpha():
                if not word in words_dict:
                    words_dict[word] = 0
                words_dict[word] += 1

    words_dict = sort_dictionary(words_dict)
    return words_dict


def sort_dictionary(word_dict):
    """
    Sort dictionary of unique words and their occurrences in descending order
    @param word_dict: Unordered word dictionary
    @return: Ordered word dictionary
    """
    return dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))


def save_dictionary(word_dict, output_file):
    """
    Save dictionary with uniques words and their occurrences to output file
    @param word_dict: dictionary with uniques words
    @param output_file: file name to save the dictionary
    @return:
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, occurrences in word_dict.items():
            f.write(f'{word}, {occurrences}\n')


# Load corpus and create English and Czech dictionary of unique words
def main(**kwargs):
    lines_en, lines_cz = load_and_split_corpus(kwargs['corpus_file'])
    en_dict = compute_unique_words_occurrences(lines_en)
    cz_dict = compute_unique_words_occurrences(lines_cz)
    save_dictionary(en_dict, kwargs['en_vocabulary_file'])
    save_dictionary(cz_dict, kwargs['cz_vocabulary_file'])


if __name__ == '__main__':
    main()