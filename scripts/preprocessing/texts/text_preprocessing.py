import re

import majka


def set_czech_lemmatizer():
    """
    Set lemmatizer for Czech language
    @return: lemmatizer
    """
    lemmatizer = majka.Majka('data/vocabularies/majka.w-lt')
    lemmatizer.flags |= majka.ADD_DIACRITICS  # find word forms with diacritics
    lemmatizer.flags |= majka.DISALLOW_LOWERCASE  # do not enable to find lowercase variants
    lemmatizer.flags |= majka.IGNORE_CASE  # ignore the word case whatsoever
    lemmatizer.flags = 0  # unset all flags
    lemmatizer.tags = True  # return just the lemma, do not process the tags
    lemmatizer.compact_tag = False  # not return tag in compact form (as returned by Majka)
    lemmatizer.first_only = True  # return only the first entry (not all entries)
    return lemmatizer


def lemmatize_czech_text(text, lemmatizer):
    """
    Lemmatize Czech text
    @param text: text to lemmatize
    @param lemmatizer: lemmatizer
    @return: lemmatized text
    """
    lemmatized_text = []
    for word in text:
        x = lemmatizer.find(word)
        if x == []:
            lemma = word
        else:
            lemma = x[0]['lemma']
            if 'negation' in x[0]['tags'] and x[0]['tags']['negation']:
                lemma = 'ne' + lemma
        lemmatized_text.append(lemma)
    return lemmatized_text


def remove_useless_spaces_and_characters(text):
    """
    Remove useless spaces between numerical values
    @param text: text to remove spaces
    @return: text without useless spaces
    """
    text = re.sub(r'(?<=\d) - (?=\d)', r'-', text)
    text = re.sub(r'(?<=\d),(?=\d)', r'.', text)
    text = re.sub(r'(?<=\d)"', r' inch', text)
    text = text.replace(' × ', '×')
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text


def preprocess_text(data, lemmatizer=None):
    """
    Lowercase and split units and values in dataset
    @param data: list of texts to preprocess
    @param lemmatizer: lemmatizer to be used to lemmatize texts
    @return: preprocessed list of texts that consists of list of words
    """
    new_data = []
    for text in data:
        text = remove_useless_spaces_and_characters(text)
        word_list = split_words(text)
        word_list = split_units_and_values(word_list)
        word_list = lower_case(word_list)
        if lemmatizer is not None:
            word_list = lemmatize_czech_text(word_list, lemmatizer)
        new_data.append(word_list)
    return new_data


def split_words(text):
    """
    Split text to the single words
    @param text: string text to be split
    @return: list of words of text
    """
    rgx = re.compile("\w+[\"\-'×.,%]?\w*")
    word_list = rgx.findall(text)
    return word_list


def split_units_and_values(word_list):
    """
    Split parameter values and units into two words
    @param word_list: list of words
    @return: list of words with split units
    """
    word_list_splitted = []
    for word in word_list:
        if re.match('^[0-9]+[a-zA-Z]+$', word) is not None:
            word_list_splitted.append(re.split('[a-zA-Z]+$', word)[0])
            word_list_splitted.append(re.split('^[0-9]+', word)[1])
        elif re.match('^[0-9]+%$', word) is not None:
            word_list_splitted.append(re.split('%', word)[0])
            word_list_splitted.append(re.split('^[0-9]+', word)[1])
        else:
            word_list_splitted.append(word)
    return word_list_splitted


def lower_case(word_list):
    """
    Lower case all names in dataset
    @param word_list: list of words
    @return: lowercased list of words
    """
    lowercased_word_list = []
    for word in word_list:
        lowercased_word_list.append(word.lower())
    return lowercased_word_list
