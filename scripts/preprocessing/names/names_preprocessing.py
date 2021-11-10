import json
import os
import re
import time

import requests

from scripts.preprocessing.description.short_description_preprocessing import detect_parameters

ID_LEN = 5
COLOR_PREFIX = '#col#'
CURRENT_SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
COLORS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/colors.txt')
BRANDS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/brands.txt')
VOCABULARY_EN_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/en_dict_cleaned.csv')
VOCABULARY_CZ_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/cz_dict_cleaned.csv')


def load_colors():
    """
    Load file with list of colors
    @return: list of loaded colors
    """
    colors = []
    file = open(COLORS_FILE, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        colors.append(line)
        if line[len(line) - 1:] == 'á':
            colors.append(line[:-1] + 'ý')
            colors.append(line[:-1] + 'é')
    return colors


def load_brands():
    """
    Load file with list of brands
    @return: list of loaded brands
    """
    brands = []
    file = open(BRANDS_FILE, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        brands.append(line)
    return brands


def load_vocabulary(vocabulary_file):
    """
    Load file with list of vocabulary
    @return: loaded list of words in vocabulary
    """
    with open(vocabulary_file, encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def does_this_word_exist(lemma):
    """
    Check whether the word is in Czech or English vocabulary in LINDAT repository
    @param lemma: the word to be checked
    @return: True if it is an existing word, otherwise False
    """
    while True:
        try:
            url_cz = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={lemma}&output=json&guesser=no&model=czech-morfflex-pdt-161115"
            url_en = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={lemma}&output=json&guesser=no&model=english-morphium-wsj-140407"

            r_cz = json.loads(requests.get(url_cz).text)['result']
            r_en = json.loads(requests.get(url_en).text)['result']

            if not r_cz or not r_en:
                return False

            if r_cz[0][0]['tag'] == 'X@-------------' and r_en[0][0]['tag'] == 'UNK':
                return False
            return True
        except:
            time.sleep(1)


def is_in_vocabulary(word):
    """
    Check whether a word is in the vocabulary which was created manually from corpus
    @param word: a word to be checked
    @return: True if it is an existing word (in one of the vocabularies), otherwise False
    """
    if word in VOCABULARY_CZ or word in VOCABULARY_EN:
        return True
    return False


def is_param(word):
    """
    Check whether string is not a parameter
    @param word: the word to be checked
    @return: True if it is a parameter from description, otherwise False
    """
    rgx = re.compile("^[0-9]+[A-Za-z]+$|^[A-Za-z]+[0-9]+$")
    if re.match(rgx, word):
        return True
    return False


def detect_id(word):
    """
    Check whether the word is not and id (whether it is a valid word)
    @param word: the word to be checked
    @return: word with marker if it is an id, otherwise the original word
    """
    # check whether is capslock ad whether is not too short
    word_sub = re.sub(r"[\W_]+", "", word, flags=re.UNICODE)
    if (not word_sub.isupper() and word_sub.isalpha()) or len(
            word_sub) < ID_LEN or word in VOCABULARY_CZ or word in VOCABULARY_EN:
        return word

    if word_sub.isnumeric():
        word = word.replace("(", "").replace(")", "")
        return '#id#' + word
    elif word_sub.isalpha():
        if not is_in_vocabulary(word_sub):
            return '#id#' + word
    else:
        word = word.replace("(", "").replace(")", "")
        if not is_param(word):
            return '#id#' + word
    return word


def detect_color(word):
    """
    Check whether the word is not in list of colors
    @param word: the word to be checked
    @return: word with marker if it is a colors, otherwise the original word
    """
    if word.lower() in COLORS:
        word = COLOR_PREFIX + word
    return word


def detect_vocabulary_words(word):
    """
    Check whether the word is in vocabulary of words
    @param word: the word to be checked
    @return: word with marker if it is in a vocabulary, otherwise the original word
    """
    if word.lower() in VOCABULARY_CZ or word.lower() in VOCABULARY_EN:
        return "#voc#" + word
    return word


def detect_brand(word, is_first, first_likelihood):
    """
    Check whether the word is not in list of brands
    @param word: the word to be checked
    @param is_first: the word to be checked
    @param first_likelihood: the probability that this word is the first one in titles that include it
    @return: word with if it is a brand, otherwise the original word
    """
    is_brand = False

    if word.lower() in BRANDS:
        is_brand = True
    elif is_first:
        if (word.isalpha() and len(word) < ID_LEN and word.isupper()) or first_likelihood > 0.9:
            is_brand = True

    return "#bnd#" + word if is_brand else word


def detect_ids_brands_colors_and_params(data, compare_words, id_detection=True, color_detection=True,
                                        brand_detection=True,
                                        parameters_detection=True):
    """
    Detect ids, colors, brands and description parameters in names
    @param data: List of product names to be checked
    @param compare_words: True if we want to compare number of words from names found in dictionary and in Marphoditta
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param parameters_detection: True if parameters should be detected
    @return: names with detected stuff, eventually number of lemmas from vocabulary and lemmas from morphoditta
    """
    data_list = []
    cnt_voc = 0
    cnt_lem = 0

    word_counts = {}
    first_likelihood = {}
    new_data = []
    for name in data:
        name = split_units_and_values(name)
        new_data.append(name)
        is_first = True
        for word in name:
            if word not in word_counts:
                word_counts[word] = 0
                first_likelihood[word] = 0
            word_counts[word] += 1
            if is_first:
                first_likelihood[word] += 1
            is_first = False

    for word in first_likelihood:
        first_likelihood[word] = first_likelihood[word] / word_counts[word]

    # print(first_likelihood)

    for name in new_data:
        word_list = []
        is_first = True
        for word in name:
            if color_detection:
                word = detect_color(word)
            if brand_detection and not word.startswith(COLOR_PREFIX):
                word = detect_brand(word, is_first, first_likelihood[word])
            if id_detection:
                word = detect_id(word)

            word_list.append(word)

            # compute number of words that are in dictionary and that were found in Morphoditta
            word = re.sub(r"[\W_]+", "", word, flags=re.UNICODE).lower()

            if compare_words:
                rec_lem = False
                rec_voc = False
                if is_in_vocabulary(word):
                    rec_voc = True
                    cnt_voc += 1
                if does_this_word_exist(word):
                    rec_lem = True
                    cnt_lem += 1
                if (rec_voc and not rec_lem) or (not rec_voc and rec_lem):
                    print(word)

            is_first = False

        if parameters_detection:
            word_list, _ = detect_parameters(word_list)
        data_list.append(word_list)

    return data_list, cnt_voc, cnt_lem


def to_list(data):
    """
    Convert list of names to list of list of words which the name constists of
    @param data: input list on names
    @return: list of names where each name is a list of words
    """
    rgx = re.compile("(\w+['-]?[\w]*)")
    data_list = []
    for d in data:
        words = rgx.findall(d)
        if words != '':
            data_list.append(words)

    return data_list


def split_units_and_values(data):
    """
    Split parameter values and units into two words
    @param data: data with list of product names
    @return: splitted data with list of product names
    """
    words_splitted = []
    for word in data:
        if re.match('^[0-9]+[a-zA-Z]+$', word) is not None:
            words_splitted.append(re.split('[a-zA-Z]+$', word)[0])
            words_splitted.append(re.split('^[0-9]+', word)[1])
        elif re.match('^[0-9]+%$', word) is not None:
            words_splitted.append(re.split('%', word)[0])
            words_splitted.append(re.split('^[0-9]+', word)[1])
        else:
            words_splitted.append(word)
    return words_splitted

def split_words(text):
    """
    Split text to the single words
    @param text: text to be split
    @return: list of words of text
    """
    rgx = re.compile("\w+[\"\-'×.,%]?\w*")
    split_text = rgx.findall(text)
    return split_text

BRANDS = load_brands()
COLORS = load_colors()
VOCABULARY_CZ = load_vocabulary(VOCABULARY_CZ_FILE)
VOCABULARY_EN = load_vocabulary(VOCABULARY_EN_FILE)
