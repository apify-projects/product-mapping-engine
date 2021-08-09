import json
import os
import re
import time

import requests

ID_LEN = 5

CURRENT_SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
COLORS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/colors.txt')
BRANDS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/brands.txt')
VOCABULARY_EN_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/en_dict_cleaned.csv')
VOCABULARY_CZ_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/cz_dict_cleaned.csv')


def load_input_file(input_file):
    """
    Load input file with product names
    @param input_file: file name
    @return: list of product names each converted to list of words
    """
    with open(input_file, encoding='utf-8') as f:
        data = [line.rstrip() for line in f]
        return to_list(data)


def load_colors():
    """
    Load file with colors
    @return: list od colors
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
    Load file with brands
    @return: list od brands
    """
    brands = []
    file = open(BRANDS_FILE, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        brands.append(line)
    return brands


def load_vocabulary(vocabulary_file):
    """
    Load file with vocabulary
    @return: list od words in vocabulary
    """
    with open(vocabulary_file, encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def write_to_file(data, output_file):
    """
    Write preprocessed product names into file
    @param data: preprocessed data
    @param output_file: output file
    @return:
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(' '.join(d))
            f.write('\n')


def does_this_word_exist(lemma):
    """
    Check whether the word is in Czech or English vocabulary in LINDAT repository
    @param lemma: word to be checked
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


def is_word_in_vocabulary(word):
    """
    Check whether a word is in vocabulary cretaed manually from corpus
    @param word: word to be checked
    @return: True if it is an existing word (in one of the vocabularies), otherwise False
    """
    if word in VOCABULARY_CZ or word in VOCABULARY_EN:
        return True
    return False


def is_param(word):
    """
    Check whether string is not parameter
    @param word: word to be checked
    @return: True if it is a parameter from specification, otherwise False
    """
    rgx = re.compile("^[0-9]+[A-Za-z]+$|^[A-Za-z]+[0-9]+$")
    if re.match(rgx, word):
        return True
    return False


def id_detection(word):
    """
    Check whether word is not and id
    @param word: word to be checked
    @return: word with marker whether it is an id
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
        if not is_word_in_vocabulary(word_sub):
            return '#id#' + word
    else:
        word = word.replace("(", "").replace(")", "")
        if not is_param(word):
            return '#id#' + word
    return word


def color_detection(word):
    """
    Check whether word is not in list of colors
    @param word: word to be checked
    @return: word with marker whether it is a colors
    """
    if word.lower() in COLORS:
        word = '#col#' + word
    return word


def vocabulary_detection(word):
    """
    Check whether word is in vocabulary of words
    @param word: word to be checked
    @return: word with marker whether it is in a vocabulary
    """
    if word.lower() in VOCABULARY_CZ or word.lower() in VOCABULARY_EN:
        return "#voc#" + word
    return word


def brand_detection(word):
    """
    Check whether word is not in list of brands
    @param word: word to be checked
    @return: word with marker whether it is a brand
    """
    is_brand = False

    if word.lower() in BRANDS:
        is_brand = True
    elif word.isalpha() and len(word) < ID_LEN and word.isupper():
        is_brand = True

    return "#bnd#" + word if is_brand else word


def detect_ids_brands_and_colors(data, compare_words, detect_id=True, detect_color=True, detect_brand=True):
    """
    Detect ids, colors, brands and specification parameters in names
    @param data: List of product names to be checked
    @param detect_id: True if id should be detected
    @param detect_color: True if color should be detected
    @param detect_brand: True if brand should be detected
    @param compare_words: True if we want to compare number of words from names found in dictionary and in Marphoditta
    @return:
    """
    data_list = []
    cnt_voc = 0
    cnt_lem = 0

    word_statistics = {}

    for name in data:
        # print(name)
        word_list = []
        for word in name:
            if detect_color:
                word = color_detection(word)
            if detect_brand:
                word = brand_detection(word)
            if detect_id:
                word = id_detection(word)

            word_list.append(word)

            # compute number of words that are in dictionary and that were found in Morphoditta
            word = re.sub(r"[\W_]+", "", word, flags=re.UNICODE).lower()

            if compare_words:
                rec_lem = False
                rec_voc = False
                if is_word_in_vocabulary(word):
                    rec_voc = True
                    cnt_voc += 1
                if does_this_word_exist(word):
                    rec_lem = True
                    cnt_lem += 1
                if (rec_voc and not rec_lem) or (not rec_voc and rec_lem):
                    print(word)

        data_list.append(word_list)
    return data_list, cnt_voc, cnt_lem


def to_list(data):
    """
    Convert list of names to list of list of words
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


BRANDS = load_brands()
COLORS = load_colors()
VOCABULARY_CZ = load_vocabulary(VOCABULARY_CZ_FILE)
VOCABULARY_EN = load_vocabulary(VOCABULARY_EN_FILE)
