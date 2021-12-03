import json
import os
import re
import time

import pandas as pd
import requests

ID_LEN = 5
ID_MARK = '#id#'
BRAND_MARK = '#bnd#'
COLOR_MARK = '#col#'
UNIT_MARK = '#unit#'

CURRENT_SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
COLORS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/colors.txt')
BRANDS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/brands.txt')
VOCABULARY_EN_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/en_dict_cleaned.csv')
VOCABULARY_CZ_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/cz_dict_cleaned.csv')

UNITS_PATH = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/units.tsv')
PREFIXES_PATH = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/prefixes.tsv')
UNITS_IMPERIAL_TO_METRIC_PATH = os.path.join(CURRENT_SCRIPT_FOLDER,
                                             '../../../data/vocabularies/unit_conversion_us-eu.tsv')

SIZE_UNITS = ['XXS', 'XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']


def load_colors():
    """
    Load file with list of colors
    @return: list of loaded colors
    """
    color_list = []
    file = open(COLORS_FILE, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        color_list.append(line)
        if line[len(line) - 1:] == 'á':
            color_list.append(line[:-1] + 'ý')
            color_list.append(line[:-1] + 'é')
    return color_list


def load_brands():
    """
    Load file with list of brands
    @return: list of loaded brands
    """
    brand_list = []
    file = open(BRANDS_FILE, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        brand_list.append(line)
    return brand_list


def load_vocabulary(vocabulary_file):
    """
    Load file with list of vocabulary
    @return: loaded list of words in vocabulary
    """
    with open(vocabulary_file, encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def create_unit_dict():
    """
    Load files with units and their possible prefixes and create all possible units variants and combination
    from shortcuts and different variants of the texts with their values for conversion to their elementary version
    @return: Dataset with units their prefixes and values for conversion to basic form
    """
    prefixes_df = pd.read_csv(PREFIXES_PATH, sep='\t', keep_default_na=False)
    units_df = pd.read_csv(UNITS_PATH, sep='\t', keep_default_na=False)
    units_dict = {}

    for idx, row in units_df.iterrows():

        shortcut_list = row['shortcut'].split(',')
        basic_shortcut = shortcut_list[0]
        units_dict[basic_shortcut] = {'value': 1, 'basic': basic_shortcut}
        if len(shortcut_list) > 1:
            for s in shortcut_list:
                units_dict[s] = {'value': 1, 'basic': basic_shortcut}
        if row['name'] != '':
            units_dict[row['name']] = {'value': 1, 'basic': basic_shortcut}
        if row['plural'] != '':
            units_dict[row['plural']] = {'value': 1, 'basic': basic_shortcut}
        if row['czech'] != '':
            czech_names = row['czech'].split(',')
            for name in czech_names:
                units_dict[name] = {'value': 1, 'basic': basic_shortcut}
        if row['prefixes'] != '':
            for p in row['prefixes'].split(','):
                value = prefixes_df.loc[prefixes_df['prefix'] == p]['value'].values[0]
                for s in shortcut_list:
                    units_dict[f'{p.lower()}{s.lower()}'] = {'value': value, 'basic': basic_shortcut}
                prefix_name = prefixes_df.loc[prefixes_df.prefix == p, "english"].values[0]
                units_dict[f'{prefix_name.lower()}{row["name"].lower()}'] = {'value': value, 'basic': basic_shortcut}
                if row['plural'] != '':
                    units_dict[f'{prefix_name.lower()}{row["plural"].lower()}'] = {'value': value,
                                                                                   'basic': basic_shortcut}
                if row['czech'] != '':
                    czech_names = row['czech'].split(',')
                    for name in czech_names:
                        units_dict[
                            f'{prefixes_df.loc[prefixes_df.prefix == p, "czech"].values[0].lower()}{name.lower()}'] = {
                            'value': value, 'basic': basic_shortcut}
    return units_dict


def load_imperial_to_metric_units_conversion_file():
    """
    Load file that contains data from conversion from us to eu units
    @return: loaded units for conversion
    """
    imperial_to_metric_units_file = pd.read_csv(UNITS_IMPERIAL_TO_METRIC_PATH, sep='\t', keep_default_na=False)
    imperial_to_metric_units_file = imperial_to_metric_units_file.set_index('shortcut').T.to_dict('list')
    return imperial_to_metric_units_file


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


def detect_id(word, next_word):
    """
    Check whether the word is not an id (whether it is a valid word)
    @param word: the word to be checked
    @param next_word: the word following detected word in the text
    @return: word with marker if it is an id, otherwise the original word
    """
    # check whether is capslock ad whether is not too short
    word_sub = re.sub(r"[\W_]+", "", word, flags=re.UNICODE)
    if (not word_sub.isupper() and word_sub.isalpha()) or len(
            word_sub) < ID_LEN or is_in_vocabulary(word):
        return word

    if word_sub.isnumeric() and is_word_unit(next_word):
        word = word.replace("(", "").replace(")", "")
        return ID_MARK + word
    elif word_sub.isalpha():
        if not is_in_vocabulary(word_sub):
            return ID_MARK + word
    else:
        word = word.replace("(", "").replace(")", "")
        if not is_param(word):
            return ID_MARK + word
    return word


def is_word_unit(word):
    """
    Checks whether a word is an unit list
    @param word: checked word
    @return: true if the word is in the dictionary of units
    """
    return word in UNITS_DICT.keys()


def detect_color(word):
    """
    Check whether the word is not in list of colors
    @param word: the word to be checked
    @return: word with marker if it is a color, otherwise the original word
    """
    if word.lower() in COLORS:
        word = COLOR_MARK + word
    return word


def detect_vocabulary_words(word):
    """
    Check whether the word is in vocabulary of words
    @param word: the word to be checked
    @return: word with marker if it is in a vocabulary, otherwise the original word
    """
    if is_in_vocabulary(word.lower()):
        return "#voc#" + word
    return word


def detect_brand(word, is_first, first_likelihood):
    """
    Check whether the word is a brand
    @param word: the word to be checked
    @param is_first: the word to be checked
    @param first_likelihood: the probability that this word is the first one in titles that include it
    @return: word with marker if it is a brand, otherwise the original word
    """
    is_brand = False

    if word.lower() in BRANDS:
        is_brand = True
    elif is_first:
        if (word.isalpha() and len(word) < ID_LEN and word.isupper()) or first_likelihood > 0.9:
            is_brand = True

    return BRAND_MARK + word if is_brand else word


BRANDS = load_brands()
COLORS = load_colors()
VOCABULARY_CZ = load_vocabulary(VOCABULARY_CZ_FILE)
VOCABULARY_EN = load_vocabulary(VOCABULARY_EN_FILE)
UNITS_DICT = create_unit_dict()
UNITS_IMPERIAL_TO_METRIC = load_imperial_to_metric_units_conversion_file()


def detect_ids_brands_colors_and_units(
        data,
        id_detection=True,
        color_detection=True,
        brand_detection=True,
        units_detection=True
):
    """
    Detect ids, colors, brands and units in texts
    @param data: List of texts that each consists of list of words
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @return: texts with detected stuff, eventually number of lemmas from vocabulary and lemmas from morphoditta
    """
    data_list = []
    first_likelihood = compute_likelihood_of_first_words(data)
    for word_list in data:
        detected_word_list = []
        is_first = True
        previous = ''
        for i, word in enumerate(word_list):
            if color_detection:
                word = detect_color(word)
            if brand_detection and not word.startswith(COLOR_MARK):
                word = detect_brand(word, is_first, first_likelihood[word])
            if units_detection and not is_first:
                new_value, word = detect_units(word, previous)
                detected_word_list[len(detected_word_list) - 1] = str(new_value)
            if id_detection:
                next_word = ''
                if i < len(word_list) - 1:
                    next_word = word_list[i + 1]
                word = detect_id(word, next_word)
            detected_word_list.append(word)
            is_first = False
            previous = word
        data_list.append(detected_word_list)

    return data_list


def compute_likelihood_of_first_words(data):
    """
    Compute likelihood that the words appears in the first place
    @param data: dataset with texts that are list of words
    @return: likelihood of each word to be the first one
    """
    word_counts = {}
    first_likelihood = {}
    for word_list in data:
        is_first = True
        for word in word_list:
            if word not in word_counts:
                word_counts[word] = 0
                first_likelihood[word] = 0
            word_counts[word] += 1
            if is_first:
                first_likelihood[word] += 1
            is_first = False
    for word in first_likelihood:
        first_likelihood[word] = first_likelihood[word] / word_counts[word]
    return first_likelihood


def convert_units_to_basic_form(dataset):
    """
    Convert units with prefixes into their basic form.
    @param dataset: List of products each containing list of units
    @return:  List of products each containing list of converted units into their basic form
    """
    converted_dataset = []
    for product in dataset:
        converted_product = []
        for unit in product:
            if is_word_unit(unit[0].lower()):
                name, value = convert_unit_and_value_to_basic_form(unit[0].lower(), unit[1])
                converted_product.append([name.lower(), value])
            else:
                converted_product.append(unit)
        converted_dataset.append(converted_product)
    return converted_dataset


def convert_imperial_to_metric_units(unit, value):
    """
    Convert us unit and value to the eu form
    @param unit: us unit to be converted
    @param value: its value
    @return: eu unit with its value
    """
    if unit in UNITS_IMPERIAL_TO_METRIC.keys():
        value = value * float(UNITS_IMPERIAL_TO_METRIC[unit][0])
        unit = UNITS_IMPERIAL_TO_METRIC[unit][1]
    return unit, value


def convert_unit_and_value_to_basic_form(unit, value):
    """
    Convert unit and value to the basic form
    @param unit: unit name
    @param value: unit value
    @return: basic form of the unit and its recomputed value
    """
    name = UNITS_DICT[unit]['basic']
    value = value * UNITS_DICT[unit]['value']
    return name, value


def detect_units(word, previous_word):
    """
    Check whether the word is not a unit
    @param word: word to be detected
    @param previous_word: previous word needed for detection
    @return: word with marker if it is an unit, otherwise the original word
    """
    if is_word_unit(word.lower()) and previous_word.replace('.', '', 1).isnumeric():
        new_word, new_value = convert_unit_and_value_to_basic_form(word.lower(), float(previous_word))
        new_word, new_value = convert_imperial_to_metric_units(new_word, new_value)
        return new_value, UNIT_MARK + new_word
    if word in SIZE_UNITS:
        return previous_word, "size " + UNIT_MARK + word
    return previous_word, word
