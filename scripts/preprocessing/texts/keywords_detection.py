import json
import os
import re
import time

import pandas as pd
import requests

ID_LEN = 5
COLOR_PREFIX = '#col#'
CURRENT_SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
COLORS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/colors.txt')
BRANDS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, '../../../data/vocabularies/brands.txt')
VOCABULARY_EN_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/en_dict_cleaned.csv')
VOCABULARY_CZ_FILE = os.path.join(CURRENT_SCRIPT_FOLDER,
                                  '../../../data/vocabularies/corpus/preprocessed/cz_dict_cleaned.csv')

UNITS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/vocabularies/units.tsv')
PREFIXES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/vocabularies/prefixes.tsv')
UNITS_US_TO_EU_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
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
    @return: Dataset with units their prefixes and values for conversion to bae form
    """
    prefixes_df = pd.read_csv(PREFIXES_PATH, sep='\t', keep_default_na=False)
    units_df = pd.read_csv(UNITS_PATH, sep='\t', keep_default_na=False)
    units_dict = {}

    for idx, row in units_df.iterrows():

        shortcut_list = row['shortcut'].split(',')
        base_shortcut = shortcut_list[0]
        if '#' in base_shortcut:
            base_shortcut = base_shortcut.replace('#', '')
        units_dict[base_shortcut] = {'value': 1, 'base': base_shortcut}
        if len(shortcut_list) > 1:
            for s in shortcut_list:
                units_dict[s] = {'value': 1, 'base': base_shortcut}
        if row['name'] != '':
            units_dict[row['name']] = {'value': 1, 'base': base_shortcut}
        if row['plural'] != '':
            units_dict[row['plural']] = {'value': 1, 'base': base_shortcut}
        if row['czech'] != '':
            units_dict[row['czech']] = {'value': 1, 'base': base_shortcut}
        if row['prefixes'] != '':
            for p in row['prefixes'].split(','):
                value = prefixes_df.loc[prefixes_df['prefix'] == p]['value'].values[0]
                for s in shortcut_list:
                    units_dict[f'{p.lower()}{s.lower()}'] = {'value': value, 'base': base_shortcut}
                prefix_name = prefixes_df.loc[prefixes_df.prefix == p, "english"].values[0]
                units_dict[f'{prefix_name.lower()}{row["name"].lower()}'] = {'value': value, 'base': base_shortcut}
                if row['plural'] != '':
                    units_dict[f'{prefix_name.lower()}{row["plural"].lower()}'] = {'value': value,
                                                                                   'base': base_shortcut}
                if row['czech'] != '':
                    units_dict[
                        f'{prefixes_df.loc[prefixes_df.prefix == p, "czech"].values[0].lower()}{row["czech"].lower()}'] = {
                        'value': value, 'base': base_shortcut}
    return units_dict


def load_us_to_eu_units_conversion_file():
    """
    Load file that contains data from conversion from us to eu units
    @return: loaded units for conversion
    """
    us_to_eu_units = pd.read_csv(UNITS_US_TO_EU_PATH, sep='\t', keep_default_na=False)
    us_to_eu_units = us_to_eu_units.set_index('shortcut').T.to_dict('list')
    return us_to_eu_units


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
    @return: word with marker if it is a brand, otherwise the original word
    """
    is_brand = False

    if word.lower() in BRANDS:
        is_brand = True
    elif is_first:
        if (word.isalpha() and len(word) < ID_LEN and word.isupper()) or first_likelihood > 0.9:
            is_brand = True

    return "#bnd#" + word if is_brand else word


BRANDS = load_brands()
COLORS = load_colors()
VOCABULARY_CZ = load_vocabulary(VOCABULARY_CZ_FILE)
VOCABULARY_EN = load_vocabulary(VOCABULARY_EN_FILE)
UNITS_DICT = create_unit_dict()
UNITS_US_TO_EU = load_us_to_eu_units_conversion_file()


def detect_ids_brands_colors_and_units(data, id_detection=True, color_detection=True,
                                       brand_detection=True,
                                       units_detection=True):
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
    data_parameters = []
    for word_list in data:
        description_parameters = []
        detected_word_list = []
        is_first = True
        previous = ''
        for word in word_list:
            if color_detection:
                word = detect_color(word)
            if brand_detection and not word.startswith(COLOR_PREFIX):
                word = detect_brand(word, is_first, first_likelihood[word])
            if id_detection:
                word = detect_id(word)
            if units_detection and not is_first:
                word, unit_and_value = detect_units(word, previous)
                if unit_and_value is not None:
                    detected_word_list[len(detected_word_list) - 1] = str(unit_and_value[1])
                    description_parameters.append(unit_and_value)
            detected_word_list.append(word)
            is_first = False
            previous = word
        data_parameters.append(description_parameters)
        data_list.append(detected_word_list)

    return data_list, data_parameters


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
    Convert units with prefixes into their basics form.
    @param dataset: List of products each containing list of units
    @return:  List of products each containing list of converted units into their basic form
    """
    converted_dataset = []
    for product in dataset:
        converted_product = []
        for units in product:
            if units[0].lower() in UNITS_DICT.keys():
                name, value = convert_unit_and_value_to_base_form(units[0].lower(), units[1])
                converted_product.append([name.lower(), value])
            else:
                converted_product.append(units)
        converted_dataset.append(converted_product)
    return converted_dataset


def convert_us_to_eu_units(unit, value):
    """
    Convert us unit and value to the eu form
    @param unit: us unit to be converted
    @param value: its value
    @return: eu unit with its value
    """
    if unit in UNITS_US_TO_EU.keys():
        value = value * float(UNITS_US_TO_EU[unit][0])
        unit = UNITS_US_TO_EU[unit][1]
    return unit, value


def convert_unit_and_value_to_base_form(unit, value):
    """
    Convert unit and value to the basics form
    @param unit: unit name
    @param value: unit value
    @return: basic form of the unit and its recomputed value
    """
    name = UNITS_DICT[unit]['base']
    value = value * UNITS_DICT[unit]['value']
    return name, value


def detect_units(word, previous_word):
    """
    Check whether the word is not a unit
    @param word: word to be detected
    @param previous_word: previous word needed for detection
    @return: word with marker if it is a brand, otherwise the original word
    """
    if word.lower() in UNITS_DICT.keys() and previous_word.replace('.', '', 1).isnumeric():
        new_word, new_value = convert_unit_and_value_to_base_form(word.lower(), float(previous_word))
        new_word, new_value = convert_us_to_eu_units(new_word, new_value)
        return "#unit#" + new_word, [new_word, new_value]
    if word in SIZE_UNITS:
        return "#unit#" + word, ['size', word]
    return word, None
