import json
import os
import re
import time

import pandas as pd
import requests

from ....configuration import MINIMAL_DETECTABLE_ID_LENGTH, \
    KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED

CURRENT_SCRIPT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../data/vocabularies/')
COLORS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'colors.txt')
BRANDS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'brands.json')
VOCABULARY_EN_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'corpus/preprocessed/en_dict_cleaned.csv')
VOCABULARY_CZ_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'corpus/preprocessed/cz_dict_cleaned.csv')

UNITS_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'units.tsv')
PREFIXES_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'prefixes.tsv')
CURRENCIES_FILE = os.path.join(CURRENT_SCRIPT_FOLDER, 'currencies.tsv')
UNITS_IMPERIAL_TO_METRIC_PATH = os.path.join(CURRENT_SCRIPT_FOLDER, 'unit_conversion_us-eu.tsv')
NUMBER_MARK = '#num#'
ID_MARK = '#id#'
BRAND_MARK = '#bnd#'
COLOR_MARK = '#col#'
UNIT_MARK = '#unit#'
MARKS = [ID_MARK, BRAND_MARK, COLOR_MARK, UNIT_MARK, NUMBER_MARK]
SIZE_UNITS = ['XXXS', 'XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']


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
    file = open(BRANDS_FILE, 'r', encoding='utf-8')
    brand_list = json.load(file)
    brand_list_joined = []
    for x in range(len(brand_list)):
        brand_list[x] = brand_list[x].lower()
        brand_list_joined.append(brand_list[x].lower().replace(' ', ''))
    return brand_list, brand_list_joined


def load_vocabulary(vocabulary_file):
    """
    Load file with list of vocabulary
    @return: loaded list of words in vocabulary
    """
    with open(vocabulary_file, encoding='utf-8') as f:
        return [line.rstrip() for line in f]

def create_currencies_dict():
    """
    Load files with currencies and their possible shortcuts and symbols and create dict from them
    @return: Dataset with currencies names (without spaces) and values for conversion to basic form,
             list of original unit names without removing spaces in them
    """
    currencies_df = pd.read_csv(CURRENCIES_FILE, sep='\t', keep_default_na=True)
    currencies_df['merged'] = currencies_df[currencies_df.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )
    currencies_df['name'] = currencies_df['name'].str.lower()
    currencies_df['merged'] = currencies_df['merged'].str.lower()
    currencies_df = currencies_df.drop(columns=['other_name', 'shortcut', 'symbol'])
    currencies_dict = currencies_df.set_index('name').T.to_dict('list')
    original_currencies_names = [k.lower() for k in list(currencies_dict.keys())]

    currencies_merged = currencies_df[currencies_df.columns].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    ).values
    currencies_dict = {}
    for currency in currencies_merged:
        currency_name_list = currency.split(',')
        for currency_name in currency_name_list:
            currencies_dict[currency_name] = {'value':1, 'basic': currency_name}
    return currencies_dict, original_currencies_names


def create_unit_dict():
    """
    Load files with units and their possible prefixes and create all possible units variants and combination
    from shortcuts and different variants of the texts with their values for conversion to their elementary version
    @return: Dataset with units names (without spaces) and values for conversion to basic form,
             list of original unit names without removing spaces in them
    """
    prefixes_df = pd.read_csv(PREFIXES_FILE, sep='\t', keep_default_na=False)
    units_df = pd.read_csv(UNITS_FILE, sep='\t', keep_default_na=False)
    units_dict = {}

    for idx, row in units_df.iterrows():

        shortcut_list = [r.lower() for r in row['shortcut'].split(',')]
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

    original_unit_names = [k for k in list(units_dict.keys())]
    units_dict = {k.replace(' ', ''): v for k, v in units_dict.items()}
    return units_dict, original_unit_names


def load_imperial_to_metric_units_conversion_file():
    """
    Load file that contains data from conversion from us to eu units
    @return: loaded units for conversion
    """
    imperial_to_metric_units_file = pd.read_csv(UNITS_IMPERIAL_TO_METRIC_PATH, sep='\t', keep_default_na=False)
    imperial_to_metric_units_file = imperial_to_metric_units_file.set_index('shortcut').T.to_dict('list')
    return imperial_to_metric_units_file


BRANDS, BRANDS_JOINED = load_brands()
COLORS = load_colors()
VOCABULARY_CZ = load_vocabulary(VOCABULARY_CZ_FILE)
VOCABULARY_EN = load_vocabulary(VOCABULARY_EN_FILE)
CURRENCIES_DICT, ORIGINAL_CURRENCIES_NAMES = create_currencies_dict()
UNITS_DICT, ORIGINAL_UNIT_NAMES = create_unit_dict()
ORIGINAL_UNIT_NAMES = ORIGINAL_UNIT_NAMES + ORIGINAL_CURRENCIES_NAMES
UNITS_DICT = {**UNITS_DICT, **CURRENCIES_DICT}
UNITS_IMPERIAL_TO_METRIC = load_imperial_to_metric_units_conversion_file()


def is_in_vocabulary(word):
    """
    Check whether a word is in the vocabulary which was created manually from corpus
    @param word: a word to be checked
    @return: True if it is an existing word (in one of the vocabularies), otherwise False
    """
    if word in VOCABULARY_CZ or word in VOCABULARY_EN:
        return True
    return False


def is_parameter(word):
    """
    Check whether string is not a parameter
    @param word: the word to be checked
    @return: True if it is a parameter from description, otherwise False
    """
    rgx = re.compile("^[0-9]+[A-Za-z]+$|^[A-Za-z]+[0-9]+$")
    return re.match(rgx, word)


def detect_id(word, following_word):
    """
    Check whether the word is not an id (whether it is a valid word)
    @param word: the word to be checked
    @param following_word: the word following detected word in the text
    @return: word with marker if it is an id, otherwise the original word
    """
    word_sub = re.sub(r"[\W_]+", "", word, flags=re.UNICODE)
    if len(word_sub) < MINIMAL_DETECTABLE_ID_LENGTH or is_in_vocabulary(word) or is_unit_mark_in_word(
            word_sub) or is_brand_mark_in_word(word_sub) or word_sub.islower():
        return word

    word = word.replace("(", "").replace(")", "")
    if is_number(word_sub):
        if not is_unit_mark_in_word(following_word):
            return ID_MARK + word
    elif word_sub.isalpha():
        if not re.match('^[A-Z]{2,}[a-z]*', word_sub):
            return ID_MARK + word
    elif word_sub.isalnum():
        if not is_parameter(word):
            return ID_MARK + word
    else:
        if re.match('^[0-9a-zA-Z]+-?/?.?[0-9a-zA-Z]+$', word_sub):
            return ID_MARK + word
    return word


def is_word_in_unit_list(word):
    """
    Checks whether a word is in the dictionary of unit names
    @param word: checked word
    @return: true if the word is in the dictionary of units
    """
    return word in UNITS_DICT.keys()

def is_unit_mark_in_word(word):
    """
    Checks whether a unit mark is in a word
    @param word: checked word
    @return: true if a unit mark is in a word
    """
    return UNIT_MARK in word


def is_word_in_brand_list(word):
    """
    Checks whether a word is in the dictionary of brands
    @param word: checked word
    @return: true if the word is in the dictionary of brands
    """
    return word.lower() in BRANDS_JOINED


def is_brand_mark_in_word(word):
    """
    Checks whether a brand mark is in a word
    @param word: checked word
    @return: true if a brand mark is in a word
    """
    return BRAND_MARK in word


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
    return "#voc#" + word if is_in_vocabulary(word.lower()) else word


def detect_unspecified_number(word, following_word):
    """
    Check whether the word is numeric and not id, brand, etc. or unit value
    @param word: the word to be checked
    @param following_word: word following the detected word
    @return: word with marker if it is in an unspecified number, otherwise the original word
    """
    if is_number(word):
        if not is_unit_mark_in_word(following_word):
            return NUMBER_MARK + word
    return word


def is_number(word):
    """
    Check whether given word is a number
    @param word: string to be checked
    @return: true if string is a number else false
    """
    try:
        float(word)
        return True
    except ValueError:
        return False


def detect_brand(word_list):
    """
    Search for the brands in the list of words
    @param word_list: list if words to be detected
    @return: word_list with marked words which are the brands, list of detected brands
    """
    # detect units with more words separated by the space in the name and merge them into single word without spaces
    word_list_joined = ' '.join(word_list)
    detected_brand_list = []
    for value in BRANDS:
        word_list_joined = word_list_joined.replace(value, value.replace(' ', ''))
    word_list = word_list_joined.split(' ')

    detected_word_list = []
    for word in word_list:
        if word in BRANDS_JOINED:
            detected_brand_list.append(word)
            word = BRAND_MARK + word
        detected_word_list.append(word)

    return detected_word_list, detected_brand_list


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


def detect_units(word_list):
    """
    Search for the units and values in the list of words and convert them into their basic forms
    @param word_list: list if words to be detected
    @return: word_list with marked words which are the units, list of detected units and values
    """
    # detect units with more words separated by the space in the name and merge them into single word without spaces
    word_list_joined = ' '.join(word_list)
    for value in ORIGINAL_UNIT_NAMES:
        word_list_joined = word_list_joined.replace(value, value.replace(' ', ''))
    word_list = word_list_joined.split(' ')

    # detect units and values
    detected_units_list = []
    detected_word_list = []
    previous_word = ''
    for i, word in enumerate(word_list):
        if previous_word == '':
            detected_word_list.append(word)
            if previous_word in SIZE_UNITS:
                detected_word_list.append(UNIT_MARK + 'size')
                detected_units_list.append(['size', previous_word])
            previous_word = word
            continue
        if is_word_in_unit_list(word.lower()) and previous_word.replace(',', '', 1).replace('.', '', 1).isnumeric():
            new_word, new_value = convert_unit_and_value_to_basic_form(word.lower(),
                                                                       float(
                                                                           previous_word.replace(',', '.', 1)
                                                                       )
                                                                       )
            new_word, new_value = convert_imperial_to_metric_units(new_word, new_value)
            detected_word_list.pop()
            detected_word_list.append(str(new_value))
            detected_word_list.append(UNIT_MARK + new_word)
            previous_word = word
            detected_units_list.append([new_word, new_value])
        elif word in SIZE_UNITS:
            previous_word = word
            detected_word_list.append(word)
            detected_word_list.append(UNIT_MARK + "size")
            detected_units_list.append(['size', previous_word])
        elif is_word_in_unit_list(word.lower()) and '×' in previous_word and all(
                is_number(subword) for subword in previous_word.split('×')):
            converted_value_list = []
            new_word = word
            for value in previous_word.split('×'):
                new_word, new_value = convert_unit_and_value_to_basic_form(word.lower(),
                                                                           float(value.replace(',', '.', 1)))
                new_word, new_value = convert_imperial_to_metric_units(new_word, new_value)
                converted_value_list.append(new_value)
            converted_value_list = [str(item) for item in converted_value_list]
            converted_value = '×'.join(converted_value_list)
            detected_word_list.pop()
            detected_word_list.append(converted_value)
            detected_word_list.append(UNIT_MARK + new_word)
            previous_word = word
            detected_units_list.append([new_word, converted_value])
        else:
            previous_word = word
            detected_word_list.append(word)

    return detected_word_list, detected_units_list


def detect_ids_brands_colors_and_units(
        data,
        id_detection=True,
        color_detection=True,
        brand_detection=True,
        units_detection=True,
        numbers_detection=True
):
    """
    Detect ids, colors, brands and units in texts
    @param data: List of texts that each consists of list of words
    @param id_detection: True if ids should be detected
    @param color_detection: True if colors should be detected
    @param brand_detection: True if brands should be detected
    @param units_detection: True if units should be detected
    @param numbers_detection: True if unspecified numbers should be detected
    @return: texts with detected stuff, eventually number of lemmas from vocabulary and lemmas from morphoditta
    """
    data_list = []
    detected_keywords_list = []
    for word_list in data:
        detected_word_list = []
        detected_keywords = {}

        detected_keywords_in_text = {}
        if id_detection:
            detected_keywords_in_text['all_ids_list'] = []
        if numbers_detection:
            detected_keywords_in_text['all_numbers_list'] = []

        # detect units
        if units_detection:
            detected_word_list, detected_unit_list = detect_units(word_list)
            detected_keywords['all_units_list'] = detected_unit_list

        # detect ids and colors and unspecified numbers
        following_word = ''
        detected_ids_in_text = []
        detected_numbers_in_text = []
        for i, word in enumerate(detected_word_list):
            if i < len(detected_word_list) - 1:
                following_word = detected_word_list[i + 1]
            if i == len(detected_word_list) - 1:
                following_word = ''
            if color_detection:
                word = detect_color(word)
            if id_detection:
                word = detect_id(word, following_word)
                if ID_MARK in word:
                    detected_ids_in_text.append(word.replace(COLOR_MARK, ''))
            if numbers_detection:
                word = detect_unspecified_number(word, following_word)
                if NUMBER_MARK in word:
                    detected_numbers_in_text.append(word.replace(COLOR_MARK, ''))
            detected_word_list[i] = word
        detected_keywords['all_ids_list'] = detected_ids_in_text
        detected_keywords['all_numbers_list'] = detected_numbers_in_text
        # detect brands
        if brand_detection:
            detected_word_list, detected_brand_list = detect_brand(detected_word_list)
            detected_keywords['all_brands_list'] = detected_brand_list

        data_list.append(detected_word_list)
        detected_keywords_list.append(detected_keywords)
    return data_list, detected_keywords_list


def update_keywords_detection_from_config(column, id_detection, brand_detection, color_detection, numbers_detection,
                                          units_detection):
    """
    Update keywords detection according to the config file
    @param column: column in dataset in which to detect keywords or not
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @param numbers_detection: True if unspecified numbers should be detected
    @return: updated booleans whether to detects keywords or not
    """
    if column in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED.keys():
        if 'id' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]:
            id_detection = False
        if 'brand' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]:
            brand_detection = False
        if 'color' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]:
            color_detection = False
        if 'words' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]:
            id_detection = False
        if 'units' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]:
            units_detection = False
        if 'numbers' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]:
            numbers_detection = False
    return brand_detection, color_detection, id_detection, numbers_detection, units_detection


def convert_keyword_dicts_to_dataframe(detected_keywords_df):
    """
    Convert dataframe with products features as dictionaries to
    dataframe with all detected keywords from all texts as arrays
    @param detected_keywords_df: dataframe with products features as columns
    @return: dataframe with all detected keywords from all texts as columns
    """
    detected_keywords_list = []
    for _, row in detected_keywords_df.iterrows():
        dict_list = [d[1] for d in row.iteritems()]
        merged_detected_keywords = {}
        for item in dict_list:
            for key in item.keys():
                if key not in merged_detected_keywords:
                    merged_detected_keywords[key] = []
                merged_detected_keywords[key] += item[key]
        merged_detected_keywords_df = pd.DataFrame([merged_detected_keywords])
        detected_keywords_list.append(merged_detected_keywords_df)

    detected_keywords = pd.concat(detected_keywords_list)

    return detected_keywords


def reindex_and_merge_dataframes(dataset, detected_keywords):
    """
    Reindex dataframe with extracted keywords and merge them with other data
    @param dataset: dataframe with textual data
    @param detected_keywords: dataframe with extracted keywords
    @return: merged dataframe with keywords
    """
    detected_keywords['index'] = [x for x in dataset.index]
    detected_keywords = detected_keywords.set_index('index')
    dataset = pd.concat([dataset, detected_keywords], axis=1)
    return dataset


# NOT USED - BUT MIGHT BE USEFUL FOR EXPERIMENTS INSTEAD OF VOCABULARIES
def does_this_word_exist(lemma):
    """
    Check whether the word is in Czech or English vocabulary in LINDAT repository
    @param lemma: the word to be checked
    @return: True if it is an existing word, otherwise False
    """
    while True:
        try:
            url_cz = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={lemma}\
                    &output=json&guesser=no&model=czech-morfflex-pdt-161115"
            url_en = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={lemma}\
                    &output=json&guesser=no&model=english-morphium-wsj-140407"

            r_cz = json.loads(requests.get(url_cz).text)['result']
            r_en = json.loads(requests.get(url_en).text)['result']

            if not r_cz or not r_en:
                return False

            if r_cz[0][0]['tag'] == 'X@-------------' and r_en[0][0]['tag'] == 'UNK':
                return False
            return True
        except:
            time.sleep(1)
