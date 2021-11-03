import os
import re

import pandas as pd

UNITS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/vocabularies/units.tsv')
PREFIXES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/vocabularies/prefixes.tsv')
UNITS_US_TO_EU_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   '../../../data/vocabularies/unit_conversion_us-eu.tsv')

SIZE_UNITS = ['XXS', 'XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']


def create_unit_dict():
    """
    Load files with units and their possible prefixes and create all possible units variants and combination
    from shortcuts and different variants of the names with their values for conversion to their elementary version
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


UNITS_DICT = create_unit_dict()
UNITS_US_TO_EU = load_us_to_eu_units_conversion_file()


def split_params(text):
    """
    Split text to single parameteres separated by comma
    @param text: input text
    @return: split parameters
    """
    return text.split(',')


def remove_useless_spaces(text):
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


def split_words(text_list):
    """
    Split list of specifications to the single words
    @param text: list of specifications to be split
    @return: list of words of specifications
    """
    split_text = []
    rgx = re.compile("\w+[\"\-'×.,%]?\w*")
    for text in text_list:
        words = rgx.findall(text)
        split_text.append(words)
    return split_text


def detect_parameters(text):
    """
    Detect units in text according to the loaded dictionary
    @param text: text to detect parameters and units
    @return: text with detected parameters, separated parameters and values
    """
    params = []
    detected_text = []
    previous = ''
    for word in text:
        new_word = word
        if word.lower() in UNITS_DICT.keys() and previous.replace('.', '', 1).isnumeric():
            new_word, new_value = convert_unit_and_value_to_base_form(word.lower(), float(previous))
            new_word, new_value = convert_us_to_eu_units(new_word, new_value)
            params.append([new_word, new_value])
            new_word = "#unit#" + new_word
            detected_text[len(detected_text) - 1] = new_value
        if word in SIZE_UNITS:
            params.append(['size', new_word])
            new_word = "#unit#" + new_word
        detected_text.append(new_word)
        previous = word
    return detected_text, params


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
    if unit in UNITS_US_TO_EU.keys():
        x = UNITS_US_TO_EU[unit][0]
        value = value * float(UNITS_US_TO_EU[unit][0])
        unit = UNITS_US_TO_EU[unit][1]
    return unit, value


def convert_unit_and_value_to_base_form(unit, value):
    """
    Convert unit and value to the bacis form
    @param unit: unit name
    @param value: unit value
    @return: basic form of the unit and its recomputed value
    """
    name = UNITS_DICT[unit]['base']
    value = value * UNITS_DICT[unit]['value']
    return name, value


def compare_units_in_descriptions(dataset1, dataset2, devation=0.05):
    """
    Compare detected units from the texts
    @param dataset1: List of products each containing list of units from the first dataset
    @param dataset2: List of products each containing list of units from the second dataset
    @param devation: percent of toleration of deviations of two compared numbers
    @return: Ratio of the same units between two products
    """
    similarity_scores = []
    dataset1 = convert_units_to_basic_form(dataset1)
    dataset2 = convert_units_to_basic_form(dataset2)
    for i, description1 in enumerate(dataset1):
        for j, description2 in enumerate(dataset2):
            description1_set = set(tuple(x) for x in description1)
            description2_set = set(tuple(x) for x in description2)
            # matches = len(description1_set.intersection(description2_set))
            matches = 0
            for d1 in description1_set:
                for d2 in description2_set:
                    if d1[0] == d2[0] and d1[1] > (1 - devation) * d2[1] and d1[1] < (1 + devation) * d2[1]:
                        matches += 1
            if not len(description2_set) == 0:
                match_ratio = matches / len(description2_set)
                similarity_scores.append([i, j, match_ratio])
            else:
                similarity_scores.append([i, j, 0])
    return similarity_scores
