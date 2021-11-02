import os
import re

import pandas as pd

UNITS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/vocabularies/units.tsv')
PREFIXES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/vocabularies/prefixes.tsv')


def load_units_with_prefixes():
    """
    Load vocabulary with units and their prefixes and create all possible units variants and combination
    Moreover create a dictionary of units with prefix wit prefix values to convert all non base units to their elementary version
    @return: Dataset with units and their prefixes; dictionary of units with prefix
    """
    prefixes_df = pd.read_csv(PREFIXES_PATH, sep='\t', keep_default_na=False)
    units_df = pd.read_csv(UNITS_PATH, sep='\t', keep_default_na=False)
    units = pd.DataFrame(columns=units_df.columns)
    prefix_units_convertor = {}
    for idx, row in units_df.iterrows():
        shortcut_list = row['shortcut'].split(',')
        base_shortcut = shortcut_list[0]
        if len(shortcut_list)>1:
            for s in shortcut_list:
                prefix_units_convertor[s] = {'value': 1, 'base': base_shortcut}
        if row['name'] != '':
            prefix_units_convertor[row['name']] = {'value': 1, 'base': base_shortcut}
        if row['plural'] != '':
            prefix_units_convertor[row['plural']] = {'value': 1, 'base': base_shortcut}
        if row['czech'] != '':
            prefix_units_convertor[row['czech']] = {'value': 1, 'base': base_shortcut}
        if row['prefixes'] != '':
            shortcut = row['shortcut'].split(',')
            prefixes = row['prefixes'].split(',')
            base_shortcut = shortcut[0]
            row['base'] = base_shortcut
            name = row['name']
            plural = row['plural']
            czech = row['czech']
            for p in prefixes:
                value = prefixes_df.loc[prefixes_df['prefix'] == p]['value'].values[0]
                for s in shortcut:
                    row['shortcut'] += f',{p}{s}'
                    prefix_units_convertor[f'{p.lower()}{s.lower()}'] = {'value': value, 'base': base_shortcut}
                prefix_name = prefixes_df.loc[prefixes_df.prefix == p, "english"].values[0]
                row['name'] += f',{prefix_name}{name}'
                prefix_units_convertor[f'{prefix_name.lower()}{name.lower()}'] = {'value': value, 'base': base_shortcut}
                if row['plural'] != '':
                    row['plural'] += f',{prefix_name}{plural}'
                    prefix_units_convertor[f'{prefix_name.lower()}{plural.lower()}'] = {'value': value,
                                                                                        'base': base_shortcut}
                if row['czech'] != '':
                    row['czech'] += f',{prefixes_df.loc[prefixes_df.prefix == p, "czech"].values[0]}{czech}'
                    prefix_units_convertor[
                        f'{prefixes_df.loc[prefixes_df.prefix == p, "czech"].values[0].lower()}{czech.lower()}'] = {
                        'value': value, 'base': base_shortcut}
        units = units.append(row)
    return units.iloc[:, :-1], prefix_units_convertor


def create_unit_vocabulary(units):
    """
    Create one list of all units from czech and english names and shortcuts
    @param units: dataframe with english, czech, plural and shortcuts of units
    @return: one list of all units variants
    """
    units_vocabulary = []
    for c in units.columns:
        col_list = units[c].tolist()
        col_words = [word.split(',') for word in col_list]
        col_words = [item.lower() for sublist in col_words for item in sublist if item != '']
        units_vocabulary.append(col_words)
    units_vocabulary = [item for sublist in units_vocabulary for item in sublist]
    return units_vocabulary


def load_units_vocabulary():
    """
    Load vocabulary with units, create list of prefixed units with their values to convert them to basics units
    @return: list of units, list of prefixed units with their values to convert them to basics units
    """
    units, prefix_units_convertor = load_units_with_prefixes()
    return create_unit_vocabulary(units), prefix_units_convertor


UNITS_VOCAB, UNITS_CONVERTOR = load_units_vocabulary()


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
    rgx = re.compile("\w+[\"\-'×.,]?\w*")
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
        if word.lower() in UNITS_VOCAB and previous.replace('.', '', 1).isnumeric():
            new_word, new_value = convert_unit_and_value_to_base_form(word.lower(), float(previous))
            new_word = "#unit#" + new_word
            params.append([new_word, new_value])
            detected_text[len(detected_text)-1]=new_value
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
            if units[0].lower() in UNITS_CONVERTOR:
                name, value = convert_unit_and_value_to_base_form(units[0].lower(), units[1])
                converted_product.append([name.lower(), value])
            else:
                converted_product.append(units)
        converted_dataset.append(converted_product)
    print(converted_dataset)
    return converted_dataset


def convert_unit_and_value_to_base_form(unit, value):
    """
    Convert unit and value to the bacis form
    @param unit: unit name
    @param value: unit value
    @return: basic form of the unit and its recomputed value
    """
    if unit in UNITS_CONVERTOR:
        name = UNITS_CONVERTOR[unit]['base']
        value = value * UNITS_CONVERTOR[unit]['value']
        return name, value
    return unit, value


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
