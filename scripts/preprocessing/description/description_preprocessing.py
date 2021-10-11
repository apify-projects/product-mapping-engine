import re

import pandas as pd

from scripts.preprocessing.names.names_preprocessing import detect_ids_brands_and_colors

UNITS_PATH = 'data/vocabularies/units.tsv'
PREFIXES_PATH = 'data/vocabularies/prefixes.tsv'

def split_params(text):
    """
    Split text to single parameteres separated by comma
    @param text: input text
    @return: split parameters
    """
    return text.split(',')


def remove_useless_spaces(text):
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


def load_units_with_prefixes():
    """
    Load vocabulary with units and their prefixes and create all possible units variants and combination
    @return: Dataset with units and their prefixes
    """
    prefixes_df = pd.read_csv(PREFIXES_PATH, sep='\t', keep_default_na=False)
    units_df = pd.read_csv(UNITS_PATH, sep='\t', keep_default_na=False)
    units = pd.DataFrame(columns=units_df.columns)
    for idx, row in units_df.iterrows():
        if row['prefixes'] != '':
            shortcut = row['shortcut'].split(',')
            prefixes = row['prefixes'].split(',')
            name = row['name']
            plural = row['plural']
            czech = row['czech']
            for p in prefixes:
                for s in shortcut:
                    row['shortcut'] += f',{p}{s}'
                prefix_name = prefixes_df.loc[prefixes_df.prefix == p, "english"].values[0]
                row['name'] += f',{prefix_name}{name}'
                if row['plural'] != '':
                    row['plural'] += f',{prefix_name}{plural}'
                if row['czech'] != '':
                    row['czech'] += f',{prefixes_df.loc[prefixes_df.prefix == p, "czech"].values[0]}{czech}'
        units = units.append(row)
    return units.iloc[:, :-1]


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


def detect_parameters(text):
    """
    Detect units in text according to the loaded dictionary
    @param text: text to detect parameters and units
    @return: text with detected parameters, separated parameters and values
    """
    params = []
    units = load_units_with_prefixes()
    unit_vocab = create_unit_vocabulary(units)
    detected_text = []
    for sentence in text:
        new_sentence = []
        previous = ''
        for word in sentence:
            word_new = word
            if word in unit_vocab and previous.replace('.', '', 1).isnumeric():
                word_new = "#UNIT#" + word
                params.append([word, float(previous)])
            new_sentence.append(word_new)
            previous = word
        detected_text.append(new_sentence)
    return detected_text, params



def compare_units_in_descriptions(dataset1, dataset2):
    similarity_scores = []
    for i, description1 in enumerate(dataset1):
        for j, description2 in enumerate(dataset2):
            description1_set = set(tuple(x) for x in description1)
            description2_set = set(tuple(x) for x in description2)
            matches = description1_set.intersection(description2_set)
            match_ratio = len(matches)/len(description2_set)

            similarity_scores.append([i, j, match_ratio])
    return similarity_scores