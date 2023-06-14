import copy
import json
import os
import re
import unicodedata
import majka
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer

from .keywords_detection import detect_ids_brands_colors_and_units, update_keywords_detection_from_config, \
    convert_keyword_dicts_to_dataframe, reindex_and_merge_dataframes
from ....configuration import LOWER_CASE_TEXT, LANGUAGE, COLUMNS_TO_BE_PREPROCESSED


def set_czech_lemmatizer():
    """
    Set lemmatizer for Czech language
    @return: lemmatizer
    """
    lemmatizer = majka.Majka(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../data/vocabularies/majka.w-lt'))
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
        if not x:
            lemma = word
        else:
            lemma = x[0]['lemma']
            if 'negation' in x[0]['tags'] and x[0]['tags']['negation']:
                lemma = 'ne' + lemma
        lemmatized_text.append(lemma)
    return lemmatized_text


def lemmatize_english_text(text):
    """
    Lemmatize English text
    @param text: text to lemmatize
    @return: lemmatized text
    """
    stemmer = SnowballStemmer("english")
    lemmatized_text = []
    for word in text:
        lemmatized_text.append(stemmer.stem(word))
    return lemmatized_text


def remove_useless_spaces_and_characters(text):
    """
    Remove useless spaces between numerical values
    @param text: text to remove spaces
    @return: text without useless spaces
    """
    text = re.sub(r'(?<=\d) - (?=\d)', r'-', text)
    text = re.sub(r'(?<=\d),(?=\d)', r'', text)
    text = re.sub(r'(?<=\d)"', r' inch', text)
    text = re.sub(r'(?<=\d)x(?=\d)', r'×', text)
    # text = re.sub(r'(?<=\d)x', r'', text)
    text = text.replace(' × ', '×')
    text = text.replace('½', '1/2')
    text = text.replace('⅓', '1/3')
    text = text.replace('¼', '1/4')
    text = text.replace('⅕', '1/5')
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text


def tokenize(text):
    """
    Split text to the single words
    @param text: string text to be split
    @return: list of words of text
    """
    rgx = re.compile("\w[\"-'×.,%]?\w*")
    word_list = rgx.findall(text)
    return word_list


def split_units_and_values(word_list):
    """
    Split parameter values and units into two words
    @param word_list: list of words
    @return: list of words with split units
    """
    word_list_split = []
    for word in word_list:
        # if word in the form: number+string (eg:12kb, 1.2kb)
        if re.match('^([0-9]*[.])?[0-9]+[a-zA-Z]+$', word) is not None:
            word_list_split.append(re.split('[a-zA-Z]+$', word)[0])
            split = re.split('^([0-9]*[.])?[0-9]+', word)
            word_list_split.append(split[len(split) - 1])
        # if word in the form: number+nonstring_unit (eg: 50°C,70.5°F, 14")
        elif re.match('^([0-9]*[.])?[0-9]+[{°C}{°F}°%£€$Ω\"\']+$', word) is not None:
            word_list_split.append(re.split('[{°C}{°F}°%£€$Ω\"\']+$', word)[0])
            split = re.split('^([0-9]*[.])?[0-9]+', word)
            word_list_split.append(split[len(split) - 1])
        # if word in the form: number-number+string (eg: 10-15h)
        elif re.match('^([0-9]*[.])?[0-9]+-([0-9]*[.])?[0-9]+[a-zA-Z]+$', word) is not None:
            word_list_split.append(re.split('[a-zA-Z]+$', word)[0])
            split = re.split('^([0-9]*[.])?[0-9]+-([0-9]*[.])?[0-9]+', word)
            word_list_split.append(split[len(split) - 1])
        # if word in the form: number×number×number+string (eg: 10×10×10cm)
        elif re.match('^([0-9]+×[0-9]+(×[0-9])*)[a-zA-Z]+$', word) is not None:
            word_list_split.append(re.split('[a-zA-Z]+$', word)[0])
            split = re.split('^([0-9]+×[0-9]+(×[0-9])*)', word)
            word_list_split.append(split[len(split) - 1])
        else:
            word_list_split.append(word)
    return word_list_split


def lower_case(word_list):
    """
    Lower case all words in the list of words
    @param word_list: list of words
    @return: lower cased list of words
    """
    lowercase_word_list = []
    for word in word_list:
        lowercase_word_list.append(word.lower())
    return lowercase_word_list


def preprocess_text(data):
    """
    Lowercase and split units and values in dataset, then do the lemmatization and stemization
    @param data: list of texts to preprocess
    @return: preprocessed list of texts that consists of list of words
    """
    new_data = []
    lemmatizer = set_czech_lemmatizer()
    for text in data:
        text = remove_useless_spaces_and_characters(text)
        word_list = tokenize(text)
        word_list = split_units_and_values(word_list)
        if LOWER_CASE_TEXT:
            word_list = lower_case(word_list)
        if LANGUAGE == 'czech':
            word_list = lemmatize_czech_text(word_list, lemmatizer)
        if LANGUAGE == 'english':
            word_list = lemmatize_english_text(word_list)
        new_data.append(word_list)
    return new_data


def add_all_texts_column(dataset):
    """
    Add a column containing a concatenation of all other text columns to the dataset
    @param dataset: dataframe in which to join text columns
    @return: dataframe with additional column containing all texts for each product
    """

    columns = [col for col in dataset.columns if '_no_detection' in col]
    dataset_subset = dataset[columns]
    joined_rows = []
    for _, row in dataset_subset.iterrows():
        joined_rows.append([r for v in row.values for r in v])
    dataset['all_texts'] = joined_rows
    return dataset

def preprocess_textual_data(dataset,
                            id_detection=True,
                            color_detection=True,
                            brand_detection=True,
                            units_detection=True,
                            numbers_detection=True):
    """
    Preprocessing of all textual data in dataset column by column
    @param dataset: dataset to be preprocessed
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @param numbers_detection: True if unspecified numbers should be detected
    @return preprocessed dataset
    """
    dataset['price'] = pd.to_numeric(dataset['price'])
    detected_keywords_df = pd.DataFrame()
    dataset = parse_specifications_and_create_copies(dataset, 'specification')
    for column in COLUMNS_TO_BE_PREPROCESSED:
        if column in dataset:
            dataset[column] = dataset[column].fillna("")
            dataset[column] = preprocess_text(dataset[column].values)
            dataset[column + '_no_detection'] = copy.deepcopy(dataset[column])
            column_brand_detection, column_color_detection, column_id_detection, column_numbers_detection, column_units_detection = \
                update_keywords_detection_from_config(
                    column, id_detection, brand_detection, color_detection, numbers_detection,
                    units_detection
                )
            dataset[column], detected_keywords_df[column] = detect_ids_brands_colors_and_units(
                dataset[column],
                column_id_detection,
                column_color_detection,
                column_brand_detection,
                column_units_detection,
                column_numbers_detection
            )

    dataset = add_all_texts_column(dataset)
    detected_keywords = convert_keyword_dicts_to_dataframe(detected_keywords_df)

    if 'specification' in dataset.columns:
        dataset['specification'] = preprocess_specifications(dataset['specification'])

    if 'code' in dataset.columns:
        # Standardize format
        dataset['code'] = dataset['code'].apply(
            lambda code: code if isinstance(code, str) else json.dumps(code)
        )

        dataset['code'] = dataset['code'].apply(
            lambda code_string: code_string.replace(' ,', ',').replace(', ', ',').replace('[', '').replace(']', '').replace('"', '').replace("'", "").split(',')
        )

    dataset = reindex_and_merge_dataframes(dataset, detected_keywords)
    return dataset


def multi_run_text_preprocessing_wrapper(args):
    """
    Wrapper for passing more arguments to preprocess_textual_data in parallel way
    @param args: Arguments of the function
    @return: call the preprocess_textual_data in parallel way
    """
    return preprocess_textual_data(*args)


def parallel_text_preprocessing(pool, num_cpu, dataset, id_detection, color_detection, brand_detection,
                                units_detection, numbers_detection):
    """
    Preprocessing of all textual data in dataset in parallel way
    @param pool: parallelling object
    @param num_cpu: number of processes
    @param dataset: dataframe to be preprocessed
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @param numbers_detection: True if unspecified numbers should be detected
    @return preprocessed dataset
    """
    dataset_list = np.array_split(dataset, num_cpu)
    dataset_list_preprocessed = pool.map(multi_run_text_preprocessing_wrapper,
                                         [(item, id_detection, color_detection, brand_detection, units_detection,
                                           numbers_detection) for item in
                                          dataset_list])
    dataset_preprocessed = pd.concat(preprocessed_data for preprocessed_data in dataset_list_preprocessed)
    return dataset_preprocessed


def convert_specifications_to_texts(dataset):
    """
    Convert specifications to list of strings for similarity computations by the same way as all other textual data
    @param dataset: list of product specifications where each consist of dict of parameter name and value
    @return: List of preprocessed texts containing whole specification as one long string
    """
    joined_dataset = []
    for product_specification in dataset:
        text = ''
        for name, value in product_specification.items():
            text += f'{name} {value} '
        joined_dataset.append(text)
    return joined_dataset


def preprocess_specifications(dataset):
    """
    Preprocess specifications for further similarity computations - separate parameter name and value
    @param dataset: list of dicts containing parsed products specifications
    @return: list of dicts containing parsed and preprocessed products specifications
    """
    preprocessed_dataset = []
    for product_specification in dataset:
        specification_dict_preprocessed = {}
        for name, value in product_specification.items():
            name = preprocess_text([name])
            value = preprocess_text([value])
            text_detected, _ = detect_ids_brands_colors_and_units(
                value,
                id_detection=False,
                color_detection=False,
                brand_detection=False,
                units_detection=True
            )
            specification_dict_preprocessed[' '.join(name[0])] = ' '.join(text_detected[0])

        preprocessed_dataset.append(specification_dict_preprocessed)
    return preprocessed_dataset


def parse_specifications(dataset):
    """
    Parse specifications from input json to list of dictionaries of separated parameter name and value
    @param dataset: json of key-value pairs of products specifications or dictionary
    @return: list of dicts containing parsed products specifications
    """
    parsed_dataset = []

    for product_specification in dataset:
        if not isinstance(product_specification, list):
            if not product_specification:
                product_specification = []
            else:
                if type(product_specification) == dict:
                    product_specification_dict_format = product_specification
                    product_specification = []
                    for key, value in product_specification_dict_format.items():
                        product_specification.append({
                            "key": key,
                            "value": value
                        })
                else:
                    product_specification = json.loads(product_specification)

        specification_dict = {}
        for item in product_specification:
            if not 'value' in item:
                item['value'] = ""

            item['value'] = str(item['value'])
            item['key'] = re.sub('\n', '', item['key'])
            item['value'] = re.sub('\n', '', item['value'])
            item['key'] = re.sub('\t', '', item['key'])
            item['value'] = re.sub('\t', '', item['value'])
            specification_dict[item['key']] = item['value']
        parsed_dataset.append(specification_dict)
    return parsed_dataset


def parse_specifications_and_create_copies(dataset, specification_name):
    """
    Parse specification from json to dict and create copies of them converted to classical text
    @param dataset: dataframe with products
    @param specification_name: name of the specification column
    @return: dataframe with products with parsed specifications and new columns of specifications converted to text
    """
    if specification_name in dataset.columns:
        dataset[specification_name] = parse_specifications(dataset[specification_name])
        specification_name_text = f'{specification_name}_text'
        if specification_name[-1] in ['1', '2']:
            specification_name_text = f'{specification_name[:-1]}_text{specification_name[-1]}'

        dataset[specification_name_text] = convert_specifications_to_texts(
            copy.deepcopy(dataset[specification_name].values))
    return dataset
