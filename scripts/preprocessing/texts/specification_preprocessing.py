import json
import re

from .keywords_detection import detect_ids_brands_colors_and_units
from .text_preprocessing import preprocess_text


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
    @param dataset: json of key-value pairs of products specifications
    @return: list of dicts containing parsed products specifications
    """
    parsed_dataset = []

    for product_specification in dataset:
        product_specification = json.loads(product_specification)
        specification_dict = {}
        for item in product_specification:
            item['key'] = re.sub('\n', '', item['key'])
            item['value'] = re.sub('\n', '', item['value'])
            item['key'] = re.sub('\t', '', item['key'])
            item['value'] = re.sub('\t', '', item['value'])
            specification_dict[item['key']] = item['value']
        parsed_dataset.append(specification_dict)
    return parsed_dataset
