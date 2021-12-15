import re
from difflib import SequenceMatcher

from scripts.preprocessing.texts.keywords_detection import UNIT_MARK
from scripts.preprocessing.texts.specification_preprocessing import preprocess_specifications


def preprocess_specifications_and_compute_similarity(dataset1, dataset2):
    """
    Preprocess two datasets of specifications and compute their similarity
    @param dataset1: first list of list of key-value pairs of products specifications
    @param dataset2: second list of list of key-value pairs of products specifications
    @return: similarity of specifications
    """
    dataset1 = preprocess_specifications(dataset1)
    dataset2 = preprocess_specifications(dataset2)
    similarity_score = compute_similarity_of_specifications(dataset1, dataset2)
    return similarity_score


def compute_similarity_of_specifications(dataset1, dataset2):
    """
    Compare two specifications and find common attributes with the same values
    @param dataset1: first dictionary of specification parameter names and values
    @param dataset2: second dictionary of specification parameter names and values
    @return: ratio of common attributes with the same values
    """
    similarity_scores = []
    for product1 in dataset1:
        for product2 in dataset2:
            similarities_dict = find_closest_keys(product1, product2, key_similarity_limit=0.9)
            matching_keys, matching_keys_and_values = compare_values_similarity(
                similarities_dict,
                number_similarity_deviation=0.1,
                string_similarity_deviation=0.1
            )
            similarity_scores.append({'matching_keys': matching_keys / len(product1),
                                      'matching_keys_values': matching_keys_and_values / len(product1)})
    return similarity_scores


def find_closest_keys(dictionary1, dictionary2, key_similarity_limit):
    """
    Find corresponding parameter pairs from both dictionaries according to their key similarity
    @param dictionary1: first dictionary
    @param dictionary2: second dictionary
    @param key_similarity_limit: percentage limit how much the keys must be similar
    @return: dictionary with keys and values from first dictionary supplemented by corresponding values for the same parameter names from the second dictionary
    """
    similarities_dict = {}
    for key1, value1 in dictionary1.items():
        similarities_dict[key1] = [value1, None]
        most_similar_key2 = max(dictionary2.keys(), key=lambda key2: SequenceMatcher(None, key1, key2).ratio())
        if SequenceMatcher(None, key1, most_similar_key2).ratio() >= key_similarity_limit:
            similarities_dict[key1] = [value1, dictionary2[most_similar_key2]]
    return similarities_dict


def is_float(str):
    """
    Test whether given string is a number
    @param str: string to be tested
    @return: true if the string is a number
    """
    try:
        float(str)
        return True
    except ValueError:
        return False


def compare_values_similarity(similarities_dict, number_similarity_deviation, string_similarity_deviation):
    """
    For each parameter name compare values from both specifications and return the ratio of same values
    @param similarities_dict: dictionary with parameter name and values from the first and second specification (if the match was found)
    @param number_similarity_deviation: percentage deviation, how much the values can differ in case of numerical values
    @param string_similarity_deviation: percentage deviation, how much the values can differ in case of textual values
    @return: number of common attributes with the same values for two specifications
    """
    matching_keys = 0
    matching_keys_and_values = 0
    for key, value in similarities_dict.items():
        if value[1] is not None:
            value[0] = re.sub(f' {UNIT_MARK}[\w]*', '', value[0])
            value[1] = re.sub(f' {UNIT_MARK}[\w]*', '', value[1])
            matching_keys += 1
            if is_float(value[0]) and is_float(value[1]):
                val1 = float(value[0])
                val2 = float(value[1])
                if val1 - number_similarity_deviation * val1 <= val2 and val2 <= val1 + number_similarity_deviation * val1:
                    matching_keys_and_values += 1
            else:
                if SequenceMatcher(None, value[0], value[1]).ratio() >= (1 - string_similarity_deviation):
                    matching_keys_and_values += 1
    return matching_keys, matching_keys_and_values
