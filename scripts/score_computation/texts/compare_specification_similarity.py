import re
from difflib import SequenceMatcher

from scripts.preprocessing.texts.keywords_detection import UNIT_MARK


def compute_similarity_of_specifications(dataset1, dataset2):
    """
    Compare two specifications and find matching keys and keys and values
    @param dataset1: first dictionary of specification parameter names and values
    @param dataset2: second dictionary of specification parameter names and values
    @return: ratio of matching keys and matching keys and values
    """
    similarity_scores = []
    for product1 in dataset1:
        for product2 in dataset2:
            similarities_dict = find_closest_keys(product1, product2, key_similarity_limit=1)
            matching_keys, matching_keys_and_values = compare_values_similarity(similarities_dict,
                                                                                number_similarity_deviation=0.1,
                                                                                string_similarity_limit=0.9)
            similarity_scores.append([matching_keys / len(product1), matching_keys_and_values / len(product1)])
    return similarity_scores


def find_closest_keys(dictionary1, dictionary2, key_similarity_limit):
    """
    Find corresponding parameter pairs from both specifications according to their name correspondence
    @param dictionary1: first dictionary of specification parameter names and values
    @param dictionary2: second dictionary of specification parameter names and values
    @param key_similarity_limit: percentage limit how much the keys must be similar
    @return: dictionary with parameter names and values from first dictionary supplemented by corresponding values for the same parameter names from the second dictionary
    """
    similarities_dict = {}
    for k1, v1 in dictionary1.items():
        similarities_dict[k1] = [v1, None]
        most_similar_k2 = max(dictionary2.keys(), key=lambda k2: SequenceMatcher(None, k1, k2).ratio())
        if SequenceMatcher(None, k1, most_similar_k2).ratio() >= key_similarity_limit:
            similarities_dict[k1] = [v1, dictionary2[most_similar_k2]]
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


def compare_values_similarity(similarities_dict, number_similarity_deviation, string_similarity_limit):
    """
    For each parameter name compare values from both specifications and return the ratio of same values
    @param similarities_dict: dictionary with parameter name and values from the first and second specification (if the match was found)
    @param number_similarity_deviation: percentage deviation, how much the values can differ in case of numerical values
    @param string_similarity_limit: percentage deviation, how much the values can differ in case of textual values
    @return: number of matching keys and number of matching values for two specifications
    """
    matching_keys = 0
    matching_keys_and_values = 0
    for k, v in similarities_dict.items():
        if v[1] is not None:
            v[0] = re.sub(f' {UNIT_MARK}[\w]*', '', v[0])
            v[1] = re.sub(f' {UNIT_MARK}[\w]*', '', v[1])
            matching_keys += 1
            c = v[0].isnumeric()
            if is_float(v[0]) and is_float(v[1]):
                val1 = float(v[0])
                val2 = float(v[1])
                if val1 - number_similarity_deviation * val1 <= val2 and val2 <= val1 + number_similarity_deviation * val1:
                    matching_keys_and_values += 1
            else:
                if SequenceMatcher(None, v[0], v[1]).ratio() >= string_similarity_limit:
                    matching_keys_and_values += 1
    return matching_keys, matching_keys_and_values
