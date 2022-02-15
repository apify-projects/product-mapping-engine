import copy
from math import floor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ...configuration import NUMBER_OF_TOP_DESCRIPTIVE_WORDS, \
    MAX_DESCRIPTIVE_WORD_OCCURRENCES_IN_TEXTS, UNITS_AND_VALUES_DEVIATION, SIMILARITIES_TO_BE_COMPUTED
from ...preprocessing.texts.keywords_detection import ID_MARK, BRAND_MARK, UNIT_MARK, MARKS, NUMBER_MARK, is_number


def compute_similarity_of_texts(dataset1, dataset2, product_pairs_idx, tf_idfs, descriptive_words):
    """
    Compute similarity score of each pair in both datasets
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @param product_pairs_idx: dict with indices of candidate matching pairs
    @param tf_idfs: tf.idfs of all words from both datasets
    @param descriptive_words: descriptive words from both datasets
    @return: dataset of pair similarity scores
    """

    half_length = floor(len(descriptive_words) / 2)
    match_ratios_list = []

    for product_idx, corresponding_indices in product_pairs_idx.items():
        product1 = dataset1.loc[[product_idx]].values[0]
        product1_no_markers = remove_markers(copy.deepcopy(product1))
        bnd1 = [word for word in product1 if BRAND_MARK in word]
        id1 = [word for word in product1 if ID_MARK in word]

        for product2_idx in corresponding_indices:
            product2 = dataset2.loc[[product2_idx]].values[0]
            match_ratios = {}

            # detect and compare ids
            if 'id' in SIMILARITIES_TO_BE_COMPUTED:
                id2 = [word for word in product2 if ID_MARK in word]
                match_ratios['id'] = compute_matching_pairs(id1, id2, True)

            # detect and compare brands
            if 'brand' in SIMILARITIES_TO_BE_COMPUTED:
                bnd2 = [word for word in product2 if BRAND_MARK in word]
                match_ratios['brand'] = compute_matching_pairs(bnd1, bnd2)

            # ratio of the similar words
            if 'words' in SIMILARITIES_TO_BE_COMPUTED:
                product2_no_markers = remove_markers(copy.deepcopy(product2))
                match_ratios['words'] = compute_matching_pairs(product1_no_markers, product2_no_markers)

            if 'cos' in SIMILARITIES_TO_BE_COMPUTED:
                # cosine similarity of vectors from tf-idf
                cos_similarity = cosine_similarity(
                    [tf_idfs.iloc[product_idx].values, tf_idfs.iloc[product2_idx + len(dataset1)].values]
                )[0][1]
                if product1 == "" or product2 == "":
                    match_ratios['cos'] = 0
                else:
                    match_ratios['cos'] = 2 * cos_similarity - 1

            if 'descriptives' in SIMILARITIES_TO_BE_COMPUTED:
                # compute number of similar words in both texts
                descriptive_words_sim = compute_descriptive_words_similarity(
                    descriptive_words.iloc[product_idx].values,
                    descriptive_words.iloc[half_length + product_idx].values
                ) / NUMBER_OF_TOP_DESCRIPTIVE_WORDS
                if product1 == "" or product2 == "":
                    match_ratios['descriptives'] = 0
                else:
                    match_ratios['descriptives'] = 2 * descriptive_words_sim - 1

            if 'units' in SIMILARITIES_TO_BE_COMPUTED:
                # compare ratio of corresponding units and values in both texts
                match_ratios['units'] = compare_units_and_values(product1, product2)

            if 'numbers' in SIMILARITIES_TO_BE_COMPUTED:
                # compare unspecified numbers in both texts
                match_ratios['numbers'] = compare_numerical_values(product1, product2)

            match_ratios_list.append(match_ratios)
    return match_ratios_list


def compute_matching_pairs(list1, list2, allow_substrings=False):
    """
    Compute matching items in two lists
    @param allow_substrings: allow substring in finding matching pairs
    @param list1: first list of items
    @param list2: second list of items
    @return: ratio of matching items
    """
    matches = 0
    list1 = set(list1)
    list2 = set(list2)
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 or (allow_substrings and (item1 in item2 or item2 in item1)):
                matches += 1
    if (len(list1) + len(list2)) == 0:
        return 0
    return 4 * matches / (len(list1) + len(list2)) - 1


def find_descriptive_words(tf_idfs, filter_limit, number_of_top_words):
    """
    Find *number_of_top_words* top words that occurs in at maximum of *filter_limit* percent of texts
    @param tf_idfs: tf-idf of data
    @param filter_limit: the max limit of occurrences of word among documents
    @param number_of_top_words: how many most important words are to be selected
    @return: found descriptive words
    """
    filter_limit = len(tf_idfs) * filter_limit
    tf_idf_filtered = []
    descriptive_words = []
    for col in tf_idfs:
        word = tf_idfs[col]
        non_zeros = np.count_nonzero(word.values)
        if non_zeros < filter_limit:
            tf_idf_filtered.append(word)
    tf_idf_filtered = pd.DataFrame(tf_idf_filtered)
    for column in tf_idf_filtered:
        vector_ordered = tf_idfs.T[column].sort_values(ascending=False)
        descriptive_words.append(vector_ordered.head(number_of_top_words))
    descriptive_words = pd.DataFrame(descriptive_words).fillna(0)
    return descriptive_words


def compute_descriptive_words_similarity(incidence_vector1, incidence_vector2):
    """
    Compute similarity of descriptive words between two descriptions
    @param incidence_vector1: vector of descriptive words of first description
    @param incidence_vector2: vector of descriptive words of second description
    @return: number of words that occur in both vectors
    """
    counter = 0
    for item1, item2 in zip(incidence_vector1, incidence_vector2):
        counter += 1 if item1 != 0 and item2 != 0 else 0
    return counter


def create_tf_idf(dataset1, dataset2):
    """
    Concatenate both datasets of texts into one and creates tf-idfs vectors for all words
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @return: tf-idfs
    """
    dataset1 = [' '.join(d) for d in dataset1]
    dataset2 = [' '.join(d) for d in dataset2]
    data = dataset1 + dataset2
    tf_idfs = compute_tf_idf(data)
    return tf_idfs


def compute_tf_idf(dataset, print_stats=False):
    """
    Compute tf-idf score for each word in dataset
    @param print_stats: whether to print tf-idf matrix score
    @param dataset: dataset with texts
    @return: tf-idf for each word
    """
    vectorizer = TfidfVectorizer(
        token_pattern='(?u)\\b[\w.,-]+\\b',
        lowercase=False
    )  # unicode and then matching \b empty line before and after word and \w matching word
    tfidf_vectors = vectorizer.fit_transform(dataset)
    feature_names = vectorizer.get_feature_names()
    dense = tfidf_vectors.todense()
    dense_list = dense.tolist()
    tf_idfs = pd.DataFrame(dense_list, columns=feature_names)
    if print_stats:
        print('Tf-idf matrix score is:')
        print(tf_idfs)
    return tf_idfs


def create_tf_idfs_and_descriptive_words(dataset1, dataset2, columns):
    """
    Create tf.idfs and descriptive words for each column in the dataset
    @param dataset1: first dataframe in which to create tf.idfs and descriptive words
    @param dataset2: second dataframe in which to create tf.idfs and descriptive words
    @param columns: list of columns to create tf.idfs and descriptive words in
    @return: dict with tf.idfs and descriptive words for each column
    """
    tf_idfs = {}
    descriptive_words = {}
    for column in columns:
        tf_idfs_col = create_tf_idf(dataset1[column], dataset2[column])
        descriptive_words_col = find_descriptive_words(
            tf_idfs_col, filter_limit=MAX_DESCRIPTIVE_WORD_OCCURRENCES_IN_TEXTS,
            number_of_top_words=NUMBER_OF_TOP_DESCRIPTIVE_WORDS
        )
        tf_idfs[column] = tf_idfs_col
        descriptive_words[column] = descriptive_words_col
    return tf_idfs, descriptive_words


def remove_markers(dataset):
    """
    Remove all ids, colors, brand and units markers from dataset
    @param dataset: original dataset words containing list of lists of words of texts
    @return: dataset without markers
    """
    for text in dataset:
        for j, word in enumerate(text):
            for mark_token in MARKS:
                if mark_token in word:
                    text[j] = word.replace(mark_token, '')
    return dataset


def compare_numerical_values(text1, text2):
    """
    Compare detected numbers from the texts
    @param text1: first list of words for number comparison
    @param text2:  second list of words for number comparison
    @return: Ratio of the same numbers between two texts
    """
    numbers1 = [word.replace(NUMBER_MARK, '') for word in text1 if NUMBER_MARK in word]
    numbers2 = [word.replace(NUMBER_MARK, '') for word in text2 if NUMBER_MARK in word]
    matches = len(set(numbers1) & set(numbers2))
    total_len = len(numbers1) + len(numbers2)
    if matches == 0:
        return 0
    if not total_len == 0:
        return (total_len - 2 * matches) / total_len
    return 0


def compare_units_and_values(text1, text2, deviation=UNITS_AND_VALUES_DEVIATION):
    """
    Compare detected units from the texts
    @param text1: List of words for unit detection and comparison
    @param text2: List of words for unit detection and comparison
    @param deviation: percent of toleration of deviations of two compared numbers
    @return: Ratio of the same units between two texts
    """
    units_list1 = extract_units_and_values(text1)
    units_list2 = extract_units_and_values(text2)
    matches = 0
    total_len = len(units_list1) + len(units_list2)
    for u1 in units_list1:
        for u2 in units_list2:
            if u1[0] == u2[0]:
                if u1[0] == f'{UNIT_MARK}size' and u1[1] == u2[1]:
                    matches += 1
                elif type(u1[1]) is str:
                    if '×' in u1[1] and sorted(u1[1].split('×')) == sorted(u2[1].split('×')):
                        matches += 1
                elif type(u2[1]) is not str:
                    if (1 - deviation) * u2[1] < u1[1] < (1 + deviation) * u2[1]:
                        matches += 1
    if matches == 0:
        return 0
    if not total_len == 0:
        return (total_len - 2 * matches) / total_len
    return 0


def extract_units_and_values(text):
    """
    Extract units and values from the list of words
    @param text: list of words to extract units and values from
    @return: list of extracted pairs unit-value
    """
    unit_list = []
    for i, word in enumerate(text):
        if UNIT_MARK in word:
            if is_number(text[i - 1]):
                unit_list.append([word, float(text[i - 1])])
            else:
                unit_list.append([word, text[i - 1]])
    return unit_list
