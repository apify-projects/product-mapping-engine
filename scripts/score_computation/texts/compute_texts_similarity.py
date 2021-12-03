import copy
from math import floor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocessing.texts.keywords_detection import ID_MARK, BRAND_MARK, COLOR_MARK, UNIT_MARK, \
    detect_ids_brands_colors_and_units

MARKS = [ID_MARK, BRAND_MARK, COLOR_MARK, UNIT_MARK]
TOP_WORDS = 10
FILTER_LIMIT = 0.5


def compute_similarity_of_texts(dataset1, dataset2, id_detection, color_detection, brand_detection, units_detection):
    """
    Compute similarity score of each pair in both datasets
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @return: dataset of pair similarity scores
    """

    dataset1 = detect_ids_brands_colors_and_units(
        dataset1,
        id_detection,
        color_detection,
        brand_detection,
        units_detection
    )
    dataset2 = detect_ids_brands_colors_and_units(
        dataset2,
        id_detection,
        color_detection,
        brand_detection,
        units_detection
    )

    dataset1_nomarkers = remove_markers(copy.deepcopy(dataset1))
    dataset2_nomarkers = remove_markers(copy.deepcopy(dataset2))
    tf_idfs = create_tf_idf(dataset1_nomarkers, dataset2_nomarkers)
    descriptive_words = find_descriptive_words(
        tf_idfs, filter_limit=FILTER_LIMIT, number_of_top_words=TOP_WORDS
    )
    half_length = floor(len(descriptive_words) / 2)
    match_ratios_list = []

    for x in range(len(dataset1)):
        match_ratios = {}
        # detect and compare ids
        id1 = [word for word in dataset1[x] if ID_MARK in word]
        id2 = [word for word in dataset2[x] if ID_MARK in word]
        match_ratios['id'] = compute_matching_pairs(id1, id2)

        # detect and compare brands
        bnd1 = [word for word in dataset1[x] if BRAND_MARK in word]
        bnd2 = [word for word in dataset2[x] if BRAND_MARK in word]
        match_ratios['brand'] = compute_matching_pairs(bnd1, bnd2)

        # ratio of the similar words
        list1 = set(dataset1_nomarkers[x])
        intersection = list1.intersection(dataset2_nomarkers[x])
        intersection_list = list(intersection)
        match_ratios['words'] = compute_matching_pairs(dataset1_nomarkers[x], dataset2_nomarkers[x])

        # cosine similarity of vectors from tf-idf
        cos_similarity = cosine_similarity([tf_idfs.iloc[x].values, tf_idfs.iloc[x + len(dataset1)].values])[0][1]

        # commpute number of similar words in both texts
        descriptive_words_sim = compute_descriptive_words_similarity(
            descriptive_words.iloc[x].values,
            descriptive_words.iloc[half_length + x].values
        ) / TOP_WORDS

        if dataset1[x] == "" or dataset2[x] == "":
            match_ratios['cos'] = 0
            match_ratios['descriptives'] = 0
        else:
            match_ratios['cos'] = 2 * cos_similarity - 1
            match_ratios['descriptives'] = 2 * descriptive_words_sim - 1

        # compare ratio of corresponding units and values in both texts
        match_ratios['units'] = compare_units_and_values(dataset1[x], dataset2[x])

        match_ratios_list.append(match_ratios)
    return match_ratios_list


def compute_matching_pairs(list1, list2):
    """
    Compute matching items in two lists
    @param list1: first list of items
    @param list2: second list of items
    @return: ratio of matching items
    """
    matches = 0
    list1 = set(list1)
    list2 = set(list2)
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
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


def remove_markers(dataset):
    """
    Remove all ids, colors, brand and units markers from dataset
    @param dataset: original dataset words containing list of lists of words of texts
    @return: dataset without markers
    """
    for text in dataset:
        for j, word in enumerate(text):
            for m in MARKS:
                text[j] = word.replace(m, '')
    return dataset


def compare_units_and_values(text1, text2, devation=0.05):
    """
    Compare detected units from the texts
    @param text1: List of words for unit detection and comparison
    @param text2: List of words for unit detection and comparison
    @param devation: percent of toleration of deviations of two compared numbers
    @return: Ratio of the same units between two products
    """
    units_list1 = extract_units_and_values(text1)
    units_list2 = extract_units_and_values(text2)
    matches = 0
    total_len = len(units_list1) + len(units_list2)
    for u1 in units_list1:
        for u2 in units_list2:
            if u1[0] == u2[0] and u1[1] > (1 - devation) * u2[1] and u1[1] < (1 + devation) * u2[1]:
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
    @return: extracted pairs unit-value
    """
    unit_list = []
    for i, word in enumerate(text):
        if UNIT_MARK in word:
            unit_list.append([word, float(text[i - 1])])
    return unit_list
