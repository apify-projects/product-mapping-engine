import copy
from math import floor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocessing.texts.keywords_detection import ID_MARK, BRAND_MARK, COLOR_MARK, UNIT_MARK

MARKS = [ID_MARK, BRAND_MARK, COLOR_MARK, UNIT_MARK]
TOP_WORDS = 10
FILTER_LIMIT = 0.5


def compute_similarity_of_texts(dataset1, dataset2):
    """
    Compute similarity score of two text datasets
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @return:
    """
    dataset1_nomarkers = remove_markers(copy.deepcopy(dataset1))
    dataset2_nomarkers = remove_markers(copy.deepcopy(dataset2))
    tf_idfs = create_tf_idf(dataset1_nomarkers, dataset2_nomarkers)
    descriptive_words = find_descriptive_words(
        tf_idfs, filter_limit=FILTER_LIMIT, number_of_top_words=TOP_WORDS
    )
    half_length = floor(len(descriptive_words) / 2)
    match_ratios_list = []

    for i, text1 in enumerate(dataset1):
        for j, text2 in enumerate(dataset2):
            match_ratios = {}
            # detect and compare ids
            id1 = [word for word in text1 if ID_MARK in word]
            id2 = [word for word in text2 if ID_MARK in word]
            match_ratios['id'] = 0
            if not id1 == []:
                match_ratios['id'] = len(set(id1) & set(id2)) / len(id1)

            # detect and compare brands
            bnd1 = [word for word in text1 if BRAND_MARK in word]
            bnd2 = [word for word in text2 if BRAND_MARK in word]
            match_ratios['brand'] = 0
            if not bnd1 == [] and bnd1 == bnd2:
                match_ratios['brand'] = len(set(bnd1) & set(bnd2)) / len(bnd1)

            # ratio of the similar words
            list1 = set(dataset1_nomarkers[i])
            intersection = list1.intersection(dataset2_nomarkers[j])
            intersection_list = list(intersection)
            match_ratios['words'] = len(intersection_list) / len(text1)

            # cosine similarity of vectors from tf-idf
            match_ratios['cos'] = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]
            match_ratios_list.append(match_ratios)

            # compare ratio of corresponding units and values in both texts
            match_ratios['units'] = compare_units_and_values(text1, text2)

            # commpute number of similar words in both texts
            match_ratios['descriptives'] = compute_descriptive_words_similarity(
                descriptive_words.iloc[i].values,
                descriptive_words.iloc[half_length + j].values
            ) / TOP_WORDS
    return match_ratios_list


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
    for i, j in zip(incidence_vector1, incidence_vector2):
        counter += 1 if i != 0 and j != 0 else 0
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
