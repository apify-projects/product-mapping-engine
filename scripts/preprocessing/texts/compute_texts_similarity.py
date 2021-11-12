from math import floor, ceil

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ID_MARK = '#id#'
BND_MARK = '#bnd#'
COL_MARK = '#col#'
UNIT_MARK = '#unit#'
MARKS = [ID_MARK, BND_MARK, COL_MARK, UNIT_MARK]


def compute_similarity_of_datasets(dataset1, dataset2):
    """
    Compute similarity score of two datasets
    @param dataset1: first dataset
    @param dataset2: second dataset
    @return:
    """
    dataset1_nomarkers = remove_markers(dataset1)
    dataset2_nomarkers = remove_markers(dataset2)
    tf_idfs = create_tf_idf(dataset1_nomarkers, dataset2_nomarkers)
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
            bnd1 = [word for word in text1 if BND_MARK in word]
            bnd2 = [word for word in text2 if BND_MARK in word]
            match_ratios['brand'] = 0
            if not bnd1 == [] and bnd1 == bnd2:
                match_ratios['brand'] = len(set(bnd1) & set(bnd2)) / len(bnd1)

            # ratio of the similar words
            list1 = set(dataset1_nomarkers[i])
            intersection = list1.intersection(dataset2_nomarkers[j])
            intersection_list = list(intersection)
            match_ratios['words'] = len(intersection_list) / len(text1)

            # cosine similarity of vectors from tf.idf
            match_ratios['cos'] = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]
            match_ratios_list.append(match_ratios)

            #TODO: compare units

            #TODO: descriptive words

    return match_ratios_list


def create_tf_idf(dataset1, dataset2):
    """
    Creates tf.idfs for datasets
    @param dataset1: first dataset
    @param dataset2: second dataset
    @return: tf.idfs
    """
    dataset1 = [' '.join(d) for d in dataset1]
    dataset2 = [' '.join(d) for d in dataset2]
    data = dataset1 + dataset2
    tf_idfs = compute_tf_idf(data)
    return tf_idfs


def compute_tf_idf(data, print_stats=False):
    """
    Compute tf.idf score for each word in dataset
    @param print_stats: whether to print statistical values
    @param data: dataset with names
    @return: tf.idf for each word
    """
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b[\w.,-]+\\b',
                                 lowercase=False)  # unicode and then matchin \b empty line before and after word and \wmatching word
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    tf_idfs = pd.DataFrame(denselist, columns=feature_names)
    if print_stats:
        print('Tf.idf matrix score is:')
        print(tf_idfs)
    return tf_idfs


def remove_markers(data):
    """
    Remove all ids, colors, brands, etc. from dataset containing list of list of words of texts
    @param data: original texts words
    @return: text words without markers
    """
    for text in data:
        for j, word in enumerate(text):
            for m in MARKS:
                text[j] = word.replace(m, '')
    return data


################# NEZPRACOVANO

def cosine_similarity_of_datasets(tf_idfs):
    """
    Compute cosine similarity of datasets
    @param tf_idfs: tf.idfs of data
    @return: cosine similarity of datasets
    """
    length = len(tf_idfs)
    cos_similarities = []
    for i in range(0, floor(length / 2)):
        for j in range(ceil(length / 2), length):
            res = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]
            cos_similarities.append(res)
    return cos_similarities


def find_descriptive_words(tf_idfs, filter_limit, top_words):
    """
    Find the mmost important words in datasets
    @param tf_idfs: tf.idf of data
    @param filter_limit: the max limit of occurences of word among documents
    @param top_words: how many the most important words are to be selected
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
        descriptive_words.append(vector_ordered.head(top_words))
    descriptive_words = pd.DataFrame(descriptive_words).fillna(0)
    return descriptive_words


def compute_descriptive_words_similarity(vector1, vector2):
    """
    Compute similarity of descriptive words between two descriptions
    @param vector1: vector of descriptive words of first description
    @param vector2: vector of descriptive words of second description
    @return: number of words that occur in both vectors
    """
    counter = 0
    for i, j in zip(vector1, vector2):
        counter += 1 if i != 0 and j != 0 else 0
    return counter


def compare_descriptive_words(tf_idfs, filter_limit, top_words):
    """
    Find and compare descriptive words
    @param tf_idfs: tf.idfs of data
    @param filter_limit: the max limit of occurences of word among documents
    @param top_words: how many the most important words are to be selected
    @return: similarity of descriptive words
    """
    descriptives = find_descriptive_words(tf_idfs, filter_limit, top_words)
    descriptives_similarity = []
    length = len(descriptives)
    for i in range(0, floor(length / 2)):
        for j in range(ceil(length / 2), length):
            # res = cosine_similarity([representatives.iloc[i].values, representatives.iloc[j].values])[0][1]
            res = compute_descriptive_words_similarity(descriptives.iloc[i].values, descriptives.iloc[j].values)
            descriptives_similarity.append(res / top_words)
    return descriptives_similarity


def compare_units_in_descriptions(dataset1, dataset2, devation=0.05):
    """
    Compare detected units from the texts
    @param dataset1: List of products each containing list of units from the first dataset
    @param dataset2: List of products each containing list of units from the second dataset
    @param devation: percent of toleration of deviations of two compared numbers
    @return: Ratio of the same units between two products
    """
    similarity_scores = []
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
