import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

ID_MARK = '#id#'
BND_MARK = '#bnd#'
COL_MARK = '#col#'
MARKS = [ID_MARK, BND_MARK, COL_MARK]


def lower_case(data):
    """
    Lower case all names in dataset
    @param data: list of names
    @return: lowercased list of names
    """
    lowercased = []
    for d in data:
        lowercased.append(d.lower())
    return lowercased


def diff(list1, list2):
    """
    Find differences between two list of words of names
    @param list1: word list from the first name
    @param list2: word list from the second name
    @return: wordlist of different words
    """
    return list(set(list1) - set(list2)) + list(set(list2) - set(list1))


def remove_markers(data):
    """
    Remove all ids, colors, brands, etc. from names
    @param data: original product name
    @return: product name without markers
    """
    replaced = []
    for d in data:
        for m in MARKS:
            d = d.replace(m, '')
        replaced.append(d)
    return replaced


def remove_colors(data):
    """
    Remove colors from names
    @param data: original product name
    @return: product name without colors
    """
    replaced = []
    for d in data:
        words = []
        for w in d.split(' '):
            if COL_MARK not in w:
                words.append(w)
        replaced.append(' '.join(words))
    return replaced


def compute_tf_idf(data, print_stats=False):
    """
    Compute tf.idf score for each word in dataset
    @param print_stats: whether to print statistical values
    @param data: dataset with names
    @return: tf.idf for each word
    """
    data = remove_markers(data)
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', lowercase=False)
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    if print_stats:
        print('Tf.idf matrix score is:')
        print(df)
    return df


def compute_name_similarities(n1, n2, i, j, tf_idfs):
    """
    Compute similarity score of two products comparing images and names
    @param n1: first name
    @param n2: second name
    @param i: first name index
    @param j: second name index
    @param tf_idfs: tf ids of names
    @return: similarity score of names
    """
    name1 = n1.split(' ')
    name2 = n2.split(' ')
    match_ratios = {}

    # detect and compare ids
    id1 = [word for word in name1 if ID_MARK in word]
    id2 = [word for word in name2 if ID_MARK in word]
    if not id1 == []:
        match_ratios['id'] = len(set(id1) & set(id2)) / len(id1)

    # detect and compare brands
    bnd1 = [word for word in name1 if BND_MARK in word]
    bnd2 = [word for word in name2 if BND_MARK in word]
    if not bnd1 == [] and bnd1 == bnd2:
        match_ratios['brand'] = len(set(bnd1) & set(bnd2)) / len(bnd1)

    # ratio of the similar words
    name1 = remove_markers(name1)
    name2 = remove_markers(name2)
    list1 = set(name1)
    intersection = list1.intersection(name2)
    intersection_list = list(intersection)
    match_ratios['words'] = len(intersection_list) / len(name1)

    # cosine similarity of vectors from tf.idf
    match_ratios['cos'] = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]

    return match_ratios
