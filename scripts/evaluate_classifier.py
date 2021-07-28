from sklearn.metrics.pairwise import cosine_similarity

from score_computation.names.compute_names_similarity import ID_MARK, BND_MARK
from score_computation.names.compute_names_similarity import remove_markers


def compute_name_similarities(n1, n2, i, j, tf_idfs):
    """
    Compute similarity score of two products comparing images and names
    @param n1: first name
    @param n2: second name
    @param i: first name index
    @param j: second name index
    @param tf_idfs: tf ids of names
    @return: similarity score of names and images
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
