import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from scripts.score_computation.images_and_names.compute_total_similarity import plot_roc

ID_MARK = '#id#'
BND_MARK = '#bnd#'
COL_MARK = '#col#'
MARKS = [ID_MARK, BND_MARK, COL_MARK]
PRINT_STATS = False


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


def diff(li1, li2):
    """
    Find differeces between two list of words of names
    @param li1: word list from the first name
    @param li2: word list from the second name
    @return: word alist of different words
    """
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def load_file(name_file):
    """
    Load file with product names
    @param name_file: name of the input file
    @return: list of product names
    """
    names = []
    file = open(name_file, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        names.append(line)
    return names


def save_to_file(output_file, name1, name2, score, are_names_same):
    """
    Save names, similarities and indicator whether the names are the same to output file
    @param output_file: output file name
    @param name1: name of first product
    @param name2: name of second product
    @param score: similarity score
    @param are_names_same: indicator whether the names are the same
    @return:
    """

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f'{name1}, {name2}, {score}, {are_names_same}\n')


def remove_markers(data):
    """
    Remove all ids, colors, brands, etc. from names
    @param data: product name
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
    @param data: product name
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


def compute_tf_idf(data):
    """
    Compute tf.idf score for each word in dataset
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
    if PRINT_STATS:
        print('Tf.idf matrix score is:')
        print(df)
    return df


def compute_similarity_score(n1, i, n2, j, tf_idfs, weights):
    """
    Compute names similarity for 2 names and their indices in dataset
    @param n1: name of first product
    @param i: index of first product
    @param n2: name of second product
    @param j: index of second product
    @param tf_idfs: tf.idf scores for each word in dataset with names
    @param weights: dict of weights of id, brand, cos similarity and same words
    @return: similarity sore of two products
    """
    similarity_score = 0
    name1 = n1.split(' ')
    name2 = n2.split(' ')

    # detect and compare ids
    id1 = [word for word in name1 if ID_MARK in word]
    id2 = [word for word in name2 if ID_MARK in word]
    if not id1 == []:
        match_ratio = len(set(id1) & set(id2)) / len(id1)
        similarity_score += weights['id'] * match_ratio
        if PRINT_STATS:
            print(f'Matching ids: {id1}')
            print(f'Ratio of matching ids: {match_ratio}')

    # detect and compare brands
    bnd1 = [word for word in name1 if BND_MARK in word]
    bnd2 = [word for word in name2 if BND_MARK in word]
    if not bnd1 == [] and bnd1 == bnd2:
        match_ratio = len(set(bnd1) & set(bnd2)) / len(bnd1)
        similarity_score += weights['brand'] * match_ratio
        if PRINT_STATS:
            print(f'Matching brands: {bnd1}')
            print(f'Ratio of matching brands: {match_ratio}')

            # ratio of the similar words
    name1 = remove_markers(name1)
    name2 = remove_markers(name2)
    list1 = set(name1)
    intersection = list1.intersection(name2)
    intersection_list = list(intersection)
    match_ratio = len(intersection_list) / len(name1)
    similarity_score += match_ratio * weights['words']
    if PRINT_STATS:
        print(f'Ratio of common words in name1: {match_ratio}')
        print(f'Common words: {intersection_list}')
        print(f'Different words: {diff(name1, name2)}')

    # cosine similarity of vectors from tf.idf
    cos_sim = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]
    similarity_score += weights['cos'] * cos_sim

    if PRINT_STATS:
        print(f'Similarity score is: {similarity_score}')
    return similarity_score


def evaluate_dataset(scores, threshs):
    """
    Evaluate dataset - compute accuracy, confusion matrix and plot ROC
    @param scores: dataset with names similarities
    @param threshs: threshold to evaluate accuracy of similarities
    @return:
    """
    true_labels = [[row[2]] for row in scores]
    pred_labels_list = []
    precs = []
    recs = []
    for t in threshs:
        pred_labels = [[1 if row[3] > t else 0] for row in scores]
        pred_labels_list.append(pred_labels)
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)
        precs.append(prec)
        rec = recall_score(true_labels, pred_labels)
        recs.append(rec)
        if PRINT_STATS:
            print(f'For thresh {t}: \n Accuracy {acc} \n Precision {prec} \n Recall {rec}')
            print('Confusion matrix')
            print(conf_matrix)
            print('======')
    plot_roc(true_labels, pred_labels_list, threshs, print_stats=False)


def are_idxs_same(i, j):
    """
    Compare whether names of products on indices in dataset should be the same
    @param i: index i
    @param j: index j
    @return: True if the product on the given indices are the same, otherwise False
    """
    if i % 5 == 1:
        return 1 if (i == j or i + 1 == j or i + 2 == j or i + 3 == j) else 0
    if i % 5 == 2:
        return 1 if (i == j or i + 1 == j or i + 2 == j) else 0
    if i % 5 == 3:
        return 1 if (i == j or i + 1 == j) else 0
    if i % 5 == 4:
        return 1 if (i == j) else 0
    return 1 if (i == j or i + 1 == j or i + 2 == j or i + 3 == j or i + 4 == j) else 0


def remove_output_file_is_necessary(output_file):
    """
    Delete output file if it already exists
    @param output_file: output file to be removed
    @return:
    """
    if os.path.exists(output_file):
        os.remove(output_file)
