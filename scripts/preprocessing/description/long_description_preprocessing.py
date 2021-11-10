import re
from math import floor, ceil

import majka
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocessing.names.names_preprocessing import split_units_and_values, split_words
from scripts.score_computation.names.compute_names_similarity import compute_tf_idf, lower_case


def replace_commas_for_dot_in_numbers(data):
    """
    Replace commas for dots in floats
    @param data: List of texts
    @return: replaced list of texts
    """
    new_data = []
    for d in data:
        d = re.sub(r'(?<=\d),(?=\d)', '', d)
        new_data.append(d)
    return new_data


def preprocess_descriptions_and_create_tf_idf(dataset1, dataset2, lemmatizer):
    """
    Preprocess data and create tf.idf of both datasets
    @param dataset1: First dataset to preprocess
    @param dataset2: Second dataset to preprocess
    @param lemmatizer: lemmatizer to be used to lemmatize texts
    @return: Preprocessed datasets and computed tf.idfs
    """
    dataset1 = preprocess_description(dataset1, lemmatizer)
    dataset1 = [' '.join(d) for d in dataset1]
    dataset2 = preprocess_description(dataset2, lemmatizer)
    dataset2 = [' '.join(d) for d in dataset2]
    data = dataset1 + dataset2
    tf_idfs = compute_tf_idf(data, do_remove_markers=False)
    return dataset1, dataset2, tf_idfs


def preprocess_description(data, lemmatizer):
    """
    Lowercase and split units and values in dataset
    @param data: data to preprocess
    @param lemmatizer: lemmatizer to be used to lemmatize texts
    @return: preprocessed data
    """
    new_data = []
    for d in data:
        d = split_words(d)
        d = split_units_and_values(d)
        d = lower_case(d)
        d = lemmatize_czech_text(d, lemmatizer)
        new_data.append(d)
    return new_data


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


def set_czech_lemmatizer():
    morph = majka.Majka('data/vocabularies/majka.w-lt')
    morph.flags |= majka.ADD_DIACRITICS  # find word forms with diacritics
    morph.flags |= majka.DISALLOW_LOWERCASE  # do not enable to find lowercase variants
    morph.flags |= majka.IGNORE_CASE  # ignore the word case whatsoever
    morph.flags = 0  # unset all flags
    morph.tags = True  # return just the lemma, do not process the tags
    morph.compact_tag = False  # not return tag in compact form (as returned by Majka)
    morph.first_only = True  # return only the first entry (not all entries)
    return morph


def lemmatize_czech_text(text, morph):
    lemmatized_text = []
    for word in text:
        x = morph.find(word)
        if x == []:
            lemma = word
        else:
            lemma = x[0]['lemma']
            if 'negation' in x[0]['tags'] and x[0]['tags']['negation']:
                lemma = 'ne' + lemma
        lemmatized_text.append(lemma)
    return lemmatized_text
