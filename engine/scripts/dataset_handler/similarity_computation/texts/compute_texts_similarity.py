import copy
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .compute_specifications_similarity import compute_similarity_of_specifications
from ...preprocessing.texts.keywords_detection import ID_MARK, BRAND_MARK, UNIT_MARK, MARKS, NUMBER_MARK, is_number
from ....configuration import NUMBER_OF_TOP_DESCRIPTIVE_WORDS, \
    MAX_DESCRIPTIVE_WORD_OCCURRENCES_IN_TEXTS, UNITS_AND_VALUES_DEVIATION, SIMILARITIES_TO_BE_COMPUTED, \
    COLUMNS_TO_BE_PREPROCESSED, KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED, \
    ALL_KEYWORDS_SIMILARITIES


def compute_similarity_of_texts(dataset1, dataset2, tf_idfs, descriptive_words, similarities_to_compute,
                                dataset2_starting_index, are_there_markers=True):
    """
    Compute similarity score of each pair in both datasets
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @param tf_idfs: tf.idfs of all words from both datasets
    @param descriptive_words: descriptive words from both datasets
    @param similarities_to_compute: similarities that should be computed
    @param dataset2_starting_index: starting index of the data from second dataset in tf_idfs and descriptive_words
    @param are_there_markers: whether there are markers detecting units, ids, brands etc in the datasets
    @return: dataset of pair similarity scores
    """
    match_ratios_list = []

    for (product1_idx, product1), products2_list in zip(dataset1.items(), dataset2):
        if are_there_markers:
            product1_no_markers = remove_markers(copy.deepcopy(product1))
        else:
            product1_no_markers = copy.deepcopy(product1)
        bnd1 = [word for word in product1 if BRAND_MARK in word]
        id1 = [word for word in product1 if ID_MARK in word]

        for product2_idx, product2 in products2_list.items():
            match_ratios = {}

            # detect and compare ids
            if 'id' in similarities_to_compute:
                id2 = [word for word in product2 if ID_MARK in word]
                match_ratios['id'] = compute_match_ratio(id1, id2, True)

            # detect and compare brands
            if 'brand' in similarities_to_compute:
                bnd2 = [word for word in product2 if BRAND_MARK in word]
                match_ratios['brand'] = compute_match_ratio(bnd1, bnd2)

            # ratio of the similar words
            if 'words' in similarities_to_compute:
                if are_there_markers:
                    product2_no_markers = remove_markers(copy.deepcopy(product2))
                else:
                    product2_no_markers = copy.deepcopy(product2)
                match_ratios['words'] = compute_match_ratio(product1_no_markers, product2_no_markers)

            if 'cos' in similarities_to_compute and tf_idfs is not None:
                # cosine similarity of vectors from tf-idf
                cos_similarity = cosine_similarity(
                    [tf_idfs.iloc[product1_idx].values, tf_idfs.iloc[product2_idx + dataset2_starting_index].values]
                )[0][1]
                if product1 == [""] or product2 == [""]:
                    match_ratios['cos'] = 0
                else:
                    match_ratios['cos'] = 2 * cos_similarity - 1

            if 'descriptives' in similarities_to_compute and descriptive_words is not None:
                # compute number of similar words in both texts
                if product1 == [""] or product2 == [""] or product1 == ["column_did_not_exist_in_scraper"] or product2 == ["column_did_not_exist_in_scraper"]:
                    match_ratios['descriptives'] = 0
                else:
                    descriptive_words_sim = compute_descriptive_words_similarity(
                        descriptive_words.iloc[product1_idx].values,
                        descriptive_words.iloc[product2_idx + dataset2_starting_index].values
                    ) / NUMBER_OF_TOP_DESCRIPTIVE_WORDS

                    match_ratios['descriptives'] = 2 * descriptive_words_sim - 1

            if 'units' in similarities_to_compute:
                # compare ratio of corresponding units and values in both texts
                match_ratios['units'] = compare_units_and_values(product1, product2)

            if 'numbers' in similarities_to_compute:
                # compare unspecified numbers in both texts
                match_ratios['numbers'] = compare_numerical_values(product1, product2)

            match_ratios_list.append(match_ratios)
    return match_ratios_list


def compute_similarity_of_keywords(keywords1, keywords2_df):
    """
    Computes similarity of keywords
    @param keywords1: first dataframe with keywords
    @param keywords2_df: list of dataframes with second keywords
    @return: list of keywords similarities
    """
    total_similarities = []
    for keywords1, keywords2_list in zip(keywords1.iterrows(), keywords2_df):
        similarities_dict = {}
        keywords1 = keywords1[1].to_dict()
        for keywords2 in keywords2_list.iterrows():
            keywords2 = keywords2[1].to_dict()
            for (key1, value1), (key2, value2) in zip(keywords1.items(), keywords2.items()):
                if key1 in ALL_KEYWORDS_SIMILARITIES:
                    value1_tuples = [tuple(v) for v in value1]
                    value2_tuples = [tuple(v) for v in value2]
                    similarity = len(set(value1_tuples) & set(value2_tuples))
                    total_len = len(value1) + len(value2)
                    if similarity != 0 and not total_len == 0:
                        similarity = (total_len - 2 * similarity) / total_len
                    similarities_dict[key1] = similarity
            total_similarities.append(similarities_dict)
    return total_similarities


def compute_match_ratio(list1, list2, allow_substrings=False):
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
        if non_zeros <= filter_limit:
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
        token_pattern='(?u)\\b[\\w.,-]+\\b',
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


def create_tf_idfs_and_descriptive_words(dataset1, dataset2):
    """
    Create tf.idfs and descriptive words for each column in the dataset
    @param dataset1: first dataframe in which to create tf.idfs and descriptive words
    @param dataset2: second dataframe in which to create tf.idfs and descriptive words
    @return: dict with tf.idfs and descriptive words for each column
    """
    tf_idfs = {}
    descriptive_words = {}
    for column in COLUMNS_TO_BE_PREPROCESSED:
        if not (column in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED.keys() and 'cos' in
                KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]):
            tf_idfs_col = create_tf_idf(dataset1[column], dataset2[column])
            tf_idfs[column] = tf_idfs_col

            if not (column in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED.keys() and
                    'descriptives' in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[column]):
                descriptive_words_col = find_descriptive_words(
                    tf_idfs_col, filter_limit=MAX_DESCRIPTIVE_WORD_OCCURRENCES_IN_TEXTS,
                    number_of_top_words=NUMBER_OF_TOP_DESCRIPTIVE_WORDS
                )
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


def compare_units_and_values(text1, text2):
    """
    Compare detected units from the texts
    @param text1: List of words for unit detection and comparison
    @param text2: List of words for unit detection and comparison
    @return: Ratio of the same units between two texts
    """
    units_list1 = extract_units_and_values(text1)
    units_list2 = extract_units_and_values(text2)
    matches = 0
    total_len = len(units_list1) + len(units_list2)
    for u1 in units_list1:
        for u2 in units_list2:
            if u1[0] == u2[0]:
                if u1[0] == f'{UNIT_MARK}size' and u2[0] == f'{UNIT_MARK}size':
                    matches += 1 if u1[1] == u2[1] else 0
                elif '×' in str(u1[1]) and '×' in str(u2[1]):
                    list1 = [float(x) for x in u1[1].split('×')]
                    list2 = [float(x) for x in u2[1].split('×')]
                    matching_values = 0
                    for val1 in list1:
                        for val2 in list2:
                            matching_values += compute_matching_units(val1, val2)
                    matches += matching_values
                    if len(list1) + len(list2) - 2 < 0:
                        print('This should not happen! Unpredictable bug!')
                    total_len += len(list1) + len(list2) - 2
                elif is_number(u1[1]) and is_number(u2[1]):
                    matches += compute_matching_units(u1[1], u2[1])
    if matches == 0:
        return 0
    if not total_len == 0:
        return (total_len - 2 * matches) / total_len
    return 0


def compute_matching_units(value1, value2):
    """
    Compute whether two values are equal with respect to some deviation
    @param value1: first value to be compared
    @param value2: second value to be compared
    @return: 1 if values are the same else 0
    """
    return 1 if (1 - UNITS_AND_VALUES_DEVIATION) * value2 < value1 < (1 + UNITS_AND_VALUES_DEVIATION) * value2 else 0


def chunks(dictionary, dict_num):
    """
    Split dictionary into several same parts
    @param dictionary: dictionary to be split
    @param dict_num: number or parts
    @return: list of dict_num dictionaries of the same size
    """
    it = iter(dictionary)
    for i in range(0, len(dictionary), dict_num):
        yield {k: dictionary[k] for k in islice(it, dict_num)}


def multi_run_text_similarities_wrapper(args):
    """
    Wrapper for passing more arguments to compute_similarity_of_texts in parallel way
    @param args: Arguments of the function
    @return: call the compute_similarity_of_texts in parallel way
    """
    return compute_text_similarities_parallelly(*args)


def compute_similarity_of_codes(dataset1, dataset2, product_pairs_idx):
    """
    Compute the ratio of matching codes in corresponding products pairs
    @param dataset1: first dataframe with products' codes
    @param dataset2: second dataframe with products' codes
    @param product_pairs_idx: dict with indices of filtered possible matching pairs
    @return: ratio of common codes in corresponding products pairs
    """
    similarity_scores = []

    for product_idx, corresponding_indices in product_pairs_idx.items():
        product1 = dataset1.loc[[product_idx]].values[0]
        for product2_idx in corresponding_indices:
            product2 = dataset2.loc[[product2_idx]].values[0]
            matches = 0
            for item1 in product1:
                for item2 in product2:
                    if item1 == item2:
                        matches += 1
            score = matches / len(product1)
            if matches == 0 and len(product1) > 0 and len(product2) > 0:
                score = -1

            similarity_scores.append(score)

    return similarity_scores


def remove_id_marks(product_ids):
    ids_list = list(set(product_ids['all_ids_list']))

    for e in range(len(ids_list)):
        ids_list[e] = ids_list[e].replace(ID_MARK, '')

    return ids_list

def merge_ids_and_codes(product_ids_and_codes):
    union = list(set(product_ids_and_codes['all_ids_list']) | set(product_ids_and_codes['code']))

    for e in range(len(union)):
        union[e] = union[e].replace(ID_MARK, '')

    return union

def create_text_similarities_data(dataset1, dataset2, product_pairs_idx, tf_idfs, descriptive_words,
                                  dataset2_starting_index, pool, num_cpu):
    """
    Compute all the text-based similarities for the product pairs
    @param dataset1: first dataset of all products
    @param dataset2: second dataset of all products
    @param product_pairs_idx: dict with indices of filtered possible matching pairs
    @param tf_idfs: tf.idfs of all words from both datasets
    @param descriptive_words: descriptive words from both datasets
    @param pool: parallelling object
    @param num_cpu: number of processes
    @param dataset2_starting_index: starting index of the data from second dataset in tf_idfs and descriptive_words
    @return: Similarity scores for the product pairs
    """
    dataset1_subsets = [dataset1.iloc[list(product_pairs_idx_part.keys())] for product_pairs_idx_part in
                        chunks(product_pairs_idx, round(len(product_pairs_idx) / num_cpu))]
    dataset2_subsets = [[dataset2.iloc[d] for d in list(product_pairs_idx_part.values())] for product_pairs_idx_part in
                        chunks(product_pairs_idx, round(len(product_pairs_idx) / num_cpu))]
    df_all_similarities_list = pool.map(multi_run_text_similarities_wrapper,
                                        [(dataset1_subsets[i], dataset2_subsets[i], descriptive_words, tf_idfs,
                                          dataset2_starting_index) for i in
                                         range(0, len(dataset1_subsets))])
    df_all_similarities = pd.concat(df_all_similarities_list, ignore_index=True)

    # in case no new similarities were computed
    if len(df_all_similarities) == 0:
        return df_all_similarities

    # for each column compute the similarity of product pairs selected after filtering
    # specification comparison with units and values preprocessed as specification
    df_all_similarities['specification_key_matches'] = 0
    df_all_similarities['specification_key_value_matches'] = 0

    if 'specification' in dataset1.columns and 'specification' in dataset2.columns:
        specification_similarity = compute_similarity_of_specifications(dataset1['specification'],
                                                                        dataset2['specification'], product_pairs_idx)
        specification_similarity = pd.DataFrame(specification_similarity)
        df_all_similarities['specification_key_matches'] = specification_similarity['matching_keys']
        df_all_similarities['specification_key_value_matches'] = specification_similarity['matching_keys_values']

    # TODO this should be parallel
    if 'code' in dataset1.columns and 'code' in dataset2.columns:
        df_all_similarities['code'] = pd.Series(
            compute_similarity_of_codes(
                dataset1['code'],
                dataset2['code'],
                product_pairs_idx
            )
        )
    else:
        df_all_similarities['code'] = 0

    df_all_similarities['codes_and_ids'] = pd.Series(
        compute_similarity_of_codes(
            dataset1[['code', 'all_ids_list']].apply(merge_ids_and_codes, axis=1) if 'code' in dataset1 else dataset1[['all_ids_list']].apply(remove_id_marks, axis=1),
            dataset2[['code', 'all_ids_list']].apply(merge_ids_and_codes, axis=1) if 'code' in dataset2 else dataset2[['all_ids_list']].apply(remove_id_marks, axis=1),
            product_pairs_idx
        )
    )

    df_all_similarities = df_all_similarities.dropna(axis=1, how='all')
    return df_all_similarities


def create_empty_dataframe_with_ids(dataset1, dataset2):
    """
    Create dataframe for text similarity results with ids of possible pairs after filtration
    @param dataset1: dataframe of the products from the first dataset
    @param dataset2: dataframe of the products from the second dataset
    @return: dataframe with ids of compared products
    """
    dataset1_ids = []
    dataset2_ids = []
    dataset1_hashes = []
    dataset2_hashes = []
    for i, (id1, hash1) in enumerate(zip(dataset1['id'].values, dataset1['all_texts_hash'].values)):
        dataset1_ids += [id1] * len(dataset2[i])
        dataset2_ids += list(dataset2[i]['id'].values)
        dataset1_hashes += [hash1] * len(dataset2[i])
        dataset2_hashes += list(dataset2[i]['all_texts_hash'].values)
    df_all_similarities = pd.DataFrame(columns=['id1', 'id2', 'all_texts_hash1', 'all_texts_hash2'])
    df_all_similarities['id1'] = dataset1_ids
    df_all_similarities['id2'] = dataset2_ids
    df_all_similarities['all_texts_hash1'] = dataset1_hashes
    df_all_similarities['all_texts_hash2'] = dataset2_hashes
    return df_all_similarities


def compute_text_similarities_parallelly(dataset1, dataset2, descriptive_words, tf_idfs, dataset2_starting_index):
    """
    Compute similarity score of each pair in both datasets parallelly for each column
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @param descriptive_words: descriptive words from both datasets
    @param tf_idfs: tf.idfs of all words from both datasets
    @param dataset2_starting_index: starting index of the data from second dataset in tf_idfs and descriptive_words
    @return: dataset of pair similarity scores
    """
    df_all_similarities = create_empty_dataframe_with_ids(dataset1, dataset2)
    similarities_to_compute = SIMILARITIES_TO_BE_COMPUTED
    for column in COLUMNS_TO_BE_PREPROCESSED:
        if column in dataset1 and column in dataset2[0]:
            similarities_to_ignore = KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED[
                column] if column in KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED else []
            similarities_to_compute = [similarity for similarity in similarities_to_compute if
                                       similarity not in similarities_to_ignore]
            tf_idfs_column = tf_idfs[column] if column in tf_idfs else None
            descriptive_words_column = descriptive_words[column] if column in descriptive_words else None
            columns_similarity = compute_similarity_of_texts(dataset1[column], [item[column] for item in dataset2],
                                                             tf_idfs_column, descriptive_words_column,
                                                             similarities_to_compute, dataset2_starting_index
                                                             )
            columns_similarity = pd.DataFrame(columns_similarity)

            for similarity_name, similarity_value in columns_similarity.items():
                if dataset2[0].iloc[0][column] == ["column_did_not_exist_in_scraper"] or dataset1.iloc[0][column] == ["column_did_not_exist_in_scraper"]:
                    df_all_similarities[f'{column}_{similarity_name}'] = 0
                else:
                    df_all_similarities[f'{column}_{similarity_name}'] = similarity_value
        else:
            for similarity_name in SIMILARITIES_TO_BE_COMPUTED:
                df_all_similarities[f'{column}_{similarity_name}'] = 0
    dataset1_keywords = dataset1.loc[:, dataset1.columns.str.contains('_list')]
    dataset2_keywords = [item.loc[:, item.columns.str.contains('_list')] for item in dataset2]
    keywords_similarity = compute_similarity_of_keywords(dataset1_keywords, dataset2_keywords)
    keywords_similarity = pd.DataFrame(keywords_similarity)
    for similarity_name, similarity_value in keywords_similarity.items():
        df_all_similarities[similarity_name] = similarity_value
    return df_all_similarities
