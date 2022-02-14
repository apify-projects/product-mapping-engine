import bisect
import os
import sys

import numpy as np

# DO NOT REMOVE
# Adding the higher level directories to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from multiprocessing import Pool
import pandas as pd
import copy
from ..evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier
from .dataset_handler import create_image_and_text_similarities, preprocess_textual_data
from .texts.compute_texts_similarity import create_tf_idfs_and_descriptive_words, compute_descriptive_words_similarity
from ..configuration import COLUMNS_TO_BE_PREPROCESSED, MIN_DESCRIPTIVE_WORDS_FOR_MATCH, \
    MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH, \
    MIN_MATCH_PRICE_RATIO, MAX_MATCH_PRICE_RATIO, IS_ON_PLATFORM, SAVE_PREPROCESSED_PAIRS, PERFORM_ID_DETECTION, \
    PERFORM_COLOR_DETECTION, PERFORM_BRAND_DETECTION, PERFORM_UNITS_DETECTION, SAVE_SIMILARITIES, \
    SIMILARITIES_TO_IGNORE


def filter_products_with_no_similar_words(product, product_descriptive_words, dataset, dataset_start_index,
                                          descriptive_words):
    """
    Filter products from the dataset of products with no same words and with low ratio of descriptive words
    @param product: product used as filter
    @param product_descriptive_words: descriptive words of the product
    @param dataset: dataset of products to be filtered
    @param dataset_start_index: starting index to index the products from second dataset in descriptive words
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return: dataset with products containing at least one same word as source product
    """
    data_subset = pd.DataFrame(columns=dataset.columns.tolist())
    for idx, second_product in dataset.iterrows():
        second_product_descriptive_words = descriptive_words.iloc[idx + dataset_start_index].values
        descriptive_words_sim = compute_descriptive_words_similarity(
            product_descriptive_words,
            second_product_descriptive_words
        )
        '''
        if "sony" in second_product['name']:
            print(f"{set(product['name'])} vs {set(second_product['name'])} = {len(set(product['name']) & set(second_product['name']))}")
        '''
        if len(set(product['name']) & set(second_product['name'])) >= MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH and \
                descriptive_words_sim >= MIN_DESCRIPTIVE_WORDS_FOR_MATCH:
            data_subset = data_subset.append(second_product)

    return data_subset


def multi_run_text_preprocessing_wrapper(args):
    """
    Wrapper for passing more arguments to preprocess_textual_data in parallel way
    @param args: Arguments of the function
    @return: call the preprocess_textual_data in parallel way
    """
    return preprocess_textual_data(*args)


def parallel_text_preprocessing(pool, num_cpu, dataset, id_detection, color_detection, brand_detection,
                                units_detection):
    """
    Preprocessing of all textual data in dataset in parallel way
    @param pool: parallelling object
    @param num_cpu: number of processes
    @param dataset: dataframe to be preprocessed
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @return preprocessed dataset
    """
    dataset_list = np.array_split(dataset, num_cpu)
    dataset_list_prepro = pool.map(multi_run_text_preprocessing_wrapper,
                                   [(item, id_detection, color_detection, brand_detection, units_detection) for item in
                                    dataset_list])
    dataset_prepro = pd.concat(dataset_list_prepro)
    return dataset_prepro


def load_model_create_dataset_and_predict_matches(
        dataset1,
        dataset2,
        images_kvs1_client,
        images_kvs2_client,
        classifier_type,
        model_key_value_store_client=None,
        task_id="basic"
):
    """
    For each product in first dataset find same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param classifier_type: Classifier used for product matching
    @param model_key_value_store_client: key-value-store client where the classifier model is stored
    @param task_id: unique identification of the current Product Mapping task
    @return: List of same products for every given product
    """
    classifier = setup_classifier(classifier_type)
    classifier.load(key_value_store=model_key_value_store_client)
    pair_identifications_file_path = "pair_identifications_{}.csv".format(task_id)
    preprocessed_pairs_file_path = "preprocessed_pairs_{}.csv".format(task_id)
    preprocessed_pairs_file_exists = os.path.exists(preprocessed_pairs_file_path)

    if SAVE_PREPROCESSED_PAIRS and preprocessed_pairs_file_exists:
        preprocessed_pairs = pd.read_csv(preprocessed_pairs_file_path)
    else:
        preprocessed_pairs = prepare_data_for_classifier(dataset1, dataset2, images_kvs1_client,
                                                         images_kvs2_client,
                                                         filter_data=True)
    if not IS_ON_PLATFORM and SAVE_PREPROCESSED_PAIRS:
        preprocessed_pairs.to_csv(preprocessed_pairs_file_path, index=False)

    if 'index1' in preprocessed_pairs.columns and 'index2' in preprocessed_pairs.columns:
        preprocessed_pairs = preprocessed_pairs.drop(['index1', 'index2'], axis=1)

    if SIMILARITIES_TO_IGNORE:
        preprocessed_pairs = preprocessed_pairs.drop(SIMILARITIES_TO_IGNORE, axis=1, errors='ignore')

    preprocessed_pairs['predicted_match'], preprocessed_pairs['predicted_scores'] = classifier.predict(
        preprocessed_pairs.drop(['id1', 'id2'], axis=1))

    if not IS_ON_PLATFORM:
        evaluate_executor_results(classifier, preprocessed_pairs, task_id)

    predicted_matches = preprocessed_pairs[preprocessed_pairs['predicted_match'] == 1][
        ['id1', 'id2', 'predicted_scores']
    ]
    return predicted_matches


def prepare_data_for_classifier(dataset1, dataset2, images_kvs1_client, images_kvs2_client,
                                filter_data):
    """
    Preprocess data, possibly filter data pairs and compute similarities
    @param dataset1: Source dataframe of products
    @param dataset2: Target dataframe with products to be searched in for the same products
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param filter_data: True whether filtering during similarity computations should be performed
    @return: dataframe with image and text similarities
    """
    # setup parallelling stuff
    pool = Pool()
    num_cpu = os.cpu_count()

    # preprocess data
    print("Text preprocessing started")
    dataset1_without_marks = copy.deepcopy(dataset1)
    dataset2_without_marks = copy.deepcopy(dataset2)
    dataset1_without_marks = parallel_text_preprocessing(pool, num_cpu, dataset1_without_marks, False, False, False, False)
    dataset2_without_marks = parallel_text_preprocessing(pool, num_cpu, dataset2_without_marks, False, False, False, False)
    dataset1 = parallel_text_preprocessing(pool, num_cpu, dataset1, PERFORM_ID_DETECTION, PERFORM_COLOR_DETECTION,
                                           PERFORM_BRAND_DETECTION, PERFORM_UNITS_DETECTION)
    dataset2 = parallel_text_preprocessing(pool, num_cpu, dataset2, PERFORM_ID_DETECTION, PERFORM_COLOR_DETECTION,
                                           PERFORM_BRAND_DETECTION, PERFORM_UNITS_DETECTION)
    # create tf_idfs
    tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_without_marks, dataset2_without_marks,
                                                                      COLUMNS_TO_BE_PREPROCESSED)
    print("Text preprocessing finished")

    if filter_data:
        # filter product pairs
        print("Filtering started")
        pairs_dataset_idx = filter_possible_product_pairs(dataset1_without_marks, dataset2_without_marks, descriptive_words, pool, num_cpu)
        pairs_count = 0
        for key, target_ids in pairs_dataset_idx.items():
            pairs_count += len(target_ids)

        print(f"Filtered to {pairs_count} pairs")
        print("Filtering ended")
    else:
        pairs_dataset_idx = {}
        for i in range(0, len(dataset1)):
            pairs_dataset_idx[i] = [i]

    # create image and text similarities
    print("Similarities creation started")
    image_and_text_similarities = create_image_and_text_similarities(dataset1, dataset2, tf_idfs, descriptive_words,
                                                                     pool, num_cpu,
                                                                     dataset_folder='.',
                                                                     dataset_dataframe=pairs_dataset_idx,
                                                                     dataset_images_kvs1=images_kvs1_client,
                                                                     dataset_images_kvs2=images_kvs2_client
                                                                     )

    print("Similarities creation ended")
    return image_and_text_similarities


def evaluate_executor_results(classifier, preprocessed_pairs, task_id):
    """
    Evaluate results of executors predictions and filtering
    @param classifier: classifier used for predicting pairs
    @param preprocessed_pairs: dataframe with predicted and filtered pairs
    @param task_id: unique identification of the currently evaluated Product Mapping task
    """
    print('{}_unlabeled_data.csv'.format(task_id))
    labeled_dataset = pd.read_csv('{}_unlabeled_data.csv'.format(task_id))
    print("Labeled dataset")
    print(labeled_dataset.shape)

    matching_pairs = labeled_dataset[['id1', 'id2', 'name1', 'name2', 'url1', 'url2', 'match', 'price1', 'price2']]
    predicted_pairs = preprocessed_pairs[['id1', 'id2', 'predicted_scores', 'predicted_match']]

    print("Predicted pairs")
    print(predicted_pairs[predicted_pairs['predicted_match'] == 1].shape)

    merged_data = predicted_pairs.merge(matching_pairs, on=['id1', 'id2'], how='outer')
    #merged_data = merged_data.drop_duplicates(subset=['id1', 'id2'])
    #merged_data = merged_data[merged_data['url2'].notna() & merged_data['url1'].notna()]

    predicted_pairs[predicted_pairs['predicted_match'] == 1][['id1', 'id2']].to_csv("predicted.csv")

    merged_data['match'] = merged_data['match'].fillna(0)
    merged_data['predicted_scores'] = merged_data['predicted_scores'].fillna(0)
    merged_data['predicted_match'] = merged_data['predicted_match'].fillna(0)

    merged_data_to_save = merged_data[merged_data['match'] == 1]
    merged_data_to_save = merged_data_to_save[merged_data_to_save['predicted_match'] == 0]
    merged_data_to_save.to_csv("merged.csv")


    merged_data = merged_data.drop(['id1', 'id2', 'url1', 'url2', 'price1', 'price2'], axis=1)
    stats = evaluate_classifier(classifier, merged_data, merged_data, False)
    print(stats)


def multi_run_filter_wrapper(args):
    """
    Wrapper for passing more arguments to filter_possible_product_pairs_parallelly in parallel way
    @param args: Arguments of the function
    @return: call the filter function in parallel way
    """
    return filter_possible_product_pairs_parallelly(*args)


def filter_possible_product_pairs(dataset1, dataset2, descriptive_words, pool, num_cpu):
    """
    Filter possible pairs of two datasets using price similar words and descriptive words filter
    @param pool: parallelling object
    @param num_cpu: number of processes
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return dict with key as indices of products from the first dataset and
            values as indices of filtered possible matching products from second dataset
    """
    dataset2_no_price_idx = dataset2.index[dataset2['price'] == 0].tolist()
    dataset1 = dataset1.sort_values(by=['price'])
    dataset2 = dataset2.sort_values(by=['price'])
    dataset_start_index = len(dataset1)

    dataset_list = np.array_split(dataset1, num_cpu)
    filtered_indices_dicts = pool.map(multi_run_filter_wrapper,
                                      [(dataset_subset, dataset2, dataset2_no_price_idx, dataset_start_index,
                                        descriptive_words) for dataset_subset in dataset_list])
    pairs_dataset_idx = {}
    for filtered_dict in filtered_indices_dicts:
        pairs_dataset_idx.update(filtered_dict)
    return pairs_dataset_idx


def filter_possible_product_pairs_parallelly(dataset1, dataset2, dataset2_no_price_idx, dataset_start_index,
                                             descriptive_words):
    """
    Filter possible pairs of two datasets using price similar words and descriptive words filter in parallel way
    @param dataset_start_index: starting index to index the products from second dataset in descriptive words
    @param dataset2_no_price_idx: indices of the products from second dataset without specified prices
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return dict with key as indices of products from the first dataset and
            values as indices of filtered possible matching products from second dataset
    """
    product = dataset1.iloc[0, :]
    idx_start, idx_to = 0, 0

    if 'price' in product.index.values and 'price' in dataset2:
        idx_start = bisect.bisect_left(dataset2['price'].values, MIN_MATCH_PRICE_RATIO * product['price'])
        idx_to = bisect.bisect(dataset2['price'].values, product['price'] * MAX_MATCH_PRICE_RATIO)
        if idx_to == len(dataset2):
            idx_to -= 1

    pairs_dataset_idx = {}
    for idx, product in dataset1.iterrows():
        data_subset_idx, idx_start, idx_to = filter_products(product, descriptive_words['all_texts'].iloc[idx].values,
                                                             dataset2, idx_start, idx_to, dataset_start_index,
                                                             descriptive_words)
        if len(data_subset_idx) == 0:
            print(f'No corresponding product for product "{product["name"]}" at index {idx}')
        if len(dataset2_no_price_idx) != 0:
            data_subset_idx = data_subset_idx + dataset2_no_price_idx
        pairs_dataset_idx[idx] = data_subset_idx
    return pairs_dataset_idx


def create_dataset_for_predictions(product, maybe_the_same_products):
    """
    Create one dataset for model to predict matches that will consist of following
    pairs: given product with every product from the dataset of possible matches
    @param product: product to be compared with all products in the dataset
    @param maybe_the_same_products: dataset of products that are possibly the same as given product
    @return: one dataset for model to predict pairs
    """
    maybe_the_same_products = maybe_the_same_products.rename(columns=lambda s: s + '2')
    final_dataset = pd.DataFrame(columns=product.index.values)
    for _ in range(0, len(maybe_the_same_products.index)):
        final_dataset = final_dataset.append(product, ignore_index=True)
    final_dataset.reset_index(drop=True, inplace=True)
    maybe_the_same_products.reset_index(drop=True, inplace=True)
    final_dataset = final_dataset.rename(columns=lambda s: s + '1')
    final_dataset = pd.concat([final_dataset, maybe_the_same_products], axis=1)
    return final_dataset


def filter_products(product, product_descriptive_words, dataset, idx_from, idx_to, dataset_start_index,
                    descriptive_words):
    """
    Filter products in dataset according to the price, category and word similarity to reduce number of comparisons
    @param product: given product for which we want to filter dataset
    @param product_descriptive_words: descriptive words of the product
    @param dataset:  dataset of products to be filtered sorted according to the price
    @param idx_from: starting index for searching for product with similar price in dataset
    @param idx_to: ending index for searching for product with similar price in dataset
    @param dataset_start_index: starting index to index the products from second dataset in descriptive words
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return: Filtered dataset of products that are possibly the same as given product
    """
    if 'price' not in product.index.values or 'price' not in dataset:
        data_filtered = dataset
    else:
        last_price = dataset.iloc[idx_from]['price']

        min_price = product['price'] * MIN_MATCH_PRICE_RATIO
        while last_price < min_price and idx_from < len(dataset) - 1:
            idx_from += 1
            last_price = dataset.iloc[idx_from]['price']

        last_price = dataset.iloc[idx_to]['price']
        max_price = product['price'] * MAX_MATCH_PRICE_RATIO
        while last_price <= max_price and idx_to < len(dataset) - 1:
            idx_to += 1
            last_price = dataset.iloc[idx_to]['price']

        if idx_to == len(dataset) - 1:
            data_filtered = dataset.iloc[idx_from:]
        else:
            data_filtered = dataset.iloc[idx_from:idx_to]

    '''
    if 'category' in product.index.values and 'category' in dataset:
        data_filtered = data_filtered[
            data_filtered['category'] == product['category'] or data_filtered['category'] is None]
    '''

    data_filtered = filter_products_with_no_similar_words(product, product_descriptive_words, data_filtered,
                                                          dataset_start_index, descriptive_words['all_texts'])
    return data_filtered.index.values, idx_from, idx_to


def load_data_and_train_model(
        classifier_type,
        dataset_folder='',
        dataset_dataframe=None,
        images_kvs1_client=None,
        images_kvs2_client=None,
        output_key_value_store_client=None,
        task_id="basic"
):
    """
    Load dataset and train and save model
    @param classifier_type: classifier type
    @param dataset_folder: (optional) folder containing data
    @param dataset_dataframe: dataframe of pairs to be compared
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param output_key_value_store_client: key-value-store client where the trained model should be stored
    @param task_id: unique identification of the current Product Mapping task
    @return:
    """
    similarities_file_path = "similarities_{}.csv".format(task_id)
    similarities_file_exists = os.path.exists(similarities_file_path)

    if SAVE_SIMILARITIES and similarities_file_exists:
        similarities = pd.read_csv(similarities_file_path)
    else:
        product_pairs = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
            os.path.join(dataset_folder, "product_pairs.csv"))

        product_pairs1 = product_pairs.filter(regex='1')
        product_pairs1.columns = product_pairs1.columns.str.replace("1", "")
        product_pairs2 = product_pairs.filter(regex='2')
        product_pairs2.columns = product_pairs2.columns.str.replace("2", "")
        preprocessed_pairs = prepare_data_for_classifier(product_pairs1, product_pairs2, images_kvs1_client,
                                                         images_kvs2_client, filter_data=False)
        if 'index1' in preprocessed_pairs.columns and 'index2' in preprocessed_pairs.columns:
            preprocessed_pairs = preprocessed_pairs.drop(columns=['index1', 'index2'])
        similarities_to_concat = [preprocessed_pairs]
        if 'match' in product_pairs.columns:
            similarities_to_concat.append(product_pairs['match'])
        similarities = pd.concat(similarities_to_concat, axis=1)
        if not IS_ON_PLATFORM and SAVE_SIMILARITIES:
            similarities.to_csv(similarities_file_path, index=False)

    classifier = setup_classifier(classifier_type)
    if SIMILARITIES_TO_IGNORE:
        similarities = similarities.drop(SIMILARITIES_TO_IGNORE, axis=1, errors='ignore')
    train_stats, test_stats = train_classifier(classifier, similarities.drop(columns=['id1', 'id2']))
    classifier.save(key_value_store=output_key_value_store_client)
    feature_names = [col for col in similarities.columns if col not in ['id1', 'id2', 'match']]
    if not classifier.use_pca:
        classifier.print_feature_importance(feature_names)
    return train_stats, test_stats


def filter_and_save_fp_and_fn(original_dataset):
    """
    Filter and save FP and FN from predicted matches
    @param original_dataset: dataframe with original data
    @return:
    """
    original_dataset['index'] = original_dataset.index
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    train_test_data = pd.concat([train_data, test_data])
    predicted_pairs = train_test_data.join(original_dataset, on='index1', how='left')
    joined_datasets = predicted_pairs.drop(['index1', 'index'], 1)
    fn_train = joined_datasets[(joined_datasets['match'] == 1) & (joined_datasets['predicted_match'] == 0)]
    fp_train = joined_datasets[(joined_datasets['match'] == 0) & (joined_datasets['predicted_match'] == 1)]
    fn_train.to_csv(f'fn_dataset.csv', index=False)
    fp_train.to_csv(f'fp_dataset.csv', index=False)
