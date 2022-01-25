import bisect
import os
import shutil
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
from .dataset_handler import preprocess_data_without_saving, preprocess_textual_data, COLUMNS, \
    create_text_similarities_data, create_image_similarities_data
from ..preprocessing.texts.text_preprocessing import preprocess_text
from .texts.compute_texts_similarity import create_tf_idfs_and_descriptive_words, compute_descriptive_words_similarity


def main(**kwargs):
    preprocess_text()
    # Load dataset and train and save model
    load_data_and_train_model('data/extra_dataset/dataset', 'DecisionTree')
    # matching_pairs = load_model_and_predict_matches('data/wdc_dataset/dataset/preprocessed', 'DecisionTree')

    # Load datasets and model and fund matching pairs
    dataset1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/extra_dataset/dataset/amazon.csv')
    dataset2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/extra_dataset/dataset/extra.csv')
    results_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/extra_dataset/results')
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
    os.mkdir(results_directory)
    output_file = os.path.join(results_directory, 'matching_pairs.csv')
    matching_pairs = load_model_create_dataset_and_predict_matches(
        pd.read_csv(dataset1),
        pd.read_csv(dataset2),
        'DecisionTree'
    )
    matching_pairs.to_csv(output_file, index=False)


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
        if len(set(product['name']) & set(second_product['name'])) > 0 and descriptive_words_sim > 5:
            data_subset = data_subset.append(second_product)
    return data_subset


def multi_run_text_prepro_wrapper(args):
    """
    Wrapper for passing more arguments to preprocess_textual_data in parallel way
    @param args: Arguments of the function
    @return: call the preprocess_textual_data in parallel way
    """
    return preprocess_textual_data(*args)


def parallel_text_preprocessing(pool, num_cpu, dataset, id_detection, color_detection, brand_detection, units_detection):
    """
    Preprocessing of all textual data in dataset in parallel way
    @param pool: parallelising object
    @param num_cpu: number of processes
    @param dataset: dataset to be preprocessed
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @return preprocessed dataset
    """
    dataset_list = np.array_split(dataset, num_cpu)
    dataset_list_prepro = pool.map(multi_run_text_prepro_wrapper,
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
        task_id="basic",
        is_on_platform=False,
        save_preprocessed_pairs=True
):
    """
    For each product in first dataset find same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param classifier_type: Classifier used for product matching
    @param model_key_value_store_client: key-value-store client where the classifier model is stored
    @return: List of same products for every given product
    """
    classifier = setup_classifier(classifier_type)
    classifier.load(key_value_store=model_key_value_store_client)
    pair_identifications_file_path = "pair_identifications_{}.csv".format(task_id)
    preprocessed_pairs_file_path = "preprocessed_pairs_{}.csv".format(task_id)
    preprocessed_pairs_file_exists = os.path.exists(preprocessed_pairs_file_path)

    if save_preprocessed_pairs and preprocessed_pairs_file_exists:
        preprocessed_pairs = pd.read_csv(preprocessed_pairs_file_path)
        pair_identifications = pd.read_csv(pair_identifications_file_path)
    else:
        print("Text preprocessing started")

        # setup parallelising stuff
        pool = Pool()
        num_cpu = os.cpu_count()

        # preprocess data
        dataset1_copy = copy.deepcopy(dataset1)
        dataset2_copy = copy.deepcopy(dataset2)

        dataset1_copy = parallel_text_preprocessing(pool, num_cpu, dataset1_copy, False, False, False, False)
        dataset2_copy = parallel_text_preprocessing(pool, num_cpu, dataset2_copy, False, False, False, False)
        dataset1 = parallel_text_preprocessing(pool, num_cpu, dataset1, True, True, True, True)
        dataset2 = parallel_text_preprocessing(pool, num_cpu, dataset2, True, True, True, True)

        # create tf_idfs
        tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_copy, dataset2_copy, COLUMNS)

        print("Text preprocessing finished")

        # filter product pairs
        pairs_dataset_idx = filter_possible_product_pairs(dataset1, dataset2, descriptive_words, pool, num_cpu)
        print("Filtered to {} pairs".format(len(pairs_dataset_idx.keys())))

        pair_identifications = []
        for source_id, target_ids in pairs_dataset_idx.items():
            for target_id in target_ids:
                pair_identifications.append({
                    'id1': dataset1['id'][source_id],
                    'name1': dataset1['name'][source_id],
                    'id2': dataset2['id'][target_id],
                    'name2': dataset2['name'][target_id],
                })
        pair_identifications = pd.DataFrame(pair_identifications)

        # preprocess data
        preprocessed_pairs = pd.DataFrame(
            preprocess_data_without_saving(dataset1, dataset2, tf_idfs, descriptive_words, pool, num_cpu,
                                           dataset_folder='.',
                                           dataset_dataframe=pairs_dataset_idx,
                                           dataset_images_kvs1=images_kvs1_client,
                                           dataset_images_kvs2=images_kvs2_client
                                           ))

        if not is_on_platform and save_preprocessed_pairs:
            preprocessed_pairs.to_csv(preprocessed_pairs_file_path, index=False)
            pair_identifications.to_csv(pair_identifications_file_path)

    preprocessed_pairs['predicted_match'], preprocessed_pairs['predicted_scores'] = classifier.predict(
        preprocessed_pairs)
    preprocessed_pairs = pd.concat([pair_identifications, preprocessed_pairs], axis=1)
    predicted_matches = preprocessed_pairs[preprocessed_pairs['predicted_match'] == 1][
        ['name1', 'id1', 'name2', 'id2', 'predicted_scores']
    ]
    return predicted_matches


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
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return dict with key as indices of products from the first dataset and values as indices of filtered possible matching products from second dataset
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
    product = dataset1.iloc[0, :]
    idx_start, idx_to = 0, 0

    if 'price' in product.index.values and 'price' in dataset2:
        idx_start = bisect.bisect_left(dataset2['price'].values, product['price'] / 2)
        idx_to = bisect.bisect(dataset2['price'].values, product['price'] * 2)
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
        min_price = product['price'] / 2
        while last_price < min_price and idx_from < len(dataset) - 1:
            idx_from += 1
            last_price = dataset.iloc[idx_from]['price']

        last_price = dataset.iloc[idx_to]['price']
        max_price = product['price'] * 2
        while last_price <= max_price and idx_to < len(dataset) - 1:
            idx_to += 1
            last_price = dataset.iloc[idx_to]['price']
        data_filtered = dataset.iloc[idx_from:idx_to]
    if 'category' in product.index.values and 'category' in dataset:
        data_filtered = data_filtered[
            data_filtered['category'] == product['category'] or data_filtered['category'] == None]

    data_filtered = filter_products_with_no_similar_words(product, product_descriptive_words, data_filtered,
                                                          dataset_start_index, descriptive_words['all_texts'])
    return data_filtered.index.values, idx_from, idx_to


def load_data_and_train_model(
        classifier_type,
        dataset_folder='',
        dataset_dataframe=None,
        images_kvs_1_client=None,
        images_kvs_2_client=None,
        output_key_value_store_client=None,
        task_id = "basic",
        is_on_platform = False,
        save_similarities = True
):
    """
    Load dataset and train and save model
    @param dataset_folder: (optional) folder containing data
    @param classifier_type: classifier type
    @param dataset_dataframe: dataframe of pairs to be compared
    @param images_kvs_1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs_2_client: key-value-store client where the images for the target dataset are stored
    @param output_key_value_store_client: key-value-store client where the trained model should be stored
    @return:
    """
    similarities_file_path = "similarities_{}.csv".format(task_id)
    similarities_file_exists = os.path.exists(similarities_file_path)

    if save_similarities and similarities_file_exists:
        similarities = pd.read_csv(similarities_file_path)
    else:
        similarities = preprocess_data_before_training(
            dataset_folder=os.path.join(os.getcwd(), dataset_folder),
            dataset_dataframe=dataset_dataframe,
            dataset_images_kvs1=images_kvs_1_client,
            dataset_images_kvs2=images_kvs_2_client
        )

        if not is_on_platform and save_similarities:
            similarities.to_csv(similarities_file_path, index=False)

    classifier = setup_classifier(classifier_type)
    train_data, test_data = train_classifier(classifier, similarities)
    train_stats, test_stats = evaluate_classifier(classifier, train_data, test_data, plot_and_print_stats=not is_on_platform)
    classifier.save(key_value_store=output_key_value_store_client)
    return train_stats, test_stats


def preprocess_data_before_training(
        dataset_folder='',
        dataset_dataframe=None,
        dataset_images_kvs1=None,
        dataset_images_kvs2=None
):
    """
    For each pair of products compute their image and name similarity without saving anything
    @param dataset_folder: folder containing data to be preprocessed
    @param dataset_dataframe: dataframe of pairs to be compared
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @return: preprocessed data
    """
    print("Text preprocessing started")

    # setup parallelising stuff
    pool = Pool()
    num_cpu = os.cpu_count()

    product_pairs = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
        os.path.join(dataset_folder, "product_pairs.csv"))

    product_pairs1 = product_pairs.filter(regex='1')
    product_pairs1.columns = product_pairs1.columns.str.replace("1", "")
    product_pairs2 = product_pairs.filter(regex='2')
    product_pairs2.columns = product_pairs2.columns.str.replace("2", "")

    dataset1_copy = copy.deepcopy(product_pairs1)
    dataset2_copy = copy.deepcopy(product_pairs2)

    dataset1_copy = parallel_text_preprocessing(pool, num_cpu, dataset1_copy, False, False, False, False)
    dataset2_copy = parallel_text_preprocessing(pool, num_cpu, dataset2_copy, False, False, False, False)
    dataset1 = parallel_text_preprocessing(pool, num_cpu, product_pairs1, True, True, True, True)
    dataset2 = parallel_text_preprocessing(pool, num_cpu, product_pairs2, True, True, True, True)

    # create tf_idfs
    tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_copy, dataset2_copy, COLUMNS)
    product_pairs_idx = {}
    for i in range(0, len(dataset1)):
        product_pairs_idx[i] = [i]

    print("Text preprocessing finished")

    print("Text similarities computation started")
    text_similarities = create_text_similarities_data(dataset1, dataset2, product_pairs_idx, tf_idfs, descriptive_words,
                                                      pool, num_cpu)
    print("Text similarities computation finished")

    image_similarities = [0] * len(product_pairs)
    image_similarities = create_image_similarities_data(
                                                        pool,
                                                        num_cpu,
                                                        product_pairs[['id1', 'image1', 'id2', 'image2']].to_dict(
                                                            orient='records'),
                                                        dataset_folder=dataset_folder,
                                                        dataset_images_kvs1=dataset_images_kvs1,
                                                        dataset_images_kvs2=dataset_images_kvs2
                                                        )
    text_similarities = pd.DataFrame(text_similarities)
    print(text_similarities)
    image_similarities = pd.DataFrame(image_similarities, columns=['hash_similarity'])
    print(image_similarities)
    dataframes_to_concat = [text_similarities, image_similarities]

    if 'match' in product_pairs.columns:
        dataframes_to_concat.append(product_pairs['match'])

    return pd.concat(dataframes_to_concat, axis=1)


def load_model_and_predict_matches(
        dataset_folder,
        classifier_type,
        model_key_value_store_client=None
):
    """
    Directly load model and already created unlabeled dataset with product pairs and predict pairs
    @param dataset_folder: folder containing test data
    @param classifier_type: classifier type
    @param model_key_value_store_client: key-value-store client where the classifier model is stored
    @return: pair indices of matches
    """
    classifier = setup_classifier(classifier_type)
    classifier.load(key_value_store=model_key_value_store_client)
    data = preprocess_data_without_saving(os.path.join(os.getcwd(), dataset_folder))
    data['predicted_match'], data['predicted_scores'] = classifier.predict(data)
    return data[data['predicted_match'] == 1]


if __name__ == "__main__":
    main()
