import os
import shutil
import sys

# DO NOT REMOVE
# Adding the higher level directories to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import pandas as pd
import copy
from ..evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier
from .dataset_handler import preprocess_data_without_saving, preprocess_textual_data, COLUMNS, \
    preprocess_data_before_training
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


def filter_products_with_no_similar_words(product, product_descriptive_words, dataset, dataset_start_index, descriptive_words):
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
        second_product_descriptive_words = descriptive_words.iloc[idx+dataset_start_index].values
        descriptive_words_sim = compute_descriptive_words_similarity(
            product_descriptive_words,
            second_product_descriptive_words
        )
        if len(set(product['name']) & set(second_product['name'])) > 0 and descriptive_words_sim > 5:
            data_subset = data_subset.append(second_product)
    return data_subset


def load_model_create_dataset_and_predict_matches(
        dataset1,
        dataset2,
        images_kvs1_client,
        images_kvs2_client,
        classifier,
        model_key_value_store_client=None
):
    """
    For each product in first dataset find same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param classifier: Classifier used for product matching
    @param model_key_value_store_client: key-value-store client where the classifier model is stored
    @return: List of same products for every given product
    """
    classifier = setup_classifier(classifier)
    classifier.load(key_value_store=model_key_value_store_client)

    # preprocess data
    dataset1_copy = copy.deepcopy(dataset1)
    dataset2_copy = copy.deepcopy(dataset2)
    dataset1_copy = preprocess_textual_data(dataset1_copy,
                                            id_detection=False,
                                            color_detection=False,
                                            brand_detection=False,
                                            units_detection=False)
    dataset2_copy = preprocess_textual_data(dataset2_copy,
                                            id_detection=False,
                                            color_detection=False,
                                            brand_detection=False,
                                            units_detection=False)
    dataset1 = preprocess_textual_data(dataset1)
    dataset2 = preprocess_textual_data(dataset2)

    # create tf_idfs
    tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_copy, dataset2_copy, COLUMNS)

    # filter product pairs
    pairs_dataset_idx = filter_possible_product_pairs(dataset1, dataset2, descriptive_words)

    # preprocess data
    preprocessed_pairs = pd.DataFrame(preprocess_data_without_saving(dataset1, dataset2, tf_idfs, descriptive_words,
                                                                     dataset_folder='.',
                                                                     dataset_dataframe=pairs_dataset_idx,
                                                                     dataset_images_kvs1=images_kvs1_client,
                                                                     dataset_images_kvs2=images_kvs2_client
                                                                     ))

    # TODO remove after speed testing
    print(preprocessed_pairs.count())
    print(preprocessed_pairs)

    preprocessed_pairs['predicted_match'], preprocessed_pairs['predicted_scores'] = classifier.predict(
        preprocessed_pairs)
    predicted_matches = preprocessed_pairs[preprocessed_pairs['predicted_match'] == 1][
        ['name1', 'id1', 'name2', 'id2', 'predicted_scores']
    ]
    return predicted_matches


def filter_possible_product_pairs(dataset1, dataset2, descriptive_words):
    """
    Filter possible pairs of two datasets using price similar words and descriptive words filter
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return dict with key as indices of products from the first dataset and values as indices of filtered possible matching products from second dataset
    """
    dataset2_no_price_idx = dataset2.index[dataset2['price'] == 0].tolist()
    idx_start = 0
    idx_to = 0
    pairs_dataset_idx = {}
    dataset_start_index = len(dataset1)
    for idx, product in dataset1.iterrows():
        data_subset_idx, idx_start, idx_to = filter_products(product, descriptive_words['all_texts'].iloc[idx].values, dataset2, idx_start, idx_to, dataset_start_index, descriptive_words)
        if len(data_subset_idx) == 0:
            print(f'No corresponding product for product "{product["name"]}" at index {idx}')
        if len(dataset2_no_price_idx) != 0:
            data_subset_idx = data_subset_idx + dataset2_no_price_idx
        pairs_dataset_idx[idx] = data_subset_idx
    pairs_dataset_idx = dict(sorted(pairs_dataset_idx.items()))
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


def filter_products(product, product_descriptive_words,  dataset, idx_from, idx_to, dataset_start_index, descriptive_words):
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

    data_filtered = filter_products_with_no_similar_words(product, product_descriptive_words, data_filtered, dataset_start_index, descriptive_words['all_texts'])
    return data_filtered.index.values, idx_from, idx_to


def load_data_and_train_model(
        classifier_type,
        dataset_folder='',
        dataset_dataframe=None,
        images_kvs_1_client=None,
        images_kvs_2_client=None,
        output_key_value_store_client=None
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
    data = preprocess_data_before_training(
        dataset_folder=os.path.join(os.getcwd(), dataset_folder),
        dataset_dataframe=dataset_dataframe,
        dataset_images_kvs1=images_kvs_1_client,
        dataset_images_kvs2=images_kvs_2_client
    )
    # data.to_csv('data.csv', index=False)
    classifier = setup_classifier(classifier_type)
    train_data, test_data = train_classifier(classifier, data)
    train_stats, test_stats = evaluate_classifier(classifier, train_data, test_data, plot_and_print_stats=False)
    classifier.save(key_value_store=output_key_value_store_client)
    return train_stats, test_stats


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
