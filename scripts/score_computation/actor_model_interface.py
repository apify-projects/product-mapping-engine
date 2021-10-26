import os
import shutil
import sys

# DO NOT REMOVE
# Adding the higher level directories to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import pandas as pd

from scripts.evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier
from scripts.score_computation.dataset_handler import preprocess_data_without_saving


def main(**kwargs):
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


def filter_products_with_no_similar_words(product, dataset):
    """
    Filter products from the dataset of products with no same words
    @param product: product used as filter
    @param dataset: dataset of products to be filtered
    @return: dataset with products containing at least one same word as source product
    """
    data_subset = pd.DataFrame(columns=dataset.columns.tolist())
    product_name_list = product['name'].split(' ')
    for idx, second_product in dataset.iterrows():
        second_product_name_list = second_product['name'].split(' ')
        o = set(product_name_list) & set(second_product_name_list)
        if len(set(product_name_list) & set(second_product_name_list)) > 0:
            data_subset = data_subset.append(second_product)
    return data_subset


def load_model_create_dataset_and_predict_matches(dataset1, dataset2, images_kvs1_client, images_kvs2_client,
                                                  classifier, model_key_value_store=None):
    """
    For each product in first dataset find same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param classifier: Classifier used for product matching
    @return: List of same products for every given product
    """
    classifier = setup_classifier(classifier)
    classifier.load(key_value_store=model_key_value_store)
    dataset1['price'] = pd.to_numeric(dataset1['price'])
    dataset2['price'] = pd.to_numeric(dataset2['price'])
    pairs_dataset_fragments = []
    for idx, product in dataset1.iterrows():
        data_subset = filter_products(product, dataset2)
        pairs = create_dataset_for_predictions(product, data_subset)
        pairs_dataset_fragments.append(pairs)

    pairs_dataset = pd.concat(pairs_dataset_fragments, axis=0)
    print(pairs_dataset.count())

    # preprocess data
    preprocessed_pairs = pd.DataFrame(preprocess_data_without_saving(
        dataset_folder='.',
        dataset_dataframe=pairs_dataset,
        dataset_images_kvs1=images_kvs1_client,
        dataset_images_kvs2=images_kvs2_client
    ))

    # TODO remove after speed testing
    print(preprocessed_pairs.count())

    pairs_dataset['predicted_match'], pairs_dataset['predicted_scores'] = classifier.predict(preprocessed_pairs)
    predicted_matches = pairs_dataset[pairs_dataset['predicted_match'] == 1][
        ['name1', 'id1', 'name2', 'id2', 'predicted_scores']]
    return predicted_matches


def create_dataset_for_predictions(product, maybe_the_same_products):
    """
    Create one dataset for model to predict matches that will consist of following pairs: given product with every product from the dataset of possible matches
    @param product: product to be compared with all products in the dataset
    @param maybe_the_same_products: dataset of products that are possibly the same as given product
    @return: one dataset for model to predict pairs
    """
    maybe_the_same_products = maybe_the_same_products.rename(columns=lambda s: s + '2')
    final_dataset = pd.DataFrame()
    for _ in range(0, len(maybe_the_same_products.index)):
        final_dataset = final_dataset.append(product, ignore_index=True)
    final_dataset = final_dataset.rename(columns=lambda s: s + '1')
    final_dataset.reset_index(drop=True, inplace=True)
    maybe_the_same_products.reset_index(drop=True, inplace=True)
    final_dataset = pd.concat([final_dataset, maybe_the_same_products], axis=1)
    return final_dataset


def filter_products(product, dataset):
    """
    Filter products in dataset according to the price, category and word similarity to reduce number of comparisons
    @param product: given product for which we want to filter dataset
    @param dataset: dataset of products to be filtered
    @return: Filtered dataset of products that are possibly the same as given product
    """
    data_filtered = dataset[
        (dataset['price'] / 2 <= product['price']) & (product['price'] <= dataset['price'] * 2)]
    if 'category' in dataset:
        data_filtered = data_filtered[data_filtered['category'] != data_filtered['category']]
    data_filtered = filter_products_with_no_similar_words(product, data_filtered)
    return data_filtered


def load_data_and_train_model(classifier_type, dataset_folder='', dataset_dataframe=None, images_kvs_1_client=None,
                              images_kvs_2_client=None, output_key_value_store=None):
    """
    Load dataset and train and save model
    @param dataset_folder: (optional) folder containing data
    @param classifier_type: classifier type
    @return:
    """
    data = preprocess_data_without_saving(
        dataset_folder=os.path.join(os.getcwd(), dataset_folder),
        dataset_dataframe=dataset_dataframe,
        dataset_images_kvs1=images_kvs_1_client,
        dataset_images_kvs2=images_kvs_2_client
    )
    # data.to_csv('data.csv', index=False)
    classifier = setup_classifier(classifier_type)
    train_data, test_data = train_classifier(classifier, data)
    train_stats, test_stats = evaluate_classifier(classifier, train_data, test_data, plot_and_print_stats=False)
    classifier.save(key_value_store=output_key_value_store)
    return train_stats, test_stats


def load_model_and_predict_matches(dataset_folder, classifier_type, model_key_value_store=None):
    """
    Directly load model and already created unlabeled dataset with product pairs and predict pairs
    @param dataset_folder: folder containing test data
    @param classifier_type: classifier type
    @return: pair indices of matches
    """
    classifier = setup_classifier(classifier_type)
    classifier.load(key_value_store=model_key_value_store)
    data = preprocess_data_without_saving(os.path.join(os.getcwd(), dataset_folder))
    data['predicted_match'], data['predicted_scores'] = classifier.predict(data)
    return data[data['predicted_match'] == 1]


if __name__ == "__main__":
    main()
