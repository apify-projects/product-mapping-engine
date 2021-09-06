import os
import sys

# DO NOT REMOVE
# Adding the higher level directory (scripts/) to sys.path so that we can import from the other folders
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from scripts.evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier
from scripts.score_computation.dataset_handler import preprocess_data_without_saving


# Load dataset and train and save model
def main(**kwargs):
    # load_data_and_train_classifier('data/wdc_dataset/dataset/preprocessed', 'DecisionTree')
    # matching_pairs = load_model_and_predict_matches('data/wdc_dataset/dataset/preprocessed', 'DecisionTree')
    matching_pairs = find_matching_products(pd.read_csv('data/wdc_dataset/dataset/preprocessed/amazon.csv'),
                                            pd.read_csv('data/wdc_dataset/dataset/preprocessed/extra.csv'),
                                            'DecisionTree')
    print(matching_pairs)


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


def find_matching_products(dataset1, dataset2, classifier):
    """
    For each product in first dataset predict same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param classifier: Classifier used for product matching
    @return: List of same products for every given product
    """
    predicted_matches = []
    classifier = setup_classifier(classifier, 'scripts/classifier_parameters/linear.json')
    classifier = classifier.load()
    dataset1['price'] = pd.to_numeric(dataset1['price'])
    dataset2['price'] = pd.to_numeric(dataset2['price'])
    for idx, product in dataset1.iterrows():
        data_subset = dataset2[
            (dataset2['price'] / 2 <= product['price']) & (product['price'] <= dataset2['price'] * 2)]
        # data_subset = data_subset[data_subset['class'] != data_subset['class']]

        # create pairs of product and each row from dataset
        data_subset = filter_products_with_no_similar_words(product, data_subset)
        data_subset = data_subset.rename(columns=lambda s: s + '2')
        rows = len(data_subset.index)
        data = pd.DataFrame()
        for _ in range(0, rows):
            data = data.append(product,ignore_index=True)
        data = data.rename(columns=lambda s: s + '1')
        data.reset_index(drop=True, inplace=True)
        data_subset.reset_index(drop=True, inplace=True)
        data = pd.concat([data, data_subset], axis=1)
        data = data.rename(columns={'url1': 'id1', 'url2': 'id2'})
        # preprocess data
        images_folder = None
        data = preprocess_data_without_saving(data, images_folder)

        data['predicted_match'], data['predicted_scores'] = classifier.predict(data)
        predicted_matches.append(data[data['predicted_match'] == 1])
    return predicted_matches


def load_data_and_train_classifier(dataset_folder, classifier):
    """
    Load dataset and train and save model
    @param dataset_folder: folder containing data
    @param classifier: classifier type
    @return:
    """
    data = preprocess_data_without_saving(os.path.join(os.getcwd(), dataset_folder))
    data.to_csv('data.csv', index=False)
    classifier = setup_classifier(classifier, 'scripts/classifier_parameters/linear.json')
    train_data, test_data = train_classifier(classifier, data)
    _, _ = evaluate_classifier(classifier, train_data, test_data, plot_and_print_stats=False)
    classifier.save()


def load_model_and_predict_matches(dataset_folder, classifier):
    """
    Load model and unlabeled dataset and predict pairs
    @param dataset_folder: folder containing test data
    @param classifier: classifier type
    @return: pair indices of matches
    """
    classifier = setup_classifier(classifier, 'scripts/classifier_parameters/linear.json')
    classifier = classifier.load()
    data = preprocess_data_without_saving(os.path.join(os.getcwd(), dataset_folder))
    data['predicted_match'], data['predicted_scores'] = classifier.predict(data)
    return data[data['predicted_match'] == 1]


if __name__ == "__main__":
    main()
