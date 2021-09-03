import os
import sys

# DO NOT REMOVE
# Adding the higher level directory (scripts/) to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from scripts.evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier
from scripts.score_computation.dataset_handler import preprocess_data


# Load dataset and train and save model
def main(**kwargs):
    load_data_and_train_classifier('data/wdc_dataset/dataset/preprocessed', 'DecisionTree')
    matching_pairs = load_and_evaluate_model('data/wdc_dataset/dataset/preprocessed', 'DecisionTree')
    print(matching_pairs)


def find_matching_products(product, dataset):
    pass


def load_data_and_train_classifier(dataset_folder, classifier):
    """
    Load dataset and train and save model
    @param dataset_folder: folder containing data
    @param classifier: classifier type
    @return:
    """
    data = preprocess_data(os.path.join(os.getcwd(), dataset_folder))
    data.to_csv('data.csv', index=False)
    classifier = setup_classifier(classifier, 'scripts/classifier_parameters/linear.json')
    train_data, test_data = train_classifier(classifier, data)
    _, _ = evaluate_classifier(classifier, train_data, test_data, plot_and_print_stats=False)
    classifier.save()


def load_and_evaluate_model(dataset_folder, classifier):
    """
    Load model and unlabeled dataset and predict pairs
    @param dataset_folder: folder containing test data
    @param classifier: classifier type
    @return: pair indices of matches
    """
    classifier = setup_classifier(classifier, 'scripts/classifier_parameters/linear.json')
    classifier = classifier.load()
    data = preprocess_data(os.path.join(os.getcwd(), dataset_folder))
    data['predicted_match'], data['predicted_scores'] = classifier.predict(data)
    return data[data['predicted_match'] == 1]


if __name__ == "__main__":
    main()
