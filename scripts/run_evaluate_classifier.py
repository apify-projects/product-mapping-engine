import json
import os

import click
import pandas as pd

from evaluate_classifier import train_classifier, evaluate_classifier, compute_mean_values, compute_and_plot_outliers
from score_computation.dataset_handler import preprocess_data


@click.command()
@click.option('--dataset_folder', '-d',
              default='data/wdc_dataset/dataset/preprocessed',
              help='Dataset to use for the evaluation')  #
@click.option('--classifier', '-c',
              default='RandomForests',
              type=click.Choice(
                  ['LinearRegression', 'LogisticRegression', 'Svm', 'NeuralNetwork', 'DecisionTree', 'RandomForests']))
@click.option('--classifier_parameters_path', '-p',
              default='scripts/classifier_parameters/linear.json')
@click.option('--runs', '-r', default=1, type=int, help='Number of trains of classifier')
# Load product names and images compute their similarity
def main(**kwargs):
    data = preprocess_data(os.path.join(os.getcwd(), kwargs['dataset_folder']))
    data.to_csv('data.csv', index=False)

    classifier_class_name = kwargs['classifier'] + 'Classifier'
    classifier_class = getattr(__import__('classifiers', fromlist=[classifier_class_name]), classifier_class_name)
    classifier_parameters_path = kwargs["classifier_parameters_path"]
    classifier_parameters_json = '{}'
    with open(classifier_parameters_path, 'r') as classifier_parameters_file:
        classifier_parameters_json = classifier_parameters_file.read()

    classifier_parameters = json.loads(classifier_parameters_json)
    classifier = classifier_class(classifier_parameters)
    statistics = pd.DataFrame(
        columns=['train_accuracy', 'train_recall', 'train_specificity', 'train_precision', 'test_accuracy',
                 'test_recall', 'test_specificity', 'test_precision'])
    for i in range(0, kwargs['runs']):
        train_data, test_data = train_classifier(classifier, data)
        train_stats, test_stats = evaluate_classifier(classifier, classifier_class_name, train_data, test_data)
        compute_and_plot_outliers(train_data, test_data, classifier_class_name)
        statistics.loc[i] = list(train_stats.values()) + list(test_stats.values())

    if kwargs['runs'] > 1:
        compute_mean_values(statistics)


if __name__ == "__main__":
    main()
