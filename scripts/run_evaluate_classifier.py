import json
import os

import click
import pandas as pd

from evaluate_classifier import evaluate_classifier, compute_mean_values
from score_computation.dataset_handler import preprocess_data


@click.command()
@click.option('--dataset_folder', '-d',
              default='data/wdc_dataset/dataset/preprocessed',
              help='Dataset to use for the evaluation')  #
@click.option('--classifier', '-c',
              default='Svm',
              type=click.Choice(
                  ['LinearRegression', 'LogisticRegression', 'Svm', 'NeuralNetwork', 'DecisionTree', 'RandomForests']))
@click.option('--classifier_parameters_path', '-p',
              default='scripts/classifier_parameters/linear.json')
# Load product names and images compute their similarity
def main(**kwargs):
    data = preprocess_data(os.path.join(os.getcwd(), kwargs['dataset_folder']))
    data.to_csv("data.csv", index=False)

    classifier_class_name = kwargs['classifier'] + "Classifier"
    classifier_class = getattr(__import__('classifiers', fromlist=[classifier_class_name]), classifier_class_name)
    classifier_parameters_path = kwargs["classifier_parameters_path"]
    classifier_parameters_json = '{}'
    with open(classifier_parameters_path, 'r') as classifier_parameters_file:
        classifier_parameters_json = classifier_parameters_file.read()

    classifier_parameters = json.loads(classifier_parameters_json)
    classifier = classifier_class(classifier_parameters)
    runs = 1
    plot_roc_curve = True
    print_stats = True
    statistics = pd.DataFrame(
        columns=['train_accuracy', 'train_recall', 'train_specificity', 'train_precision', 'test_accuracy',
                 'test_recall', 'test_specificity', 'test_precision'])
    for i in range(0, runs):
        train_stats, test_stats = evaluate_classifier(classifier, classifier_class_name, data, plot_roc_curve, print_stats)
        statistics.loc[i] = list(train_stats.values()) + list(test_stats.values())
    if runs>1:
        compute_mean_values(statistics)

if __name__ == "__main__":
    main()
