import os
import sys

import click
import pandas as pd

# DO NOT REMOVE
# Adding the higher level directory (scripts/) to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from evaluate_classifier import train_classifier, evaluate_classifier, compute_mean_values, setup_classifier, \
    compute_and_plot_outliers
from score_computation.dataset_handler import preprocess_data


@click.command()
@click.option('--dataset_folder', '-d',
              default='data/wdc_dataset/dataset/preprocessed',
              help='Dataset to use for the evaluation')  #
@click.option('--classifier', '-c',
              default='RandomForests',
              type=click.Choice(
                  ['LinearRegression', 'LogisticRegression', 'SvmLinear', 'SvmRbf', 'SvmPoly', 'NeuralNetwork',
                   'DecisionTree', 'RandomForests']))
@click.option('--classifier_parameters_path', '-p',
              default='scripts/classifier_parameters/linear.json')
@click.option('--runs', '-r', default=1, type=int, help='Number of trains of classifier')
# Load product names and images compute their similarity
def main(**kwargs):
    data = preprocess_data(os.path.join(os.getcwd(), kwargs['dataset_folder']))
    data.to_csv('data.csv', index=False)
    classifier = setup_classifier(kwargs["classifier"], kwargs["classifier_parameters_path"])
    statistics = pd.DataFrame(
        columns=['train_accuracy', 'train_recall', 'train_specificity', 'train_precision', 'test_accuracy',
                 'test_recall', 'test_specificity', 'test_precision'])
    plot_outliers = False
    for i in range(0, kwargs['runs']):
        train_data, test_data = train_classifier(classifier, data)
        train_stats, test_stats = evaluate_classifier(classifier, train_data, test_data, plot_and_print_stats=False)
        if plot_outliers:
            compute_and_plot_outliers(train_data, test_data, classifier.name)
        statistics.loc[i] = list(train_stats.values()) + list(test_stats.values())

    if kwargs['runs'] > 1:
        compute_mean_values(statistics)


if __name__ == "__main__":
    main()
