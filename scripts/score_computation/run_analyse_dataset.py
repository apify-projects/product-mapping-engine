import json
import os

import click
import pandas as pd

from dataset_handler import preprocess_data, analyse_dataset


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
    analyse_dataset(data)

if __name__ == "__main__":
    main()
