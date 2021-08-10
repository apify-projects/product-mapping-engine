import json
import os

import click

from evaluate_classifier import evaluate_classifier
from score_computation.dataset_handler import preprocess_data


@click.command()
@click.option('--dataset_folder', '-d',
              default='data/wdc_dataset/dataset/preprocessed',
              help='Dataset to use for the evaluation')  #
@click.option('--classifier', '-c',
              default='Svm',
              type=click.Choice(['Linear', 'Svm']))
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

    evaluate_classifier(classifier, classifier_class_name, data)


if __name__ == "__main__":
    main()
