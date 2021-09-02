import os
import sys

import click
import matplotlib.pyplot as plt

# DO NOT REMOVE
# Adding the higher level directory (scripts/) to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from evaluate_classifier import train_classifier, create_roc_curve_points, create_thresh, setup_classifier
from score_computation.dataset_handler import preprocess_data


@click.command()
@click.option('--dataset_folder', '-d',
              default='data/wdc_dataset/dataset/preprocessed',
              help='Dataset to use for the evaluation')
# Load product names and images compute their similarity with all classifiers and compare them
def main(**kwargs):
    data = preprocess_data(os.path.join(os.getcwd(), kwargs['dataset_folder']))
    data.to_csv('data.csv', index=False)
    roc_data = {}
    for classifier in ['LinearRegression', 'LogisticRegression', 'SvmLinear', 'SvmRbf', 'SvmPoly', 'NeuralNetwork',
                       'DecisionTree', 'RandomForests']:
        classifier = setup_classifier(classifier, 'scripts/classifier_parameters/linear.json')
        train_data, test_data = train_classifier(classifier, data)
        threshs = create_thresh(train_data['predicted_scores'], 10)
        out_train = []
        out_test = []
        for t in threshs:
            out_train.append([0 if score < t else 1 for score in train_data['predicted_scores']])
            out_test.append([0 if score < t else 1 for score in test_data['predicted_scores']])
        tprs_test, fprs_test = create_roc_curve_points(test_data['match'].tolist(), out_test, threshs,
                                                       classifier.name)
        roc_data[classifier.name] = [tprs_test, fprs_test]

    cmap = plt.cm.get_cmap('Set1')
    for i, (classifier, (tprs_test, fprs_test)) in enumerate(roc_data.items()):
        plt.plot(fprs_test, tprs_test, marker='.', label=classifier, color=cmap(i))

    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC test curve for all classifiers')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
