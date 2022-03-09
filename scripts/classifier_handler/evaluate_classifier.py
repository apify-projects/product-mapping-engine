import json
from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from ..configuration import TEST_DATA_PROPORTION, NUMBER_OF_THRESHES, NUMBER_OF_THRESHES_FOR_AUC, MAX_FP_RATE, \
    PRINT_ROC_AND_STATISTICS


def setup_classifier(classifier_type, classifier_parameters_file=None):
    """
    Setup particular classifier
    @param classifier_type: type of classifier
    @param classifier_parameters_file: file with classifier params
    @return: set up classifier
    """
    classifier_class_name = classifier_type + 'Classifier'

    classifier_class = getattr(__import__('classifier_handler.classifiers', fromlist=[classifier_class_name]), classifier_class_name)
    classifier_parameters = {}
    if classifier_parameters_file is not None:
        classifier_parameters_path = classifier_parameters_file
        with open(classifier_parameters_path, 'r') as classifier_parameters_file:
            classifier_parameters_json = classifier_parameters_file.read()
        classifier_parameters = json.loads(classifier_parameters_json)
    classifier = classifier_class(classifier_parameters)
    return classifier


def train_classifier(classifier, data):
    """
    Train classifier on given dataset
    @param classifier: classifier to train
    @param data: dataset used for training
    @return: train and test datasets with predictions
    """
    train_data, test_data = train_test_split(data, test_size=TEST_DATA_PROPORTION)
    classifier.fit(train_data)
    train_data['predicted_scores'] = classifier.predict(train_data, predict_outputs=False)
    test_data['predicted_scores'] = classifier.predict(test_data, predict_outputs=False)

    train_stats, test_stats = evaluate_classifier(
        classifier,
        train_data,
        test_data,
        True
    )
    classifier.save()
    return train_stats, test_stats


def evaluate_classifier(classifier, train_data, test_data, set_threshold):
    """
    Compute accuracy, recall, specificity and precision + plot ROC curve, print feature importance
    @param classifier: classifier to train and evaluate
    @param train_data: training data to evaluate classifier
    @param test_data: testing data to evaluate classifier
    @param set_threshold: bool whether to calculate and set optimal threshold to the classifier (relevant in Trainer)
    @return: train and test accuracy, recall, specificity, precision
    """
    threshes = create_thresh(train_data['predicted_scores'], NUMBER_OF_THRESHES)
    out_train = []
    out_test = []
    for t in threshes:
        out_train.append([0 if score < t else 1 for score in train_data['predicted_scores']])
        out_test.append([0 if score < t else 1 for score in test_data['predicted_scores']])

    true_positive_rates_train, false_positive_rates_train = create_roc_curve_points(train_data['match'].tolist(),
                                                                                    out_train, threshes, 'train')

    if set_threshold:
        optimal_threshold = threshes[0]
        for x in range(len(threshes)):
            optimal_threshold = threshes[x]
            if false_positive_rates_train[x] <= MAX_FP_RATE:
                break

        classifier.set_threshold(optimal_threshold)
        train_data.drop(columns=['predicted_scores'], inplace=True)
        train_data['predicted_match'], train_data['predicted_scores'] = classifier.predict(train_data)
        test_data.drop(columns=['predicted_scores'], inplace=True)
        test_data['predicted_match'], test_data['predicted_scores'] = classifier.predict(test_data)

    train_stats = compute_prediction_accuracies(train_data, 'train')
    test_stats = compute_prediction_accuracies(test_data, 'test')

    if PRINT_ROC_AND_STATISTICS:
        plot_train_test_roc(
            true_positive_rates_train,
            false_positive_rates_train,
            test_data['match'].tolist(),
            out_test,
            threshes,
            classifier.name
        )
        data = pd.concat([train_data, test_data])
        correlation_matrix = data.drop(['match', 'predicted_match', 'predicted_scores'], axis=1).corr()
        print_whole_correlation_matrix = False
        if print_whole_correlation_matrix:
            sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
            plt.show()
            print(correlation_matrix)
        upper_limit = 0.7
        lower_limit = -0.7
        print(f'Correlations bigger than {upper_limit} ar smaller than {lower_limit}')
        counter = 0
        for column in correlation_matrix:
            for i in range(0, counter):
                row = correlation_matrix[column][i]
                if correlation_matrix.columns[i] != column and (row > upper_limit or row < lower_limit):
                    print(f'{column} + {correlation_matrix.columns[i]}: {row}')
            counter += 1
        print('\n----------------------------\n')
    return train_stats, test_stats


def compute_prediction_accuracies(data, data_type):
    """
    Compute accuracy, precision, recall and show confusion matrix
    @param data: all dataset used for training and prediction
    @param data_type: whether data are train or test
    @return: accuracy, recall, specificity, precision
    """
    data_count = data.shape[0]
    mismatched_count = data[data['predicted_match'] != data['match']].shape[0]
    actual_positive_count = data[data['match'] == 1].shape[0]
    actual_negative_count = data[data['match'] == 0].shape[0]
    true_positive_count = data[(data['predicted_match'] == 1) & (data['match'] == 1)].shape[0]
    false_positive_count = data[(data['predicted_match'] == 1) & (data['match'] == 0)].shape[0]
    true_negative_count = data[(data['predicted_match'] == 0) & (data['match'] == 0)].shape[0]

    accuracy = (data_count - mismatched_count) / data_count if data_count != 0 else 0
    recall = true_positive_count / actual_positive_count if actual_positive_count != 0 else 0
    specificity = true_negative_count / actual_negative_count if actual_negative_count != 0 else 0
    precision = true_positive_count / (
            true_positive_count + false_positive_count) if true_positive_count + false_positive_count != 0 else 0
    conf_matrix = confusion_matrix(data['match'], data['predicted_match'])

    print(f'Classifier results for {data_type} data')
    print('----------------------------')
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Specificity: {specificity}')
    print(f'Precision: {precision}')
    print('Confusion matrix:')
    print(conf_matrix)
    print('----------------------------')
    print('\n\n')

    return {"accuracy": accuracy, "recall": recall, "specificity": specificity, "precision": precision,
            "confusion_matrix": conf_matrix}


def create_thresh(scores, intervals):
    """
    Create dummy threshes from values by sorting them and splitting into k intervals containing the same number of items
    @param scores: data to create threshes
    @param intervals: how many threshes should be created
    @return: created threshes
    """
    scores = np.asarray(sorted(scores))
    sub_arrays = np.array_split(scores, intervals)
    return [(s[-1]) for s in sub_arrays][:-1]


def plot_train_test_roc(
        true_positive_rates_train,
        false_positive_rates_train,
        true_test_labels,
        predicted_test_labels_list,
        threshes,
        classifier
):
    """
    Plot roc curve
    @param true_positive_rates_train: true positive rates for thresholds from the threshes parameter
    @param false_positive_rates_train: false positive rates for thresholds from the threshes parameter
    @param true_test_labels:  true test labels
    @param predicted_test_labels_list: predicted test labels
    @param threshes: threshold to evaluate accuracy of similarities
    @param classifier: classifier name to whose plot should be created
    @return:
    """
    true_positive_rates_test, false_positive_rates_test = create_roc_curve_points(true_test_labels,
                                                                                  predicted_test_labels_list, threshes,
                                                                                  'test')

    plt.plot(false_positive_rates_train, true_positive_rates_train, marker='.', label='train', color='green')
    plt.plot(false_positive_rates_test, true_positive_rates_test, marker='.', label='test', color='red')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for {classifier}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(['train', 'test'])
    plt.show()
    plt.clf()


def create_roc_curve_points(true_labels, predicted_labels_list, threshes, label):
    """
    Create points for roc curve
    @param true_labels: true labels
    @param predicted_labels_list: predicted labels
    @param threshes: threshold to evaluate accuracy of similarities
    @param label: whether working with train or test dataset
    @return: list of true positives, list of false positives
    """
    false_positive_rates = []
    true_positive_rates = []
    false_positive_rates.append(1)
    true_positive_rates.append(1)
    gap_between_auc_scores_outputs = ceil(NUMBER_OF_THRESHES / NUMBER_OF_THRESHES_FOR_AUC)
    threshes_counter = 0
    print(f'AUC score for different threshes for {label} data')
    print('----------------------------')
    for thresh, predicted_labels in zip(threshes, predicted_labels_list):
        # calculate auc score and roc curve
        auc = roc_auc_score(true_labels, predicted_labels)
        false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, predicted_labels)
        false_positive_rates.append(false_positive_rate[1])
        true_positive_rates.append(true_positive_rate[1])
        if threshes_counter % gap_between_auc_scores_outputs == 0:
            print(f'thresh={round(thresh, 3)} AUC={round(auc, 3)}')
        threshes_counter += 1
    print('----------------------------')
    print('\n\n')
    false_positive_rates.append(0)
    true_positive_rates.append(0)
    return true_positive_rates, false_positive_rates


# OTHER UNUSED METHODS
def compute_mean_values(statistics):
    """
    Compute mean values of accuracy, recall, specificity and precision after several runs
    @param statistics: accuracy, recall, specificity and precision from all the runs
    @return:
    """
    mean_train_accuracy = statistics['train_accuracy'].mean()
    mean_train_recall = statistics['train_recall'].mean()
    mean_train_specificity = statistics['train_specificity'].mean()
    mean_train_precision = statistics['train_precision'].mean()
    mean_test_accuracy = statistics['test_accuracy'].mean()
    mean_test_recall = statistics['test_recall'].mean()
    mean_test_specificity = statistics['test_specificity'].mean()
    mean_test_precision = statistics['test_precision'].mean()
    print(f'Evaluation results after {statistics.shape[0]} runs:')
    print("----------------------------")
    print(f"Mean Train Accuracy: \t{round(mean_train_accuracy * 100, 2)}")
    print(f"Mean Train Recall: \t{round(mean_train_recall * 100, 2)}")
    print(f"Mean Train Specificity: \t{round(mean_train_specificity * 100, 2)}")
    print(f"Mean Train Precision: \t{round(mean_train_precision * 100, 2)}")
    print(f"Mean Test Accuracy: \t{round(mean_test_accuracy * 100, 2)}")
    print(f"Mean Test Recall: \t{round(mean_test_recall * 100, 2)}")
    print(f"Mean Test Specificity: \t{round(mean_test_specificity * 100, 2)}")
    print(f"Mean Test Precision: \t{round(mean_test_precision * 100, 2)}")
    print("----------------------------")
    print("\n\n")


def compute_and_plot_outliers(train_data, test_data, classifier_class_name):
    """
    Compute number of FP and FN and plot their distribution
    @param classifier_class_name: name of the classifier
    @param train_data: train data
    @param test_data: test data
    @return:
    """
    for data, data_type in zip([train_data, test_data], ['train_data', 'test_data']):
        print(data_type)
        mismatched_count = data[data['predicted_match'] != data['match']].shape[0]
        tp_data = data[(data['predicted_match'] == 1) & (data['match'] == 1)]
        fp_data = data[(data['predicted_match'] == 1) & (data['match'] == 0)]
        tn_data = data[(data['predicted_match'] == 0) & (data['match'] == 0)]
        fn_data = data[(data['predicted_match'] == 0) & (data['match'] == 1)]

        print(f'Number of mismatching values for {data} is: {mismatched_count}')
        print('----------------------------')
        print(f'Number of FPs is: {len(fp_data)}')
        print(f'Number of FNs is: {len(fn_data)}')
        print('----------------------------')
        print('\n\n')

        visualize_outliers(tp_data, fp_data, ['TP', 'FP', data_type])
        visualize_outliers(tn_data, fn_data, ['TN', 'FN', data_type])

        mismatched = data[data['predicted_match'] != data['match']]
        mismatched.to_csv(f'results/mismatches/data/{classifier_class_name}_{data_type}.csv')


def visualize_outliers(correct_data, wrong_data, labels):
    """
    Plot outliers according to the feature values
    @param correct_data: correctly predicted data
    @param wrong_data: wrongly predicted data
    @param labels: labels whether iit is TP+FP or TN+FN
    @return:
    """
    for column in correct_data.iloc[:, :-3]:
        plt.scatter(np.arange(0, len(correct_data)), correct_data[column], color='green')
        plt.scatter(np.arange(0, len(wrong_data)), wrong_data[column], color='red')
        plt.title(f'{labels[0]} and {labels[1]} distribution on {labels[2]} by {column} match')
        plt.xlabel(f'{labels[2]}')
        plt.ylabel(f'{column} data match value')
        plt.legend(labels)
        plt.show()
        plt.clf()
