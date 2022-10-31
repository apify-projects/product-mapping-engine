import itertools
import random
import warnings
from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from ..configuration import TEST_DATA_PROPORTION, NUMBER_OF_THRESHES, NUMBER_OF_THRESHES_FOR_AUC, \
    PRINT_ROC_AND_STATISTICS, PERFORMED_PARAMETERS_SEARCH, RANDOM_SEARCH_ITERATIONS, \
    NUMBER_OF_TRAINING_REPETITIONS_TO_AVERAGE_RESULTS, MINIMAL_PRECISION, MINIMAL_RECALL, \
    BEST_MODEL_SELECTION_CRITERION, PRINT_CORRELATION_MATRIX, CORRELATION_LIMIT, PERFORM_TRAIN_TEST_SPLIT, \
    SAVE_TRAIN_TEST_SPLIT, SAMPLE_VALIDATION_DATA_FROM_TRAIN_DATA, VALIDATION_DATA_PROPORTION


def setup_classifier(classifier_type):
    """
    Setup particular classifier or more classifiers if grid search is performed
    @param classifier_type: type of classifier
    @return: set up classifier(s) and its/theirs parameters
    """
    classifier_class_name = classifier_type + 'Classifier'
    classifier_class = getattr(__import__('classifier_handler.classifiers', fromlist=[classifier_class_name]),
                               classifier_class_name)
    if PERFORMED_PARAMETERS_SEARCH == 'grid' or PERFORMED_PARAMETERS_SEARCH == 'random':
        classifier_parameters_name = classifier_type + '_CLASSIFIER_PARAMETERS_SEARCH'
    else:
        classifier_parameters_name = classifier_type + '_CLASSIFIER_PARAMETERS'
    classifier_parameters = getattr(__import__('configuration', fromlist=[classifier_parameters_name]),
                                    classifier_parameters_name)

    if PERFORMED_PARAMETERS_SEARCH == 'grid' and classifier_parameters != {}:
        classifier = []
        for parameter_name in classifier_parameters:
            if not isinstance(classifier_parameters[parameter_name], list):
                classifier_parameters[parameter_name] = [classifier_parameters[parameter_name]]
        keys, values = zip(*classifier_parameters.items())
        classifier_parameters_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for classifier_parameters in classifier_parameters_permutations:
            if is_valid_combination_of_parameters(classifier_type, classifier_parameters):
                classifier.append(classifier_class({}, classifier_parameters))
    elif PERFORMED_PARAMETERS_SEARCH == 'random' and classifier_parameters != {}:
        classifier = []
        for _ in range(RANDOM_SEARCH_ITERATIONS):
            classifier_parameters_random = {}
            for parameter_name in classifier_parameters:
                if isinstance(classifier_parameters[parameter_name], list):
                    if all(isinstance(x, (int, float)) for x in classifier_parameters[parameter_name]):
                        min_val = min(classifier_parameters[parameter_name])
                        max_val = max(classifier_parameters[parameter_name])
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            classifier_parameters_random[parameter_name] = random.randint(min_val, max_val)
                        else:
                            classifier_parameters_random[parameter_name] = random.uniform(min_val, max_val)
                    else:
                        classifier_parameters_random[parameter_name] = random.choice(classifier_parameters[parameter_name])
                else:
                    classifier_parameters_random[parameter_name] = classifier_parameters[parameter_name]
            if is_valid_combination_of_parameters(classifier_type, classifier_parameters_random):
                classifier.append(classifier_class({}, classifier_parameters_random))
    else:
        classifier = classifier_class({}, classifier_parameters)
    return classifier, classifier_parameters


def is_valid_combination_of_parameters(classifier_type, parameters):
    """
    Check whether the created combination of parameters is valid
    @param classifier_type: type of classifier
    @param parameters: parameters of the classifier
    @return:
    """
    if classifier_type == 'LogisticRegression':
        if parameters['solver'] in ['lbfgs', 'newton-cg', 'sag'] and parameters['penalty'] == 'l1':
            return False
        if parameters['solver'] in ['lbfgs', 'newton-cg'] and parameters['penalty'] == 'elasticnet':
            return False
        if parameters['penalty'] == 'elasticnet' and parameters['penalty'] != 'saga':
            return False
        if parameters['penalty'] == 'none' and parameters['solver'] == 'liblinear':
            return False
    return True


def train_classifier(classifier, data, task_id, train_data=None, test_data=None):
    """
    Train classifier on given dataset
    @param classifier: classifier to train
    @param data: dataset used for training
    @param task_id: id of the task
    @param train_data: train data if passed
    @param test_data: test data if passed
    @return: train and test datasets with predictions and statistics
    """
    if train_data is None and test_data is None:
        train_data, test_data = create_train_test_data(data, task_id)

    classifier.fit(train_data)
    train_data['predicted_scores'] = classifier.predict(train_data, predict_outputs=False)
    test_data['predicted_scores'] = classifier.predict(test_data, predict_outputs=False)

    train_stats, test_stats = evaluate_classifier(
        classifier,
        train_data,
        test_data,
        True,
        'trainer data'
    )
    classifier.save()
    return train_stats, test_stats


def create_train_test_data(data, task_id):
    """
    Create or load train and test data from all data
    @param data: dataframe to create train and test data
    @param task_id: id of the task
    @return: dataframes with train and test data
    """
    if PERFORM_TRAIN_TEST_SPLIT:
        if SAVE_TRAIN_TEST_SPLIT:
            train_data, test_data = train_test_split(data, test_size=TEST_DATA_PROPORTION)
            train_data.to_csv(task_id + '-train_data.csv', index=False)
            test_data.to_csv(task_id + '-test_data.csv', index=False)

            train_data = train_data.drop(columns=['id1', 'id2'])
            test_data = test_data.drop(columns=['id1', 'id2'])
        else:
            train_data, test_data = train_test_split(data.drop(columns=['id1', 'id2']), test_size=TEST_DATA_PROPORTION)
    else:
        train_data = pd.read_csv(task_id + '-train_data.csv').drop(columns=['id1', 'id2'])
        test_data = pd.read_csv(task_id + '-test_data.csv').drop(columns=['id1', 'id2'])
    return train_data, test_data


def sample_data_according_to_weights(data, weights, sample_proportion):
    sample = data.sample(weights=weights, frac=sample_proportion, replace=True)
    return sample


def ensemble_models_training(similarities, classifier_type, task_id):
    """
    Training ensembles of several models
    @param similarities: dataframe with precomputed similarities
    @param classifier_type: type of the ensemble
    @param task_id: id of the task
    @return: combined classifier and its train and test stats
    """

    classifiers, _ = setup_classifier(classifier_type)
    data = similarities.drop(columns=['id1', 'id2'])
    if PERFORM_TRAIN_TEST_SPLIT:
        train_data, test_data = train_test_split(data.drop(columns=['id1', 'id2']), test_size=TEST_DATA_PROPORTION)
    else:
        train_data = pd.read_csv(task_id + '-train_data.csv').drop(columns=['id1', 'id2'])
        test_data = pd.read_csv(task_id + '-test_data.csv').drop(columns=['id1', 'id2'])
    predicted_scores_train = []
    predicted_scores_test = []

    weights = [1] * train_data.shape[0]
    for classifier in classifiers.model:
        train_data_sample = sample_data_according_to_weights(train_data, weights, 0.2)
        classifier.fit(train_data_sample)
        predicted_scores_train.append(classifier.predict(train_data, predict_outputs=False))
        predicted_scores_test.append(classifier.predict(test_data, predict_outputs=False))
        if classifier_type == 'Boosting':
            evaluation_data = train_data[['match']]
            evaluation_data['predicted_scores'] = classifiers.combine_predictions_from_classifiers(
                predicted_scores_train,
                'score'
            )
            weights = evaluation_data.apply(
                lambda row: min(1 / (row.predicted_scores if row.match == 1 else 1 - row.predicted_scores), 10),
                axis=1
            )

    train_data['predicted_scores'] = classifiers.combine_predictions_from_classifiers(predicted_scores_train, 'score')
    test_data[f'predicted_scores'] = classifiers.combine_predictions_from_classifiers(predicted_scores_test, 'score')
    train_stats, test_stats = evaluate_classifier(
        classifiers,
        train_data,
        test_data,
        True,
        'trainer data'
    )
    classifiers.save()
    return classifiers, train_stats, test_stats


def parameters_search_and_best_model_training(similarities, classifier_type, task_id):
    """
    Setup classifier and perform grid or random search to find the best parameters for given type of model
    @param similarities: precomputed similarities used as training data
    @param classifier_type: classifier type
    @param task_id: id of the task
    @return: classifier with the highest test f1 score and its train and test stats
    """
    classifiers, classifier_parameters = setup_classifier(classifier_type)
    best_classifier = None
    best_classifier_f1_score = -1
    similarities_without_ids = similarities.drop(columns=['id1', 'id2'])
    if not isinstance(classifiers, list) or len(classifiers) == 1:
        if isinstance(classifiers, list) and len(classifiers) == 1:
            classifiers = classifiers[0]

        train_stats, test_stats = train_classifier(classifiers, similarities_without_ids, task_id)
        warnings.warn(
            f'Warning: {PERFORMED_PARAMETERS_SEARCH} search not performed as there is only one model to train')

        return classifiers, train_stats, test_stats
    rows_to_dataframe = []
    if SAMPLE_VALIDATION_DATA_FROM_TRAIN_DATA:
        train_data, test_data = create_train_test_data(similarities, task_id)
        validation_data = train_data.sample(frac=VALIDATION_DATA_PROPORTION)
        validation_data.to_csv('validation_data_params_search.csv', index=False)
        train_data_params_search = train_data.drop(validation_data.index)
        train_data_params_search.to_csv('train_data_params_search.csv', index=False)
    sub_train_data = pd.read_csv('train_data_params_search.csv')
    validation_data = pd.read_csv('validation_data_params_search.csv')
    for classifier in classifiers:
        if 'predicted_scores' in validation_data.columns:
            validation_data = validation_data.drop('predicted_scores', axis=1)
        if 'predicted_match' in validation_data.columns:
            validation_data = validation_data.drop('predicted_match', axis=1)
        if 'predicted_scores' in sub_train_data.columns:
            sub_train_data = sub_train_data.drop('predicted_scores', axis=1)
        if 'predicted_match' in sub_train_data.columns:
            sub_train_data = sub_train_data.drop('predicted_match', axis=1)
        if NUMBER_OF_TRAINING_REPETITIONS_TO_AVERAGE_RESULTS == 1:
            train_stats, test_stats = train_classifier(classifier, None, task_id, sub_train_data, validation_data)
        else:
            train_stats_array = []
            test_stats_array = []
            for i in range(NUMBER_OF_TRAINING_REPETITIONS_TO_AVERAGE_RESULTS):
                train_stats, test_stats = train_classifier(classifier, similarities_without_ids, task_id)
                train_stats_array.append(train_stats)
                test_stats_array.append(test_stats)
            train_stats = average_statistics_from_several_runs(train_stats_array)
            test_stats = average_statistics_from_several_runs(test_stats_array)
        row_to_dataframe = []
        for parameter in classifier_parameters:
            row_to_dataframe.append(getattr(classifier.model, parameter))
        row_to_dataframe = row_to_dataframe + [train_stats['f1_score'], train_stats['accuracy'],
                                               train_stats['precision'], train_stats['recall'],
                                               test_stats['f1_score'], test_stats['accuracy'],
                                               test_stats['precision'], test_stats['recall']]
        rows_to_dataframe.append(row_to_dataframe)
        if test_stats['f1_score'] > best_classifier_f1_score:
            best_classifier = classifier
            best_classifier_f1_score = test_stats['f1_score']

    best_train_stats, best_test_stats = train_classifier(best_classifier, similarities_without_ids, task_id)
    print(f'{PERFORMED_PARAMETERS_SEARCH.upper()} SEARCH PERFORMED')
    print_best_classifier_results(best_train_stats, best_test_stats)
    print(f'Best classifier parameters')
    for parameter in classifier_parameters:
        print(f'{parameter}: {getattr(best_classifier.model, parameter)}')
    print('----------------------------')
    models_results = pd.DataFrame(rows_to_dataframe,
                                  columns=list(classifier_parameters.keys()) + ['train_f1_score', 'train_accuracy',
                                                                                'train_precision', 'train_recall',
                                                                                'test_f1_score', 'test_accuracy',
                                                                                'test_precision', 'test_recall'])
    models_results = models_results.sort_values(by=['test_f1_score'], ascending=False)
    models_results.to_csv(f'results/{classifier_type}_models_comparison.csv')
    return best_classifier, best_train_stats, best_test_stats


def average_statistics_from_several_runs(statistics_from_runs):
    """
    Average statistical values from several runs
    @param statistics_from_runs: List with dicts of statistical values from several runs
    @return: Dict with average values
    """
    statistics_average = {}
    for stats in statistics_from_runs:
        for key, value in stats.items():
            statistics_average.setdefault(key, []).append(value)
    for key in statistics_average:
        statistics_average[key] = sum(statistics_average[key]) / len(statistics_average[key])
    return statistics_average


def evaluate_classifier(classifier, train_data, test_data, set_threshold, data_type):
    """
    Find the best thresh and compute f1 score, accuracy, recall, specificity and precision + plot ROC curve
    @param classifier: classifier to train and evaluate
    @param train_data: training data to evaluate classifier
    @param test_data: testing data to evaluate classifier
    @param set_threshold: bool whether to calculate and set optimal threshold to the classifier (relevant in Trainer)
    @param data_type: string value specifying the evaluated data type
    @return: train and test  f1 score, accuracy, recall, specificity, precision
    """
    # create threshes
    threshes = create_thresh(train_data['predicted_scores'], NUMBER_OF_THRESHES)
    out_train = []
    out_test = []
    for t in threshes:
        out_train.append([0 if score < t else 1 for score in train_data['predicted_scores']])
        out_test.append([0 if score < t else 1 for score in test_data['predicted_scores']])

    true_positive_rates_train, false_positive_rates_train = create_roc_curve_points(train_data['match'].tolist(),
                                                                                    out_train, threshes, 'train')
    if 'predicted_scores' in train_data.columns:
        train_data.drop(columns=['predicted_scores'], inplace=True)
    if 'predicted_scores' in test_data.columns:
        test_data.drop(columns=['predicted_scores'], inplace=True)
    # find the best thresh
    if set_threshold:
        optimal_threshold = threshes[0]
        optimal_value = -1 if BEST_MODEL_SELECTION_CRITERION != 'balanced_precision_recall' else 1000
        optimal_minimal_value = -1
        for x, thresh in enumerate(threshes):

            # prepare and test classifier
            classifier.set_threshold(thresh)
            test_data['predicted_match'], test_data['predicted_scores'] = classifier.predict(
                test_data)
            test_data_results = compute_prediction_accuracies(test_data, 'test', print_classifier_results=False)

            # compare results and select the best thresh
            if BEST_MODEL_SELECTION_CRITERION == 'max_f1':
                if test_data_results['f1_score'] > optimal_value:
                    optimal_threshold = thresh
                    optimal_value = test_data_results['f1_score']
            elif BEST_MODEL_SELECTION_CRITERION == 'max_precision':
                if is_model_better_than_previous(test_data_results, 'precision', optimal_value, 'recall',
                                                 MINIMAL_RECALL, optimal_minimal_value):
                    optimal_threshold = thresh
                    optimal_value = test_data_results['precision']
                    optimal_minimal_value = test_data_results['recall']
            elif BEST_MODEL_SELECTION_CRITERION == 'max_recall':
                if is_model_better_than_previous(test_data_results, 'recall', optimal_value, 'precision',
                                                 MINIMAL_PRECISION, optimal_minimal_value):
                    optimal_threshold = thresh
                    optimal_value = test_data_results['recall']
                    optimal_minimal_value = test_data_results['precision']
            elif BEST_MODEL_SELECTION_CRITERION == 'balanced_precision_recall':
                if abs(test_data_results['precision'] - test_data_results['recall']) < optimal_value:
                    optimal_threshold = thresh
                    optimal_value = abs(test_data_results['precision'] - test_data_results['recall'])
            else:
                raise SystemExit('Invalid value of BEST_MODEL_SELECTION_CRITERION parameter.')

            # clean the data
            test_data.drop(columns=['predicted_match'], inplace=True)
            test_data.drop(columns=['predicted_scores'], inplace=True)

        # evaluate data
        classifier.set_threshold(optimal_threshold)
        train_data['predicted_match'], train_data['predicted_scores'] = classifier.predict(train_data)
        test_data['predicted_match'], test_data['predicted_scores'] = classifier.predict(test_data)

    # compute prediction accuracies
    train_stats = compute_prediction_accuracies(train_data, 'train')
    test_stats = compute_prediction_accuracies(test_data, 'test')

    if PRINT_ROC_AND_STATISTICS:
        plot_train_test_roc(true_positive_rates_train, false_positive_rates_train, test_data['match'].tolist(),
                            out_test, threshes, classifier.name, data_type)
    if PRINT_CORRELATION_MATRIX:
        plot_correlation_matrix(test_data, train_data)

    return train_stats, test_stats


def plot_correlation_matrix(test_data, train_data):
    """
    Print correlation matrix and the most correlated values
    @param test_data: dataframe with train data
    @param train_data: dataframe with test data
    @return:
    """
    data = pd.concat([train_data, test_data])
    correlation_matrix = data.drop(['match', 'predicted_match', 'predicted_scores'], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()
    print(correlation_matrix)
    print(f'Correlations bigger than {CORRELATION_LIMIT} ar smaller than {-CORRELATION_LIMIT}')
    counter = 0
    for column in correlation_matrix:
        for i in range(0, counter):
            row = correlation_matrix[column][i]
            if correlation_matrix.columns[i] != column and (row > CORRELATION_LIMIT or row < -CORRELATION_LIMIT):
                print(f'{column} + {correlation_matrix.columns[i]}: {row}')
        counter += 1
    print('\n----------------------------\n')


def compute_prediction_accuracies(data, data_type, print_classifier_results=True):
    """
    Compute  f1 score, accuracy, precision, recall and show confusion matrix
    @param data: all dataset used for training and prediction
    @param data_type: whether data are train or test
    @return:  f1 score, accuracy, recall, specificity, precision
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
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    if print_classifier_results:
        print(f'Classifier results for {data_type} data')
        print('----------------------------')
        print(f'F1 score: {f1_score}')
        print(f'Accuracy: {accuracy}')
        print(f'Recall: {recall}')
        print(f'Specificity: {specificity}')
        print(f'Precision: {precision}')
        print('Confusion matrix:')
        print(conf_matrix)
        print('----------------------------')
        print('\n\n')

    return {'f1_score': f1_score, 'accuracy': accuracy, 'recall': recall, 'specificity': specificity,
            'precision': precision, 'confusion_matrix': conf_matrix}


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
        classifier,
        data_type
):
    """
    Plot roc curve
    @param true_positive_rates_train: true positive rates for thresholds from the threshes' parameter
    @param false_positive_rates_train: false positive rates for thresholds from the threshes' parameter
    @param true_test_labels:  true test labels
    @param predicted_test_labels_list: predicted test labels
    @param threshes: threshold to evaluate accuracy of similarities
    @param classifier: classifier name to whose plot should be created
    @param data_type: string value specifying the evaluated data type
    @return:
    """
    true_positive_rates_test, false_positive_rates_test = create_roc_curve_points(true_test_labels,
                                                                                  predicted_test_labels_list, threshes,
                                                                                  'test')

    plt.plot(false_positive_rates_train, true_positive_rates_train, marker='.', label='train', color='green')
    plt.plot(false_positive_rates_test, true_positive_rates_test, marker='.', label='test', color='red')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for {classifier} for {data_type}')
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
        try:
            auc = roc_auc_score(true_labels, predicted_labels)
            if threshes_counter % gap_between_auc_scores_outputs == 0:
                print(f'thresh={round(thresh, 3)} AUC={round(auc, 3)}')
        except ValueError as e:
            warnings.warn(str(e))
        false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, predicted_labels)
        false_positive_rates.append(false_positive_rate[1])
        true_positive_rates.append(true_positive_rate[1])
        threshes_counter += 1
    print('----------------------------')
    print('\n\n')
    false_positive_rates.append(0)
    true_positive_rates.append(0)
    return true_positive_rates, false_positive_rates


def select_best_classifier(classifiers):
    """
    Select best classifier from several training runs according to given criterion
    @param classifiers: list of classifiers
    @return: the best classifier and its train and test stats
    """
    best_classifier = None
    best_train_stats = None
    best_test_stats = None
    best_compared_value = -1
    best_minimal_value = -1

    for classifier in classifiers:
        if BEST_MODEL_SELECTION_CRITERION == 'max_f1':
            if classifier['test_stats']['f1_score'] > best_compared_value:
                best_classifier = classifier['classifier']
                best_train_stats = classifier['train_stats']
                best_test_stats = classifier['test_stats']
                best_compared_value = classifier['test_stats']['f1_score']
        elif BEST_MODEL_SELECTION_CRITERION == 'max_precision':
            if is_model_better_than_previous(classifier['test_stats'], 'precision', best_compared_value, 'recall',
                                             MINIMAL_RECALL, best_minimal_value):
                best_classifier = classifier['classifier']
                best_train_stats = classifier['train_stats']
                best_test_stats = classifier['test_stats']
                best_compared_value = classifier['test_stats']['precision']
                best_minimal_value = classifier['test_stats']['recall']
        elif BEST_MODEL_SELECTION_CRITERION == 'max_recall':
            if is_model_better_than_previous(classifier['test_stats'], 'recall', best_compared_value, 'precision',
                                             MINIMAL_PRECISION, best_minimal_value):
                best_classifier = classifier['classifier']
                best_train_stats = classifier['train_stats']
                best_test_stats = classifier['test_stats']
                best_compared_value = classifier['test_stats']['recall']
                best_minimal_value = classifier['test_stats']['precision']
        elif BEST_MODEL_SELECTION_CRITERION == 'balanced_precision_recall':
            if abs(classifier['test_stats']['precision'] - classifier['test_stats']['recall']) < best_compared_value:
                best_classifier = classifier['classifier']
                best_train_stats = classifier['train_stats']
                best_test_stats = classifier['test_stats']
                best_compared_value = abs(classifier['test_stats']['precision'] - classifier['test_stats']['recall'])
        else:
            raise SystemExit('Invalid value of BEST_MODEL_SELECTION_CRITERION parameter.')

    if best_classifier is None:
        raise SystemExit('No classifier satisfying requested parameters for best model selection process was found.')
    print_best_classifier_results(best_train_stats, best_test_stats)
    return best_classifier, best_train_stats, best_test_stats


def is_model_better_than_previous(
        model_results,
        compared_parameter_type,
        optimal_value,
        minimal_parameter_type,
        minimal_parameter_value,
        optimal_parameter_minimum
):
    """
    Check whether the model has better results than previous models
    @param model_results: dataframe with testing data
    @param compared_parameter_type: precision or recall, depends on what is compared
    @param optimal_value: value of optimal solution
    @param minimal_parameter_type: recall or precision, tho opposite of compared_parameter_type
    @param minimal_parameter_value: minimal tolerated value of minimal_parameter_type (precision or recall)
                                    defined in config
    @param optimal_parameter_minimum: value of minimal_parameter_type of optimal model
    @return: True if the model with given thresh is better than previous threshes
    """
    return (model_results[compared_parameter_type] > optimal_value and
            model_results[minimal_parameter_type] > minimal_parameter_value) \
           or \
           (model_results[compared_parameter_type] == optimal_value and
            model_results[minimal_parameter_type] > optimal_parameter_minimum)


def print_best_classifier_results(best_train_stats, best_test_stats):
    """
    Print best classifier results
    @param best_train_stats: the best classifier training statistics
    @param best_test_stats: the best classifier training statistics
    @return:
    """
    print('BEST CLASSIFIER')
    print(f'Best classifier train F1 score: {best_train_stats["f1_score"]}')
    print(f'Best classifier train accuracy: {best_train_stats["accuracy"]}')
    print(f'Best classifier train precision: {best_train_stats["precision"]}')
    print(f'Best classifier train recall: {best_train_stats["recall"]}')
    print('Confusion matrix:')
    print(best_train_stats["confusion_matrix"])

    print(f'Best classifier test F1 score: {best_test_stats["f1_score"]}')
    print(f'Best classifier test accuracy: {best_test_stats["accuracy"]}')
    print(f'Best classifier test precision: {best_test_stats["precision"]}')
    print(f'Best classifier test recall: {best_test_stats["recall"]}')
    print('Confusion matrix:')
    print(best_test_stats["confusion_matrix"])
    print('----------------------------')
    print('\n\n')
    pass
