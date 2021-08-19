import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def train_classifier(classifier, data):
    """
    Train classifier on given dataset
    @param classifier: classifier to train
    @param data: dataset used for training
    @return: train and test datasets with predictions
    """
    train, test = train_test_split(data, test_size=0.25)
    classifier.fit(train)
    train['predicted_match'], train['predicted_scores'] = classifier.predict(train)
    test['predicted_match'], test['predicted_scores'] = classifier.predict(test)
    return train, test


def evaluate_classifier(classifier, classifier_class_name, train_data, test_data):
    """
    Compute accuracy, recall, specificity and precision + plot ROC curve, print feature importance
    @param classifier: classifier to train and evaluate
    @param classifier_class_name: name of the classifier
    @param train_data: training data to evaluate classifier
    @param test_data: testing data to evaluate classifier
    @return: train and test accuracy, recall, specificity, precision
    """
    train_stats = compute_prediction_accuracies(train_data, "train")
    test_stats = compute_prediction_accuracies(test_data, "test")
    threshs = create_thresh(train_data['predicted_scores'], 10)
    out_train = []
    out_test = []
    for t in threshs:
        out_train.append([0 if score < t else 1 for score in train_data['predicted_scores']])
        out_test.append([0 if score < t else 1 for score in test_data['predicted_scores']])
    plot_roc(train_data['match'].tolist(), out_train, test_data['match'].tolist(), out_test, threshs,
             classifier_class_name)
    classifier.print_feature_importances()
    return train_stats, test_stats


def compute_and_plot_outliers(train_data, test_data):
    """
    Compute number of FP and FN and plot their distribution
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
        mismatched.to_csv(f'results/mismatches_{data_type}.csv')


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
    false_negative_count = data[(data['predicted_match'] == 0) & (data['match'] == 1)].shape[0]

    accuracy = (data_count - mismatched_count) / data_count
    recall = true_positive_count / actual_positive_count
    specificity = true_negative_count / actual_negative_count
    precision = true_positive_count / (true_positive_count + false_positive_count)
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

    return {"accuracy": accuracy, "recall": recall, "specificity": specificity, "precision": precision}


def create_thresh(scores, intervals):
    """
    Create dummy threshs from values by sorting them and splitting into k intervals containing the same number of items
    @param scores: data to create threshs
    @param intervals: how many threshs should be created
    @return: created threshs
    """
    scores = np.asarray(sorted(scores))
    subarrays = np.array_split(scores, intervals)
    return [(s[-1]) for s in subarrays][:-1]


def plot_roc(true_train_labels, pred_train_labels_list, true_test_labels, pred_test_labels_list, threshs, classifier):
    """
    Plot roc curve
    @param true_train_labels:  true train labels
    @param pred_test_labels_list: predicted train labels
    @param true_train_labels:  true test labels
    @param pred_test_labels_list: predicted test labels
    @param classifier: classifier name to whose plot should be created
    @param threshs: threshold to evaluate accuracy of similarities
    @return:
    """
    tprs_train, fprs_train = create_roc_curve_points(true_train_labels, pred_train_labels_list, threshs, 'train')
    tprs_test, fprs_test = create_roc_curve_points(true_test_labels, pred_test_labels_list, threshs, 'test')

    plt.plot(fprs_train, tprs_train, marker='.', label='train', color='green')
    plt.plot(fprs_test, tprs_test, marker='.', label='test', color='red')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for {classifier}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(['train', 'test'])
    plt.show()
    plt.clf()


def create_roc_curve_points(true_labels, pred_labels_list, threshs, label):
    """
    Create points for roc curve
    @param true_labels: true labels
    @param pred_labels_list: predicted labels
    @param threshs: threshold to evaluate accuracy of similarities
    @param label: whether working with train or test dataset
    @return: list of true positives, list of false positives
    """
    fprs = []
    tprs = []
    fprs.append(1)
    tprs.append(1)
    print(f'AUC score for different threshs for {label} data')
    print('----------------------------')
    for t, pred_labels in zip(threshs, pred_labels_list):
        # calculate auc score and roc curve
        auc = roc_auc_score(true_labels, pred_labels)
        fpr, tpr, _ = roc_curve(true_labels, pred_labels)
        fprs.append(fpr[1])
        tprs.append(tpr[1])
        print(f'thresh={round(t, 3)} AUC={round(auc, 3)}')
    print('----------------------------')
    print('\n\n')
    fprs.append(0)
    tprs.append(0)
    return tprs, fprs


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
    print(f"Mean Train Accuracy: {round(mean_train_accuracy * 100, 2)}")
    print(f"Mean Train Recall: {round(mean_train_recall * 100, 2)}")
    print(f"Mean Train Specificity: {round(mean_train_specificity * 100, 2)}")
    print(f"Mean Train Precision: {round(mean_train_precision * 100, 2)}")
    print(f"Mean Test Accuracy: {round(mean_test_accuracy * 100, 2)}")
    print(f"Mean Test Recall: {round(mean_test_recall * 100, 2)}")
    print(f"Mean Test Specificity: {round(mean_test_specificity * 100, 2)}")
    print(f"Mean Test Precision: {round(mean_test_precision * 100, 2)}")
    print("----------------------------")
    print("\n\n")
