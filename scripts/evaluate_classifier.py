import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def evaluate_classifier(classifier, classifier_class_name, data, plot_roc_curve=False, print_stats=False):
    """
    Evaluate classifier - let it train and predict weights for images and hashes and plot ROC curve
    @param classifier: classifier to train and evaluate
    @param classifier_class_name: name of the classifier
    @param data: data to train and evaluate classifier
    @param plot_roc_curve: whether to plot ROC curve of single run
    @param print_stats: whether to print accuracy, recall, specificity, precision and confusion matrix of single run
    @return: train and test accuracy, recall, specificity, precision
    """
    train, test = train_test_split(data, test_size=0.25)
    classifier.fit(train)
    out_train, scores_train = classifier.predict(train)
    out_test, scores_test = classifier.predict(test)
    train_stats = evaluate_predictions(train, out_train, "train", print_stats)
    test_stats = evaluate_predictions(test, out_test, "test", print_stats)
    threshs = create_thresh(scores_train, 10)
    out_train = []
    out_test = []
    for t in threshs:
        out_train.append([0 if score < t else 1 for score in scores_train])
        out_test.append([0 if score < t else 1 for score in scores_test])
    if plot_roc_curve:
        plot_roc(train['match'].tolist(), out_train, test['match'].tolist(), out_test, threshs, classifier_class_name,
                 print_stats=False)
        classifier.print_feature_importances()
    explore_outliers(train, out_train, test, out_test)
    return train_stats, test_stats

def explore_outliers(train_data, train_data_predictions, test_data, test_data_predictions):
    pass

def evaluate_predictions(data, outputs, data_type, print_stats):
    """
    Compute accuracy, precision, recall and show confusion matrix
    @param data: all dataset used for training and prediction
    @param outputs: predicted outputs
    @param data_type: whether data are train or test
    @param print_stats: whether to print accuracy, recall, specificity, precision and confusion matrix of single run
    @return: accuracy, recall, specificity, precision
    """
    data['match_prediction'] = outputs
    data_count = data.shape[0]
    mismatched = data[data['match_prediction'] != data['match']]
    mismatched_count = data[data['match_prediction'] != data['match']].shape[0]
    actual_positive_count = data[data['match'] == 1].shape[0]
    actual_negative_count = data[data['match'] == 0].shape[0]
    true_positive_count = data[(data['match_prediction'] == 1) & (data['match'] == 1)].shape[0]
    false_positive_count = data[(data['match_prediction'] == 1) & (data['match'] == 0)].shape[0]
    true_negative_count = data[(data['match_prediction'] == 0) & (data['match'] == 0)].shape[0]
    false_negative_count = data[(data['match_prediction'] == 0) & (data['match'] == 1)].shape[0]

    accuracy = (data_count - mismatched_count) / data_count
    recall = true_positive_count / actual_positive_count
    specificity = true_negative_count / actual_negative_count
    precision = true_positive_count / (true_positive_count + false_positive_count)
    conf_matrix = confusion_matrix(data['match'], data['match_prediction'])
    if print_stats:
        print(f'\n\nClassifier results for {data_type} data')
        print('----------------------------')
        print(f'Accuracy: {accuracy}')
        print(f'Recall: {recall}')
        print(f'Specificity: {specificity}')
        print(f'Precision: {precision}')
        print('Confusion matrix:')
        print(conf_matrix)
        print('----------------------------')
        print('\n\n')

    mismatched.to_csv("mismatches.csv")
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


def plot_roc(true_train_labels, pred_train_labels_list, true_test_labels, pred_test_labels_list, threshs, classifier,
             print_stats=False):
    """
    Plot roc curve
    @param true_train_labels:  true train labels
    @param pred_test_labels_list: predicted train labels
    @param true_train_labels:  true test labels
    @param pred_test_labels_list: predicted test labels
    @param classifier: classifier name to whose plot should be created
    @param threshs: threshold to evaluate accuracy of similarities
    @param print_stats:
    @return:
    """
    tprs_train, fprs_train = create_roc_curve_points(true_train_labels, pred_train_labels_list, print_stats, threshs)
    tprs_test, fprs_test = create_roc_curve_points(true_test_labels, pred_test_labels_list, print_stats, threshs)

    plt.plot(fprs_train, tprs_train, marker='.', label='train', color='green')
    plt.plot(fprs_test, tprs_test, marker='.', label='test', color='red')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for {classifier}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(['train', 'test'])
    plt.show()


def create_roc_curve_points(true_labels, pred_labels_list, print_stats, threshs):
    """
    Create points for roc curve
    @param true_labels: true labels
    @param pred_labels_list: predicted labels
    @param print_stats: whether statistical values should be printed
    @param threshs: threshold to evaluate accuracy of similarities
    @return: list of true positives, list of false positives
    """
    fprs = []
    tprs = []
    fprs.append(1)
    tprs.append(1)
    for t, pred_labels in zip(threshs, pred_labels_list):
        # calculate auc score and roc curve
        auc = roc_auc_score(true_labels, pred_labels)
        fpr, tpr, _ = roc_curve(true_labels, pred_labels)
        fprs.append(fpr[1])
        tprs.append(tpr[1])
        if print_stats:
            print(f'thresh={round(t, 3)} AUC={round(auc, 3)}\n')
    fprs.append(0)
    tprs.append(0)
    return tprs, fprs


def compute_mean_values(statistics):
    mean_train_accuracy = statistics['train_accuracy'].mean()
    mean_train_recall = statistics['train_recall'].mean()
    mean_train_specificity = statistics['train_specificity'].mean()
    mean_train_precision = statistics['train_precision'].mean()
    mean_test_accuracy = statistics['test_accuracy'].mean()
    mean_test_recall = statistics['test_recall'].mean()
    mean_test_specificity = statistics['test_specificity'].mean()
    mean_test_precision = statistics['test_precision'].mean()
    print("----------------------------")
    print(f'Evaluation results after {statistics.shape[0]} runs:')
    print(f"Mean Train Accuracy: {round(mean_train_accuracy*100, 2)}")
    print(f"Mean Train Recall: {round(mean_train_recall*100, 2)}")
    print(f"Mean Train Specificity: {round(mean_train_specificity*100, 2)}")
    print(f"Mean Train Precision: {round(mean_train_precision*100, 2)}")
    print(f"Mean Test Accuracy: {round(mean_test_accuracy*100, 2)}")
    print(f"Mean Test Recall: {round(mean_test_recall*100, 2)}")
    print(f"Mean Test Specificity: {round(mean_test_specificity*100, 2)}")
    print(f"Mean Test Precision: {round(mean_test_precision*100, 2)}")
    print("----------------------------")
    print("\n\n")
