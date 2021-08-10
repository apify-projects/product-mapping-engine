import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def evaluate_classifier(classifier, classifier_class_name, data):
    """
    Evaluate classifier - let it train and predict weights for images and hashes and plot ROC curve
    @param classifier: classifier to train and evaluate
    @param classifier_class_name: name of the classifier
    @param data: data to train and evaluate classifier
    @return:
    """
    train, test = train_test_split(data, test_size=0.25)
    classifier.fit(train)
    out_train, scores_train = classifier.predict(train)
    out_test, scores_test = classifier.predict(test)
    evaluate_predictions(train, out_train, "train")
    evaluate_predictions(test, out_test, "test")
    threshs = create_thresh(scores_train, 10)
    out_train = []
    out_test = []
    for t in threshs:
        out_train.append([0 if score < t else 1 for score in scores_train])
        out_test.append([0 if score < t else 1 for score in scores_test])
    plot_roc(train['match'].tolist(), out_train, threshs, classifier_class_name, print_stats=False, )
    plot_roc(test['match'].tolist(), out_test, threshs, classifier_class_name, print_stats=False)


def evaluate_predictions(data, outputs, data_type):
    """
    Compute accuracy, precision, recall and show confusion matrix
    @param data: all dataset used for training and prediction
    @param outputs: predicted outputs
    @param data_type: whether data are train or test
    @return:
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
    print("Classifier results for {} data".format(data_type))
    print("----------------------------")
    print("Accuracy: {}".format((data_count - mismatched_count) / data_count))
    print("Recall: {}".format(true_positive_count / actual_positive_count))
    print("Specificity: {}".format(true_negative_count / actual_negative_count))
    print("Precision: {}".format(true_positive_count / (true_positive_count + false_positive_count)))
    conf_matrix = confusion_matrix(data['match'], data['match_prediction'])
    print('Confusion matrix')
    print(conf_matrix)
    print("----------------------------")
    print("\n\n")

    mismatched.to_csv("mismatches.csv")


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


def plot_roc(true_labels, pred_labels_list, threshs, classifier, print_stats=False):
    """
    Plot roc curve
    @param classifier: classifier name to whose plot should be created
    @param true_labels: true labels
    @param pred_labels_list: predicted labels
    @param threshs: threshold to evaluate accuracy of similarities
    @param print_stats: whether statistical values should be printed
    @return:
    """
    fprs = []
    tprs = []
    labels = ''
    fprs.append(1)
    tprs.append(1)
    for t, pred_labels in zip(threshs, pred_labels_list):
        # calculate auc score and roc curve
        auc = roc_auc_score(true_labels, pred_labels)
        fpr, tpr, _ = roc_curve(true_labels, pred_labels)
        fprs.append(fpr[1])
        tprs.append(tpr[1])
        if print_stats:
            labels += f'thresh={round(t, 3)} AUC={round(auc, 3)}\n'
            print(f'ROC AUC={round(auc, 3)}')
    fprs.append(0)
    tprs.append(0)

    plt.plot(fprs, tprs, marker='.', label=labels, color='red')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for {classifier}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
