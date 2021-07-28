from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from score_computation.names.compute_names_similarity import ID_MARK, BND_MARK
from score_computation.names.compute_names_similarity import remove_markers


def compute_name_similarities(n1, n2, i, j, tf_idfs):
    """
    Compute similarity score of two products comparing images and names
    @param n1: first name
    @param n2: second name
    @param i: first name index
    @param j: second name index
    @param tf_idfs: tf ids of names
    @return: similarity score of names and images
    """
    name1 = n1.split(' ')
    name2 = n2.split(' ')
    match_ratios = {}

    # detect and compare ids
    id1 = [word for word in name1 if ID_MARK in word]
    id2 = [word for word in name2 if ID_MARK in word]
    if not id1 == []:
        match_ratios['id'] = len(set(id1) & set(id2)) / len(id1)

    # detect and compare brands
    bnd1 = [word for word in name1 if BND_MARK in word]
    bnd2 = [word for word in name2 if BND_MARK in word]
    if not bnd1 == [] and bnd1 == bnd2:
        match_ratios['brand'] = len(set(bnd1) & set(bnd2)) / len(bnd1)

    # ratio of the similar words
    name1 = remove_markers(name1)
    name2 = remove_markers(name2)
    list1 = set(name1)
    intersection = list1.intersection(name2)
    intersection_list = list(intersection)
    match_ratios['words'] = len(intersection_list) / len(name1)

    # cosine similarity of vectors from tf.idf
    match_ratios['cos'] = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]

    return match_ratios


def plot_roc(true_labels, pred_labels_list, threshs, print_stats):
    """
    Plot roc curve
    @param true_labels: true labels
    @param pred_labels_list: predicted labels
    @param threshs: threshold to evaluate accuracy of similarities
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
        labels += f'thresh={t} AUC={round(auc, 3)}\n'
        if print_stats:
            print(f'ROC AUC={round(auc, 3)}')
    fprs.append(0)
    tprs.append(0)

    plt.plot(fprs, tprs, marker='.', label=labels, color='red')

    plt.plot([0, 1], [0, 1], 'b--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
