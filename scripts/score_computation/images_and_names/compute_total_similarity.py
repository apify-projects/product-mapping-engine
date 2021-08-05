import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score


def load_file(file_name):
    """
    Load input files with distances
    @param file_name: name of the input file
    @return: loaded data
    """
    data = []
    file = open(file_name, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        data.append(line)
    return data


def save_to_file(data, output_file):
    """
    Save names, similarities and whether they are the same to output file
    @param data: dat to be saved
    @param output_file: file  name to save the data
    @return:
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        for d in data:
            f.write(f'{d[0]}, {d[1]}, {d[2]}, {d[3]}\n')


def create_thresh(scores, intervals):
    """
    Create dummy threshs from values by sorting them and splitting into k inretvals of the same length
    @param scores: data to create threshs
    @param intervals: how many thresh should be created
    @return: threshs
    """
    scores = np.asarray(sorted(scores))
    subarrays = np.array_split(scores, intervals)
    return [(s[-1]) for s in subarrays][:-1]


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
        if print_stats:
            labels += f'thresh={round(t,3)} AUC={round(auc, 3)}\n'
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


def evaluate_dataset(scores, chunks, print_stats):
    """
    Eevaluate dataset - compute accuracy, confusion matric and plot ROC
    @param scores: dataset with names similarities
    @param chunks: number of threshs to be created
    @return:
    """
    true_labels = [[row[3]] for row in scores]
    pred_labels_list = []
    precs = []
    recs = []
    chunks = create_thresh([i[2] for i in scores], chunks)
    for t in chunks:
        pred_labels = [[1 if row[2] > t else 0] for row in scores]
        pred_labels_list.append(pred_labels)
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)
        precs.append(prec)
        rec = recall_score(true_labels, pred_labels)
        recs.append(rec)
        if print_stats:
            print(f'For thresh {t}: \n Accuracy {acc} \n Precision {prec} \n Recall {rec}')
            print('Confusion matrix')
            print(conf_matrix)
            print('======')
    plot_roc(true_labels, pred_labels_list, chunks, print_stats)


def compute_distance(images_data, names_data, name_weight, image_weight, print_stats):
    """
    Compute distance among products comparing name and image distance
    @param images_data: input data with product images
    @param names_data: input data with product names
    @param name_weight: weight of name similarity
    @param image_weight: weight of name image
    @param print_stats: indicator whether print statistical values
    @return: distances among products
    """
    total_distances = []
    thresh_img = max([float(i.split(',')[2]) for i in images_data])
    if print_stats:
        print(f'Images thresh is: {thresh_img}')
    for i, (name, img) in enumerate(zip(names_data, images_data)):
        imgs_dst = float(img.split(',')[2])
        name = name.split(',')
        name_sim = float(name[2])
        imgs_sim = 0 if imgs_dst > thresh_img else (thresh_img - imgs_dst) / thresh_img * 100
        distance = name_sim * name_weight + imgs_sim * image_weight
        if print_stats:
            print(f'{name[0]} | {name[1]} | {distance}')
        total_distances.append([name[0], name[1], distance, int(name[3])])
    return total_distances
