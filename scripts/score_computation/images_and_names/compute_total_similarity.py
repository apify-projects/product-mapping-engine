from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from scripts.evaluate_classifier import plot_roc


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


def evaluate_dataset(scores, threshs, print_stats):
    """
    Eevaluate dataset - compute accuracy, confusion matric and plot ROC
    @param scores: dataset with names similarities
    @param threshs: threshold to evaluate accuracy of similarities
    @return:
    """
    true_labels = [[row[3]] for row in scores]
    pred_labels_list = []
    precs = []
    recs = []
    for t in threshs:
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
    plot_roc(true_labels, pred_labels_list, threshs, print_stats)


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
