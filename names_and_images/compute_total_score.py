from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from matplotlib import pyplot as plt

# PARAMS
IMG_WEIGHT = 1
NAME_WEIGHT = 1
THRESH_IMG = 0
PRINT_STATS = True
THRESHS = [50, 70, 90, 100, 120]


names_file = '../names/data/results/scores_ab.txt'
images_file = '../img_hash/data/results/all_dist_bin_cropped.txt'
output_file = 'results/name_img_similarity.txt'

def load_file(file_name):
    data = []
    file = open(file_name, 'r', encoding='utf-8')
    lines = file.read().splitlines() 
    for line in lines:
        data.append(line)
    return data
      
# save names, similarities and whether they are the same to output file
def save_to_file(data):
    with open(output_file, 'a', encoding='utf-8') as f:
        for d in data:
            f.write(f'{d[0]}, {d[1]}, {d[2]}, {d[3]}\n')
        
# evaluate dataset - compute accuracy, confusion matric and plot ROC
def evaluate_dataset(distances, true_labels):
    pred_labels_list = []
    precs = []
    recs = []
    for t in THRESHS:
        pred_labels = [[1 if val>t else 0] for val in distances]
        pred_labels_list.append(pred_labels)
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)
        precs.append(prec)
        rec = recall_score(true_labels, pred_labels)
        recs.append(rec)
        if PRINT_STATS:
            print(f'For thresh {t}: \n Accuracy {acc} \n Precision {prec} \n Recall {rec}')
            print('Confusion matrix')
            print(conf_matrix) 
            print('======')
    plot_roc(true_labels, pred_labels_list)

# plot roc curve
def plot_roc(true_labels, pred_labels_list):
    fprs = []
    tprs = []
    labels= ''
    fprs.append(1)
    tprs.append(1)
    for t, pred_labels in zip(THRESHS, pred_labels_list):
        # calculate auc score and roc curve
        auc = roc_auc_score(true_labels, pred_labels)
        fpr, tpr, _ = roc_curve(true_labels, pred_labels)
        fprs.append(fpr[1])
        tprs.append(tpr[1])
        labels+=f'thresh={t} AUC={round(auc, 3)}\n' 
        if PRINT_STATS:
            print('ROC AUC=%.3f' % (auc))
    fprs.append(0)
    tprs.append(0)

    
    plt.plot(fprs, tprs, marker='.', label=labels, color='red')
    
    plt.plot([0, 1], [0, 1],'b--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
def main():
    names_data = load_file(names_file)
    images_data = load_file(images_file)

    total_distances = []
    THRESH_IMG = max([float(i.split(',')[2]) for i in images_data])
    #print(max_val)
    
    for i, (name, img) in enumerate(zip(names_data, images_data)):
        imgs_dst = float(img.split(',')[2])
        name = name.split(',')
        name_sim = float(name[2])
        imgs_sim = 0 if imgs_dst > THRESH_IMG else (THRESH_IMG-imgs_dst)/THRESH_IMG*100
        distance = name_sim*NAME_WEIGHT + imgs_sim*IMG_WEIGHT
        if PRINT_STATS:
            print(f'{name[0]} | {name[1]} | {distance}')
        total_distances.append([name[0], name[1], distance, int(name[3])])
    
    dstces =  [i[2] for i in total_distances]
    labels = [i[3] for i in total_distances]
    
    save_to_file(total_distances)
    evaluate_dataset(dstces, labels)
if __name__ == "__main__":
    main()
        