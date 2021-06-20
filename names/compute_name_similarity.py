from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from matplotlib import pyplot as plt
import os

# CONSTANTS 
ID_MARK = '#id#'
BND_MARK = '#bnd#'
COL_MARK = '#col#'
MARKS = [ID_MARK, BND_MARK, COL_MARK]

# PARAMETERS
THRESHS = [5, 10, 20, 30, 50, 60, 70, 100]

ID_SIMILARITY_SCORE = 100
BRAND_SIMILARITY_SCORE = 10
COS_SIM_WEIGHT = 10
SAMEWORD_WEIGHT = 0

PRINT_STATS = False


# remove ids, colors and brands from name
def remove_identificators(word_list):
    words = []
    for word in word_list:
        if not any(ident in word for ident in MARKS):
            words.append(word)
    return words

# lower case all names in dataset
def lower_case(data):
    lowercased = []
    for d in data:
        lowercased.append(d.lower())
    return lowercased

# find diffference between two lists
def diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

# load file with names
def load_file(name_file):
    names = []
    file = open(name_file, 'r', encoding='utf-8')
    lines = file.read().splitlines() 
    for line in lines:
        names.append(line)
    return names

# save names, similarities and whether they are the same to output file
def save_to_file(output_file, name1, name2, score, are_names_same):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f'{name1}, {name2}, {score}, {are_names_same}\n')

# remove all ids, colors, brands, etc. from names
def remove_markers(data):
    replaced = []
    for d in data:
        for m in MARKS:
            d = d.replace(m, '')
        replaced.append(d)
    return replaced

# remove colors from names
def remove_colors(data):
    replaced = []
    for d in data:
        words = []
        for w in d.split(' '):
            if COL_MARK not in w:
                words.append(w)
        replaced.append(' '.join(words))
    return replaced

# compute tf.idf score for given dataset
def compute_tf_idf(data):
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b',lowercase=False)
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    if PRINT_STATS:
        print('Tf.idf matrix score is:')
        print(df)
    return df

# compute names similarity for 2 names and their indices in dataset
def compute_similarity_score(n1, i, n2, j, tf_idfs):
    similarity_score = 0
    name1 = n1.split(' ')
    name2 = n2.split(' ')

    # detect and compare ids
    id1 = [word for word in name1 if ID_MARK in word]
    id2 = [word for word in name2 if ID_MARK in word]
    if not id1==[]:
        match_ratio = len(set(id1) & set(id2))/len(id1)
        similarity_score += ID_SIMILARITY_SCORE*match_ratio
        if PRINT_STATS:
            print(f'Matching ids: {id1}')
            print(f'Ratio of matching ids: {match_ratio}')
    
    #detect and compare brands
    bnd1 = [word for word in name1 if BND_MARK in word]
    bnd2 = [word for word in name2 if BND_MARK in word]
    if not bnd1==[] and bnd1 == bnd2:
        match_ratio = len(set(bnd1) & set(bnd2))/len(bnd1)    
        similarity_score += BRAND_SIMILARITY_SCORE*match_ratio
        if PRINT_STATS:
            print(f'Matching brands: {bnd1}')
            print(f'Ratio of matching brands: {match_ratio}')       

    # ratio of the similar words
    list1 = set(name1)
    intersection = list1.intersection(name2)
    intersection_list = list(intersection)
    match_ratio = len(intersection_list)/len(name1)
    similarity_score += match_ratio*SAMEWORD_WEIGHT
    if PRINT_STATS:
        print(f'Ratio of common words in name1: {match_ratio}')
        print(f'Common words: {intersection_list}')
        print(f'Different words: {diff(name1, name2)}')

    # cosine similarity of vectors from tf.idf
    cos_sim = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]
    similarity_score += COS_SIM_WEIGHT*cos_sim

    if PRINT_STATS:
        print(f'Similarity score is: {similarity_score}')
    return similarity_score

# evaluate dataset - compute accuracy, confusion matric and plot ROC
def evaluate_dataset(scores):
    true_labels = [[row[2]] for row in scores]
    pred_labels_list = []
    precs = []
    recs = []
    for t in THRESHS:
        pred_labels = [[1 if row[3]>t else 0] for row in scores]
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

# compare whether names on indexes should be the same
def are_idxs_same(i, j):
    if i%5==1:
        return 1 if (i==j or i+1==j or i+2==j or i+3==j ) else 0
    if i%5==2:
        return 1 if (i==j or i+1==j or i+2==j) else 0
    if i%5==3:
        return 1 if (i==j or i+1==j) else 0
    if i%5==4:
        return 1 if (i==j) else 0
    return 1 if (i==j or i+1==j or i+2==j or i+3==j or i+4==j) else 0

# delete output file if it already exists
def remove_output_file_is_necessary(output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
     
# QUICK TEST DATA
# Apple MacBook Air 13 M1 #id#MGNE3CZ/A Gold
#name1 = "#bnd#Lenovo IdeaPad Gaming 3 #id#15IMH05 #id#81Y400K8CK"
#name2 =  "#bnd#Lenovo IdeaPad Gaming 3 #id#15IMH05 #id#81Y400K8CK"

#name1 = '#bnd#Apple MacBook Air 15 M1 #id#MGNE3CZ/A'
#name2 = '#bnd#Apple MacBook Air 13 M1 #id#MGNE3CZ/A'
   
     
def main():
    
    scores = []
    
    ''' FOR COMPARISON OF NAMES FROM 2 FILES 
    output_file = 'data/results/scores_ab.txt'
    remove_output_file_is_necessary(output_file)
    
    names_list1 = load_file('data/results/names_a_prepro.txt')
    names_list2 = load_file('data/results/names_b_prepro.txt')
    
    names_list1 = lower_case(names_list1)
    names_list2 = lower_case(names_list2)
    
    names_voc1 = remove_colors(names_list1)
    names_voc2 = remove_colors(names_list2)
    
    # tf.idf for creation of vectors of words
    names_voc = names_voc1 + names_voc2
    tf_idfs = compute_tf_idf(names_voc)
    
    for i, n1 in enumerate(names_list1):
        for j, n2 in enumerate(names_list2):
            similarity_score = compute_similarity_score(n1, i, n2, j, tf_idfs)
            are_names_same = 1 if i==j else 0
            scores.append([n1, n2, are_names_same, similarity_score])
            save_to_file(output_file, n1, n2, similarity_score, are_names_same)
    evaluate_dataset(scores)
    '''
    
     
    ''' FOR COMPARISON OF NAMES FROM 1 FILE   '''
    output_file = 'data/results/scores_100.txt'
    remove_output_file_is_necessary(output_file)
    
    names_list = load_file('data/results/names_100_prepro.txt')
    names_list = lower_case(names_list)
    names_list = remove_colors(names_list)
    
    tf_idfs = compute_tf_idf(names_list)
    for i, n1 in enumerate(names_list):
        for j, n2 in enumerate(names_list[i+1::]):
            j += i+1
            similarity_score = compute_similarity_score(n1, i, n2, j, tf_idfs)
            are_names_same = are_idxs_same(i ,j)
            
            #if are_names_same==1 and similarity_score<20:
            #    print(f'{i}| {names_list[i]} | {j} | {names_list[j]} | {similarity_score}')
            
            if PRINT_STATS:
                print(similarity_score)
                
            scores.append([n1, n2, are_names_same, similarity_score])
            save_to_file(output_file, n1, n2, similarity_score, are_names_same)
    evaluate_dataset(scores)
     
if __name__ == "__main__":
    main()
