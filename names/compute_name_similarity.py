from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from matplotlib import pyplot as plt

ID_MARK = '#id#'
BND_MARK = '#bnd#'
COL_MARK = '#col#'
MARKS = [ID_MARK, BND_MARK, COL_MARK]
THRESH = 10
 
ID_SIMILARITY_SCORE = 100
BRAND_SIMILARITY_SCORE = 10
COS_SIM_WEIGHT = 10
SIM_WORDS_WEIGHT = 10

PRINT_STATS = False
output_file = 'data/results/score.txt'

# remove ids, colors and brands from name
def remove_identificators(word_list):
    words = []
    for word in word_list:
        if not any(ident in word for ident in MARKS):
            words.append(word)
    return words

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

def save_to_file(name1, name2, score):
    with open(output_file, 'a', encoding='utf-8') as f:
       f.write(f'{name1}, {name2}, {score}\n')
     
def remove_markers(data):
    replaced = []
    for d in data:
        for m in MARKS:
            d = d.replace(m, '')
        replaced.append(d)
    return replaced

def compute_tf_idf(data1, data2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([data1, data2])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df = df.T
    return df #.to_dict()

# Apple MacBook Air 13 M1 #id#MGNE3CZ/A Gold
#name1 = "#bnd#Lenovo IdeaPad Gaming 3 #id#15IMH05 #id#81Y400K8CK"
#name2 =  "#bnd#Lenovo IdeaPad Gaming 3 #id#15IMH05 #id#81Y400K8CK"

#name1 = '#bnd#Apple MacBook Air 15 M1 #id#MGNE3CZ/A'
#name2 = '#bnd#Apple MacBook Air 13 M1 #id#MGNE3CZ/A'

names_list1 = load_file('data/results/names_a_prepro.txt')
names_list2 = load_file('data/results/names_b_prepro.txt')
scores = []

names_voc1 = remove_markers(names_list1)
names_voc2 = remove_markers(names_list2)

names_voc = names_voc1 + names_voc2

# tf.idf for creation of vectors of words
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b',lowercase=False)
vectors = vectorizer.fit_transform(names_voc)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
if PRINT_STATS:
    print('Tf.idf matrix score is:')
    print(df.T)


for i, n1 in enumerate(names_list1):
    for j, n2 in enumerate(names_list2):
        similarity_score = 0
        name1 = n1.split(' ')
        name2 = n2.split(' ')

        # detect and compare ids
        id1 = [word for word in name1 if ID_MARK in word]
        id2 = [word for word in name2 if ID_MARK in word]
        if not id1==[] and id1 == id2:
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
           
        # remove marked words
        words1 = remove_identificators(name1)
        words2 = remove_identificators(name2)

        
        # ratio of the similar words
        list1 = set(words1)
        intersection = list1.intersection(words2)
        intersection_list = list(intersection)
        match_ratio = len(intersection_list)/len(words1)
        if PRINT_STATS:
            print(f'Ratio of common words in name1: {match_ratio}')
            print(f'Common words: {intersection_list}')
            print(f'Different words: {diff(words1, words2)}')

        # cosine similarity of vectors from tf.idf
        cos_sim = cosine_similarity([df.iloc[i].values, df.iloc[j+10].values])[0][1]
        similarity_score += COS_SIM_WEIGHT*cos_sim
        
        if PRINT_STATS:
            print(f'Similarity score is: {similarity_score}')
        save_to_file(n1, n2, similarity_score)
        
        scores.append([n1, n2, 1 if i==j else 0, 1 if similarity_score>THRESH else 0, similarity_score])
        
# ROC curve
true_labels = [row[2] for row in scores]
pred_labels = [row[3] for row in scores]

# calculate auc score and roc curve
auc = roc_auc_score(true_labels, pred_labels)

random_probs = [0 for i in range(len(true_labels))]
p_fpr, p_tpr, _ = roc_curve(true_labels, random_probs, pos_label=1)
if PRINT_STATS:
    print('ROC AUC=%.3f' % (auc))
fpr, tpr, _ = roc_curve(true_labels, pred_labels)
plt.plot(fpr, tpr, marker='.', label='ROC AUC=%.3f' % (auc), color='red')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
