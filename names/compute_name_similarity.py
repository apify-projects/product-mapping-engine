from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ID_MARK = '#id#'
BND_MARK = '#bnd#'
COL_MARK = '#col#'
MARKS = [ID_MARK, BND_MARK, COL_MARK]

ID_SIMILARITY_SCORE = 100
BRAND_SIMILARITY_SCORE = 10
COS_SIM_WEIGHT = 1
SIM_WORDS_WEIGHT = 1

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

# Apple MacBook Air 13 M1 #id#MGNE3CZ/A Gold
name1 = "#bnd#Lenovo IdeaPad Gaming 3 #id#15IMH05 #id#81Y400K8CK"
name2 =  "#bnd#Lenovo IdeaPad Gaming 3 #id#15IMH05 #id#81Y400K8CK"

name1 = '#bnd#Apple MacBook Air 15 M1 #id#MGNE3CZ/A'
name2 = '#bnd#Apple MacBook Air 13 M1 #id#MGNE3CZ/A'

similarity_score = 0


name1 = name1.split(' ')
name2 = name2.split(' ')

# detect and compare ids
id1 = [word for word in name1 if ID_MARK in word]
id2 = [word for word in name2 if ID_MARK in word]
if id1 == id2:
    match_ratio = len(set(id1) & set(id2))/len(id1)
    similarity_score += ID_SIMILARITY_SCORE*match_ratio
    print(f'Matching ids: {id1}')
    print(f'Ratio of matching ids: {match_ratio}')

#detect and compare brands
bnd1 = [word for word in name1 if BND_MARK in word]
bnd2 = [word for word in name2 if BND_MARK in word]
if bnd1 == bnd2:
    match_ratio = len(set(id1) & set(id2))/len(id1)    
    similarity_score += BRAND_SIMILARITY_SCORE*match_ratio
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
print(f'Ratio of common words in name1 {match_ratio}')
print(f'Common words: {intersection_list}')
print(f'Different words: {diff(words1, words2)}')

# tf.idf for creation of vectors of words
names = [' '.join(words1), ' '.join(words2)]
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b',lowercase=False)
vectors = vectorizer.fit_transform(names)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print('Tf.idf matrix score is:')
print(df.T)

# cosine similarity of vectors from tf.idf
cos_sim = cosine_similarity(df)[0][1]
print(cos_sim)
similarity_score += COS_SIM_WEIGHT*cos_sim

print(f'Similarity score is: {similarity_score}')