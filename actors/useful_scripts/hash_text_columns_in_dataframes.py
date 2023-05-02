import pandas as pd
import numpy as np
import hashlib

TEXT_HASH_SIZE = 16
d1= pd.read_csv('promapen1.csv')
d2= pd.read_csv('promapen2.csv')

def hex_to_bin(val):
    hash_bin = []
    for i in val:
        hash_bin.append(bin(int(i, 16))[2:].zfill(4))
    return str.join('', [val for sub in hash_bin for val in sub])

def dec_similarity(list1, list2):
    diff = 0
    for i, j in zip(list1, list2):
        diff += (abs(i - j))
    return diff
def hash_text_using_sha256(text):

    xx = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), TEXT_HASH_SIZE)
    return [int(x) for x in str(xx)]


d1 = d1.drop(columns={'price'})
for c in d1.columns:
    vals = []
    for v in d1[c].values:
        vals.append(hash_text_using_sha256(' '.join(v)))
    d1[c] = vals
    
d2 = d2.drop(columns={'price'})
for c in d2.columns:
    vals = []
    for v in d2[c].values:
        vals.append(hash_text_using_sha256(' '.join(v)))
    d2[c] = vals    


sim_df = pd.DataFrame(columns=d1.columns)
for c in d1.columns:
    similarities = []
    for first_hash, second_hash in zip(d1[c].values, d2[c].values):
        sim = dec_similarity(first_hash, second_hash)
        similarities.append(sim)
    sim_df[c] = similarities
sim_df=(sim_df-sim_df.mean())/sim_df.std()
    
sim_df.to_csv('promapen_text_hash_similarities.csv', index=False)
