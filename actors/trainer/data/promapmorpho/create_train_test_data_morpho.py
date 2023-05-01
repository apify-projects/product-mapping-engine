import pandas as pd
import numpy as np

cz = True
if cz:
    ctr = pd.read_csv('promapcz-train.csv')
    ctr = ctr[['id1', 'id2']]
    cte = pd.read_csv('promapcz-test.csv')
    cte = cte[['id1', 'id2']]
else:
    ctr = pd.read_csv('promapen-train.csv')
    ctr = ctr[['id1', 'id2']]
    cte = pd.read_csv('promapen-test.csv')
    cte = cte[['id1', 'id2']]

folder = 'cz_none'
'''
file = 'promapen_cos'
data = pd.read_csv(f'{folder}/{file}_similarities.csv')
d = pd.merge(ctr, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d.to_csv(f'{folder}/{file}-train_data_similarities.csv', index=False)
d = pd.merge(cte, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d.to_csv(f'{folder}/{file}-test_data_similarities.csv', index=False)
'''

file = 'promapcz_euclid'
data = pd.read_csv(f'{folder}/{file}_similarities.csv')
d = pd.merge(ctr, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d.to_csv(f'{folder}/{file}-train_data_similarities.csv', index=False)
d = pd.merge(cte, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d.to_csv(f'{folder}/{file}-test_data_similarities.csv', index=False)
'''

file = 'promapen_manhattan'
data = pd.read_csv(f'{folder}/{file}_similarities.csv')
d = pd.merge(ctr, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d.to_csv(f'{folder}/{file}-train_data_similarities.csv', index=False)
d = pd.merge(cte, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d.to_csv(f'{folder}/{file}-test_data_similarities.csv', index=False)
'''