import pandas as pd
import json


d = pd.read_csv('promapen-train_data_translated.csv')
dd = pd.read_csv(('promapen-test_data_translated.csv'))
data = pd.read_csv(('promapen-translated_similarities.csv'))
d = d[['id1', 'id2']]
dd = dd[['id1', 'id2']]
dd= pd.merge(dd, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
d = pd.merge(d, data,  how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
dd.to_csv('promapen-train_data_translated_similarities.csv', index=False)
d.to_csv('promapen-train_data_translated_similarities.csv', index=False)