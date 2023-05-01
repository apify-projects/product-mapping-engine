import pandas as pd
import numpy as np
data = pd.read_csv('promapen_cats.csv')
data_test = pd.read_csv('promapen-test_data_similarities.csv')


data['new'] = data[['id1', 'id2']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
data_test['new'] = data_test[['id1', 'id2']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
data = data[['new', 'match_type']]
data_test  = data_test.merge(data, on='new', how='left')
data_test= data_test.drop(columns={'new'})

data = data_test[data_test['match_type']!='close_nonmatch']
data = data.drop(columns={'match_type'})
data.to_csv('promapen_medium_nonmatches-test_data_similarities.csv', index=False)
print(len(data))
data = data_test[data_test['match_type']!='medium_nonmatch']
data = data.drop(columns={'match_type'})
data.to_csv('promapen_close_nonmatches-test_data_similarities.csv', index=False)
print(len(data))