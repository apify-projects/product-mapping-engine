import pandas as pd
import numpy as np
data = pd.read_csv('promapcz_enriched.csv')
test = pd.read_csv('promapcz_enriched-train_data_similarities.csv')


print(len(test))
data['new'] = data[['id1', 'id2']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
data = data[['new', 'match_type']]
test['new'] = test[['id1', 'id2']].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)

test  = test.merge(data, on='new', how='left')

test= test.drop(columns={'new'})


print(len(test))
test.to_csv('promapcz_enriched-train_data_similarities.csv', index=False)
