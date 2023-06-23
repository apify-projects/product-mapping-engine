import pandas as pd

dt = pd.read_csv('promapen-train_data.csv')
de = pd.read_csv('promapen-test_data.csv')
data = pd.concat([dt, de])

dt = pd.read_csv('promapmulti-train_data_similarities.csv')
de = pd.read_csv('promapmulti-test_data_similarities.csv')
dt = dt[['id1', 'id2']]
de = de[['id1', 'id2']]


dt = pd.merge(dt, data, how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
de = pd.merge(de, data, how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
dt.to_csv('promapmulti_lemmatized-train_data.csv', index=False)
dt.to_csv('promapmulti_lemmatized-test_data.csv', index=False)
