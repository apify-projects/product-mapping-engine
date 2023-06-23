import pandas as pd
import numpy as np

# load train test data
all_data = pd.read_csv('promapen-all_pairs.csv')
train_data = pd.read_csv('promapen-train_data_similarities.csv')
test_data = pd.read_csv('promapen-test_data_similarities.csv')
print(len(all_data))
print(len(train_data))
print(len(test_data))
test_duplicates = False


# split to train and test data
all_data['ids'] = all_data["id1"] + all_data["id2"]
train_data = train_data['id1'] + train_data['id2']
test_data = test_data['id1'] + test_data['id2']
print(len(train_data))
print(len(test_data))


# find product pairs by id
train_data_all = all_data[all_data['ids'].isin(train_data)]
test_data_all = all_data[all_data['ids'].isin(test_data)]
print(len(train_data_all))
print(len(test_data_all))


train_data_all = train_data_all.drop(columns={'ids'})
test_data_all = test_data_all.drop(columns={'ids'})
# save output data
train_data_all.to_csv('promapen-train_data.csv', index=False)
test_data_all.to_csv('promapen-test_data.csv', index=False)