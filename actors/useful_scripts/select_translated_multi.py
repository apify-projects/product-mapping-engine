import pandas as pd

dt = pd.read_csv('promapen-train_data_translated.csv')
de = pd.read_csv('promapen-test_data_translated.csv')
file = 'promapen_multi-train_data.csv'
dm = pd.read_csv(file)

d = pd.concat([dt, de])
dm = dm[['id1', 'id2']]
dt = pd.merge(dm, d, how='left', left_on=['id1', 'id2'], right_on=['id1', 'id2'])
dt.to_csv(f'{file[:-4]}_translated.csv', index=False)
