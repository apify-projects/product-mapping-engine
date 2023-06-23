import pandas as pd
import json

d = pd.read_csv('promapen-train_data_translated.csv')
dd = pd.read_csv(('promapen-test_data_translated.csv'))
data = pd.concat([d, dd])
data.to_csv('promapen_translated.csv', index=False)
