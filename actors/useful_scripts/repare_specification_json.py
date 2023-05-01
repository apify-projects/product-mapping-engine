import pandas as pd
import json

filename = 'promapcz-test_data_translated.csv'
data = pd.read_csv(filename)
d = data[['specification1','specification2']]
d = d.replace({"\\\\xa0" : ""}, regex=True)
d = d.replace({"\\\\'s" : "'s"}, regex=True)
d = d.replace({"" : ''}, regex=True)
d = d.replace({'"' : "'"}, regex=True)
d = d.replace({"'key': '" : '"key": "'}, regex=True)
d = d.replace({"', 'value': '" : '", "value": "'}, regex=True)
d = d.replace({"'}, {" : '"}, {'}, regex=True)
d = d.replace({"'}]" : '"}]'}, regex=True)
d = d.replace({"', 'value':" : '", "value":'}, regex=True)
d = d.replace({"'key':" : '"key":'}, regex=True)
d = d.replace({"'value': '" : '"value": "'}, regex=True)
#d = d.replace({"" : ''}, regex=True)

data['specification1'] = d['specification1']
data['specification2'] = d['specification2']
data.to_csv('promapcz-test_translated.csv', index=False)

