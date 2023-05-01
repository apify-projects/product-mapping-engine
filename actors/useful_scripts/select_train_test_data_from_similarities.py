import pandas as pd
import numpy as np
import requests
import json
import sys



data = pd.read_csv('promapcz_enriched.csv')
d = pd.read_csv('promapcz-train_data_similarities.csv')
d = d[['id1', 'id2']]
d['new'] = d.id1.astype(str) + d.id2.astype(str)
d = d.drop(columns={'id1', 'id2'})
data['new'] = data.id1.astype(str) + data.id2.astype(str)
dat = d.merge(data, on='new', how='left')
dat = dat.drop(columns={'new'})
dat.to_csv('promapcz-train_data.csv')