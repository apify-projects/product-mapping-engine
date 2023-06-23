import pandas as pd
import numpy as np
import copy

d = pd.read_csv('ProMapEn/promapen-train_data.csv')

duni = d.drop_duplicates('id1')

dsam = np.random.choice(duni.index.values, 356, replace=False) #356
ds = d.loc[dsam]
print(d.columns)


newdf = pd.DataFrame(columns=d.columns)
for index, row in ds.iterrows():
    x =  d.loc[(d['id1'] != row['id1']) & (d['category'] != row['category'])] 
    one = x.sample(n=1, random_state=1) 
    new = pd.DataFrame()
    new1 = pd.DataFrame(row[['name1', 'short_description1', 'long_description1', 'specification1', 'image1','price1', 'id1', 'category', 'image_url1']]) 
    new2 = pd.DataFrame(one[['name2', 'short_description2', 'long_description2', 'specification2', 'image2', 'price2', 'id2', 'image_url2']])
    new1 = new1.T
    new1['x'] = 0
    new2['x']= 0
    new = pd.merge(new1, new2, how= 'inner', on=['x'])
    new = new.drop(columns=['x'], axis=1)
    new['match'] = 0
    new['match_type'] = 'distant_nonmatch'
    newdf = newdf.append(new, ignore_index = True)


neww = pd.concat([d, newdf])
neww.to_csv('ProMapEn/promapenext-train_data.csv', index=False)
