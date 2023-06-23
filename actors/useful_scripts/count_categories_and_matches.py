import pandas as pd
import numpy as np
data = pd.read_csv('promapenext.csv')

values = ['1_pets', '2_bags', '3_garden', '4_appliances', '5_phones', 
          '6_household',  '7_laptops', '8_toys', '9_clothes', '10_health'] #'8_tvs', '9_headphones', '10_fridges']
          
print(len(data))
for value in values:
    data_part = data[data['category']==value]
    data_part_matches = data_part[data_part['match_type']=='match']
    data_part_close_nonmatches = data_part[data_part['match_type']=='close_nonmatch']
    data_part_medium_nonmatches = data_part[data_part['match_type']=='medium_nonmatch']
    data_part_distant_nonmatches = data_part[data_part['match_type']=='distant_nonmatch']
    print(value)
    print('total len')
    print(len(data_part))
    print('matches')
    print(len(data_part_matches))
    print('close nonmatches')
    print(len(data_part_close_nonmatches))
    print('medium nonmatches')
    print(len(data_part_medium_nonmatches))
    print('distant nonmatches')
    print(len(data_part_distant_nonmatches))
    print('===')

print(len(data[data['match_type']=='medium_nonmatch']))
print(len(data[data['match_type']=='close_nonmatch']))
print(len(data[data['match_type']=='distant_nonmatch']))
print(len(data[data['match_type']=='match']))

x =data[['id1']].values
print(len(np.unique(x)))


x =data[['id2']].values
print(len(np.unique(x)))

data = data.drop(columns=['match_type', 'category'])
data['match'] = data['match'].astype(int)
#data.to_csv('full-en-dataset-all_pairs.csv', index=False)