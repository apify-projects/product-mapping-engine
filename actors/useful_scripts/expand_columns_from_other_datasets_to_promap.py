import pandas as pd

agtr = pd.read_csv('amazon_google-train_data_similarities.csv')
awtr = pd.read_csv('amazon_walmart-train_data_similarities.csv')
agte = pd.read_csv('amazon_google-test_data_similarities.csv')
awte = pd.read_csv('amazon_walmart-test_data_similarities.csv')

cztr = pd.read_csv('promapcz-train_data_similarities.csv')
entr = pd.read_csv('promapen-train_data_similarities.csv')
czte = pd.read_csv('promapcz-test_data_similarities.csv')
ente = pd.read_csv('promapen-test_data_similarities.csv')
aw= True
ag = False

print(awtr.columns)
if ag:
    awcols = ['id1', 'id2', 'name_id', 'name_brand', 'name_words', 'name_cos',
           'name_descriptives', 'name_units', 'name_numbers',
           'short_description_id', 'short_description_brand',
           'short_description_words', 'short_description_cos',
           'short_description_descriptives', 'short_description_units',
           'short_description_numbers', 'all_texts_id', 'all_texts_brand',
           'all_texts_words', 'all_texts_cos', 'all_texts_descriptives',
           'all_texts_units', 'all_texts_numbers', 'all_units_list',
           'all_ids_list', 'all_numbers_list', 'all_brands_list',
           'specification_key_matches', 'specification_key_value_matches',
           'match']
    awczte = czte[awcols]
    awcztr = cztr[awcols]
    awentr = entr[awcols]
    awente = ente[awcols]
    awagtr = agtr[awcols]
    awagte = agte[awcols]

    awcztr.to_csv('promapcz-train_data_similarities_amazon_google.csv', index=False) 
    awczte.to_csv('promapcz-test_data_similarities_amazon_google.csv', index=False) 
    awentr.to_csv('promapen-train_data_similarities_amazon_google.csv', index=False) 
    awente.to_csv('promapen-test_data_similarities_amazon_google.csv', index=False) 
    awagtr.to_csv('amazon_walmart-train_data_similarities_amazon_google.csv', index=False) 
    awagte.to_csv('amazon_walmart-test_data_similarities_amazon_google.csv', index=False)
    

if aw:
    awcols = ['id1', 'id2', 'name_id', 'name_brand', 'name_words', 'name_cos',
       'name_descriptives', 'name_units', 'name_numbers',
       'short_description_id', 'short_description_brand',
       'short_description_words', 'short_description_cos',
       'short_description_descriptives', 'short_description_units',
       'short_description_numbers', 'long_description_cos',
       'long_description_descriptives', 'long_description_units',
       'long_description_numbers', 'all_texts_id', 'all_texts_brand',
       'all_texts_words', 'all_texts_cos', 'all_texts_descriptives',
       'all_texts_units', 'all_texts_numbers', 'all_units_list',
       'all_ids_list', 'all_numbers_list', 'all_brands_list',
       'specification_key_matches', 'specification_key_value_matches',
       'match']
    awcols2 = ['id1', 'id2', 'name_id', 'name_brand', 'name_words', 'name_cos',
       'name_descriptives', 'name_units', 'name_numbers',
       'short_description_id', 'short_description_brand',
       'short_description_words', 'short_description_cos',
       'short_description_descriptives', 'short_description_units',
       'short_description_numbers', 'all_texts_id', 'all_texts_brand',
       'all_texts_words', 'all_texts_cos', 'all_texts_descriptives',
       'all_texts_units', 'all_texts_numbers', 'all_units_list',
       'all_ids_list', 'all_numbers_list', 'all_brands_list',
       'specification_key_matches', 'specification_key_value_matches',
       'match']

    awcztr = cztr[awcols]
    awczte = czte[awcols]
    awentr = entr[awcols]
    awente = ente[awcols]
    awagtr = agtr[awcols2]
    awagte = agte[awcols2]
    awagtr['long_description_cos']=[0]*len(awagtr)
    awagtr['long_description_descriptives']=[0]*len(awagtr)
    awagtr['long_description_numbers']=[0]*len(awagtr)
    awagtr['long_description_units']=[0]*len(awagtr)
    awagte['long_description_cos']= [0]*len(awagte)
    awagte['long_description_descriptives']=[0]*len(awagte)
    awagte['long_description_numbers']=[0]*len(awagte)
    awagte['long_description_units']= [0]*len(awagte)


    awcztr.to_csv('promapcz-train_data_similarities_amazon_walmart.csv', index=False) 
    awczte.to_csv('promapcz-test_data_similarities_amazon_walmart.csv', index=False) 
    awentr.to_csv('promapen-train_data_similarities_amazon_walmart.csv', index=False) 
    awente.to_csv('promapen-test_data_similarities_amazon_walmart.csv', index=False) 
    awagtr.to_csv('amazon_google-train_data_similarities_amazon_walmart.csv', index=False) 
    awagte.to_csv('amazon_google-test_data_similarities_amazon_walmart.csv', index=False) 
# 'long_description_cos','long_description_descriptives', 'long_description_units','long_description_numbers',