import pandas as pd
import json
f = 'promap_multi_sears_translated-test_data_similarities.csv'

d = pd.read_csv(f)

d['short_description_id'] = [0]*len(d)
d['short_description_brand'] = [0]*len(d)
d['short_description_words'] = [0]*len(d)
d['short_description_cos'] = [0]*len(d)
d['short_description_descriptives'] = [0]*len(d)
d['short_description_units'] = [0]*len(d)
d['short_description_numbers'] = [0]*len(d)

d = d[['id1', 'id2', 'name_id', 'name_brand', 'name_words', 'name_cos',
       'name_descriptives', 'name_units', 'name_numbers',
       'short_description_id', 'short_description_brand',
       'short_description_words', 'short_description_cos',
       'short_description_descriptives', 'short_description_units',
       'short_description_numbers', 'long_description_cos',
       'long_description_descriptives', 'long_description_units',
       'long_description_numbers', 'specification_text_units',
       'specification_text_numbers', 'all_texts_id', 'all_texts_brand',
       'all_texts_words', 'all_texts_cos', 'all_texts_descriptives',
       'all_texts_units', 'all_texts_numbers', 'all_units_list',
       'all_ids_list', 'all_numbers_list', 'all_brands_list',
       'specification_key_matches', 'specification_key_value_matches',
       'hash_similarity', 'match']]
d.to_csv(f, index=False)

