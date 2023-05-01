import pandas as pd


test = pd.read_csv('amazon_walmart-train_data_similarities.csv')

new_test = test.iloc[: , :20]
second_part = test.iloc[: , 20:-1]
match = test.iloc[: , -1]

new_test['specification_text_units'] = [0]*len(test)
new_test['specification_text_numbers'] = [0]*len(test)
new_test = pd.concat([new_test,second_part], axis=1)
new_test['hash_similarity'] = [0]*len(test)
new_test['match']=match.values

new_test.to_csv('amazon_walmart-train_data_similarities_extended.csv', index=False)

