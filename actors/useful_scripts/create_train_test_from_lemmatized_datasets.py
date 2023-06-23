import pandas as pd


filename = 'promapen'
first = pd.read_csv(filename+'_morphoditta_dataset2_without_marks.csv')
first = first.rename(columns={'name': 'name1', 'short_description':'short_description1', 'long_description':'long_description1', 'specification_text':'specification_text1',
       'all_texts':'all_texts1', 'price':'price1'})
second = pd.read_csv(filename+'_morphoditta_dataset1_without_marks.csv')
second = second.rename(columns={'name': 'name2', 'short_description':'short_description2', 'long_description':'long_description2', 'specification_text':'specification_text2',
       'all_texts':'all_texts2', 'price':'price2'})
result = pd.concat([first, second], axis=1)


all_data = pd.read_csv(filename+'_enriched.csv')
all_data = all_data[[ 'id1','id2', 'match', 'category', 'match_type', 'image_url1', 'image_url2','specification1', 'specification2']]
result = pd.concat([result, all_data], axis=1)
data = result[['name1', 'short_description1', 'long_description1', 'specification1', 'specification_text1', 'price1', 'id1', 'name2', 'short_description2',
       'long_description2', 'specification2', 'specification_text2', 'price2', 'id2',
       'match', 'category', 'match_type', 'image_url1', 'image_url2']]



train = pd.read_csv(filename+'-train_data_similarities.csv') 
test = pd.read_csv(filename+'-test_data_similarities.csv') 

data["new"] = data["id1"] + data["id2"]
train["new"] = train["id1"] + train["id2"]
test["new"] = test["id1"] + test["id2"]
train_list = train['new'].values.tolist()                  
test_list = test['new'].values.tolist()   
        
new_data = data[data['new'].isin(train_list)]
new_data = new_data.drop_duplicates()
new_data = new_data.drop(columns={'new'})
new_data.to_csv(filename+'-train_data.csv', index=False)

new_dataa = data[data['new'].isin(test_list)]
new_dataa = new_dataa.drop_duplicates()
new_dataa = new_dataa.drop(columns={'new'})
new_dataa.to_csv(filename+'-test_data.csv', index=False)
