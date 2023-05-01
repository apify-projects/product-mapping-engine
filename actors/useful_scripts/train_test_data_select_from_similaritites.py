import pandas as pd

filename = 'amazon_google'
data = pd.read_csv(filename+'.csv')
train = pd.read_csv(filename+'-train_data_similarities.csv') 

data["new"] = data["id1"] + data["id2"]
train["new"] = train["id1"] + train["id2"]
train_list = train['new'].values.tolist()                  
             
new_data = data[data['new'].isin(train_list)]
new_data = new_data.drop_duplicates()
new_data = new_data.drop(columns={'new'})
new_data.to_csv(filename+'-train_data.csv', index=False)
