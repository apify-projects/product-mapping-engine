import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re


input_file1 = 'data/names_mall.csv'
input_file2 = 'data/names_czc.csv'
output_file = 'data/tf_idf.csv'
encoding= 'utf-16'
colors = ['šedý', 'šedá', 'grey', 'gray', 'černý', 'černá', 'black', 'bílý', 'bílá', 'white', 
          'stříbrný','stříbrná', 'silver', 'červený','červená', 'red', 'modrý', 'modrá', 'blue', 
          'zelená', 'zelený', 'green', 'zlatá', 'zlatý', 'gold', 'béžový', 'béžová']

def remove_nonalfanum_chars(data):
    data_list = []
    for word in data.split(' '):
        word = re.sub(r"[\W_]+", "", word, flags=re.UNICODE).lower()
        if any(w in word for w in colors):
            data_list.append('color')
        elif word.isnumeric():
            continue
        else:
            data_list.append(word)
    return data_list

df1 = pd.read_csv(input_file1, encoding=encoding)
df1.dropna(inplace=True)
df2 = pd.read_csv(input_file2, encoding=encoding)
df2.dropna(inplace=True)

data1 = str.join(' ', [val for val in df1.values[:,1]])
data2 = str.join(' ', [val for val in df2.values[:,1]])

data1 = str.join(' ', remove_nonalfanum_chars(data1))
data2 = str.join(' ', remove_nonalfanum_chars(data2))


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([data1, data2])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df = df.T
df.to_csv(output_file, encoding=encoding)
