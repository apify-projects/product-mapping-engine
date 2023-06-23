import py_stringmatching as sm
import pandas as pd


SIMILARITY = 'cos' # cos tfidf
TOKENIZER = 'alnum' # alnum whitespace
COLUMNS = ['name', 'short_description', 'long_description'] #, 'specification']

# load data
filename = 'amazon_google'
data_type = '-test_data' #-train_data test_data
data = pd.read_csv(filename+data_type+'.csv')
similarities = pd.DataFrame({'name_similarity': pd.Series(dtype='float'),
                             'short_description_similarity': pd.Series(dtype='float'),
                             'long_description_similarity': pd.Series(dtype='float'),
                             'specification_similarity': pd.Series(dtype='float')})

# create tokenizer
if SIMILARITY == 'cos':
    tokenizer = sm.AlphanumericTokenizer(return_set=True)
else:
    tokenizer =  sm.WhitespaceTokenizer(return_set=True)

# similarity measure object
if SIMILARITY == 'cos':
    similarity_object = sm.Cosine()
else:
    similarity_object = sm.TfIdf()
  

# compute similarities
for column in COLUMNS:
    scores = []
    col = [col for col in data.columns if column in col]
    column1 = data[col[0]].values
    column2 = data[col[1]].values
    
    for col1, col2 in zip(column1, column2):
        if type(col1) != str:
            col1=''
        if type(col2) != str:
            col2=''
        sim = similarity_object.get_sim_score(tokenizer.tokenize(col1), tokenizer.tokenize(col2))
        scores.append(sim)
    similarities[column+'_similarity'] = scores
similarities['match'] = data['match']
similarities['id1'] = data['id1']
similarities['id2'] = data['id2']
similarities.to_csv(filename+ '_'+ SIMILARITY + '_'+TOKENIZER +data_type+'_similarities.csv', index=False)