import pandas as pd
import re


input_file = 'data/names_mall.csv'
input_file2 = 'data/names_czc.csv'
output_file = 'data/tf_idf.csv'
encoding= 'utf-16'
colors_file = 'data/colors.txt'
ID_LEN = 5


def load_colors(colors_file):
    colors = []
    file = open(colors_file, 'r', encoding='utf-8')
    lines = file.read().splitlines() 
    for line in lines:
        colors.append(line)
        if line[len(line)-1:] == 'á':
            colors.append(line[:-1]+'ý')
            colors.append(line[:-1]+'é')
    return colors

# detect ids in names
def detect_ids(data):
    data_list = []
    for name in data:
        word_list = []
        for word in name:
            word_sub = re.sub(r"[\W_]+", "", word, flags=re.UNICODE).lower()
            if word_sub.isalnum() and not word_sub.isnumeric() and not word_sub.isalpha() and len(word_sub)>ID_LEN:
                word = word.replace("(", "").replace(")", "")
                word = '#id#'+word
            word_list.append(word)
        data_list.append(word_list)
    return data_list

# detect colors in names
def detect_colors(data, colors_file):
    colors = load_colors(colors_file)
    print(colors)
    data_list = []
    for name in data:
        word_list= []
        for word in name:
            word_sub = re.sub(r"[\W_]+", "", word, flags=re.UNICODE).lower()
            if word_sub in colors:
                word = '#col#'+word
                #print(word)
            word_list.append(word)
        data_list.append(word_list)
    return data_list

# convert list of names to list of list of words
def to_list(data):
    data_list = []
    for d in data:
        words = [w for w in d.split(" ")]
        data_list.append(words)
    return data_list


df= pd.read_csv(input_file, encoding=encoding)
df.dropna(inplace=True)
data = to_list(df.values[:,1])
data = detect_ids(data)
data = detect_colors(data, colors_file)
print(data)


