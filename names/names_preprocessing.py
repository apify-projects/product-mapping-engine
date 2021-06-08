import pandas as pd
import re
import json
import requests
import time

colors_file = 'data/vocabularies/colors.txt'
brand_file = 'data/vocabularies/brands.txt'
input_file = 'data/names/names_mall.csv'
input_file2 = 'data/names/names_czc.csv'
output_file = 'data/results/names_detected.txt'
encoding = 'utf-16'
vocabulary_file_cz = 'data/bigger_corpus/cz_cleaned.csv'
vocabulary_file_en = 'data/bigger_corpus/en_cleaned.csv'
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

def load_brands(brand_file):
    brands = []
    file = open(brand_file, 'r', encoding='utf-8')
    lines = file.read().splitlines() 
    for line in lines:
        brands.append(line)
    return brands

def load_vocabulary(vocabulary_file):
    with open(vocabulary_file, encoding='utf-8') as f:
        return [line.rstrip() for line in f]
 
def write_to_file(data):
    with open(output_file, 'w') as f:
        for d in data:
           f.write(' '.join(d))
           f.write('\n')
    

# check whether the word is in Czech or English vocabulary in LINDAT repository
def does_this_word_exist(lemma):
    while True:
        try:
            url_cz = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={lemma}&output=json&guesser=no&model=czech-morfflex-pdt-161115"
            url_en = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={lemma}&output=json&guesser=no&model=english-morphium-wsj-140407"
            
            r_cz = json.loads(requests.get(url_cz).text)['result']
            r_en = json.loads(requests.get(url_en).text)['result']
            
            if not r_cz or not r_en:
                return False
        
            if r_cz[0][0]['tag']=='X@-------------' and r_en[0][0]['tag']=='UNK':
                return False
            return True
        except:
            time.sleep(1)


# check whether a word i s in vocabulary cretaed manually from corpus
def is_word_in_vocabulary(word):
    if word in vocabulary_cz or word in vocabulary_en:
        return True
    return False


# detect ids in names
def id_detection(word):
    # check whether is capslock ad whether is not too short
    word_sub = re.sub(r"[\W_]+", "", word, flags=re.UNICODE)
    if not word_sub.isupper() or len(word_sub)<ID_LEN or word in vocabulary_cz or word in vocabulary_en:
        return word
    
    if word_sub.isnumeric():
        word = word.replace("(", "").replace(")", "")
        return '#id#'+word 
    elif word_sub.isalpha():
        if not is_word_in_vocabulary(word_sub):
            return '#id#'+word  
    else:
        word = word.replace("(", "").replace(")", "")
        return '#id#'+word
    return word

# detect colors in names
def color_detection(word): 
    if word.lower() in colors:
        word = '#col#'+word
    return word

def vocabulary_detection(word):
    if word.lower() in vocabulary_cz or word.lower()  in vocabulary_en:
        return "#voc#"+word
    return word


def brand_detection(word):
    if word.lower() in brands:
        return "#bnd#"+word
    return word

# detect ids and colors in names
def detect_ids_and_colors(data, detect_id, detect_color, detect_brand, compare_words):
    data_list = []
    CNT_VOC = 0
    CNT_LEM = 0
    for name in data:
        word_list = []
        for word in name:
            if detect_color:
                word = color_detection(word)
            if detect_brand:
                 word = brand_detection(word)
            if detect_id:
                word = id_detection(word)
            word_list.append(word)
            
            # compute number of words that are in dictionary and that were found in morphodita
            word = re.sub(r"[\W_]+", "", word, flags=re.UNICODE).lower()
           
            if compare_words:
                rec_lem = False
                rec_voc = False
                if is_word_in_vocabulary(word):
                    rec_voc = True
                    CNT_VOC+=1
                if does_this_word_exist(word):
                    rec_lem = True
                    CNT_LEM+=1
                if (rec_voc and not rec_lem) or (not rec_voc and  rec_lem):
                    print(word)
            
        data_list.append(word_list)
        #break
    return data_list, CNT_VOC, CNT_LEM


# convert list of names to list of list of words
def to_list(data):
    data_list = []
    for d in data:
        words = [w for w in d.split(" ")]
        data_list.append(words)
    return data_list

# compute similarity of two names 
def compute_word_similarity(name1, name2):
    return 0


colors = load_colors(colors_file)
brands = load_brands(brand_file)
vocabulary_cz = load_vocabulary(vocabulary_file_cz)
vocabulary_en = load_vocabulary(vocabulary_file_en)
 
df= pd.read_csv(input_file, encoding=encoding)
df.dropna(inplace=True)
data = to_list(df.values[:,1])

compare_words=False
data, CNT_VOC, CNT_LEM = detect_ids_and_colors(data, detect_id=True, detect_color=True, detect_brand=True, compare_words=compare_words)

if compare_words:
    print('Number of words in names that were in manually created vocabulary: ' + str(CNT_VOC))
    print('Number of words in names that were recognised in Morphoditta: ' + str(CNT_LEM))

write_to_file(data)
