import csv
import requests
import json
import time

# switch to load czech or english dictionary to process 
CZ_CONFIG = True
if CZ_CONFIG:
    input_file = 'data/bigger_corpus/cz.csv'
    output_file = 'data/bigger_corpus/cz_cleaned.csv' 
else:
    input_file = 'data/bigger_corpus/en.csv'
    output_file = 'data/bigger_corpus/en_cleaned.csv' 
    
words = []
batch = []
i = 0
BATCH_SIZE = 500


# lemmatize batch
def lemmatize_batch():
    if CZ_CONFIG:
        url = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={batch_str}&output=json&guesser=no&model=czech-morfflex-pdt-161115"
    else:
        url = f"http://lindat.mff.cuni.cz/services/morphodita/api/tag?data={batch_str}&output=json&guesser=no&model=english-morphium-wsj-140407"
    
    r = json.loads(requests.get(url).text)['result']
    for word in r[0]:
        if (not CZ_CONFIG and word['tag']!='UNK') or (CZ_CONFIG and word['tag']!='X@-------------'):
            words.append(word['token'])

# check whether the word in manually created vocabulary from corpus are existing words using MORPHODITA
with open(input_file, encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        batch.append(row[0])
        if len(batch)==BATCH_SIZE:
            print(i) #print progress
            i+=1
            batch_str = ' '.join(b for b in batch)
            try: 
                lemmatize_batch()
                batch = []
                time.sleep(1)
            except Exception as e:
                print(e)
    lemmatize_batch()  
      
with open(output_file, 'w', encoding='utf-8') as f:
    for w in words:
        f.write(f"{w}\n")

