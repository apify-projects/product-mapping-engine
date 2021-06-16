import re
import csv
import requests
import json

CORPUS_FILE = 'data/bigger_corpus/en-cs.txt'
OUTPUT_CZ = 'data/bigger_corpus/cz.csv'
OUTPUT_EN = 'data/bigger_corpus/en.csv'

cz = {}
en = {}

# load corpus file and create dictionary of unique words
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line_en = line.split('\t')[0]
        line_cz = line.split('\t')[1]
        
        for word in line_cz.split(' '):
            word = word.lower()
            if word.isalpha():
                if not word in cz:
                    cz[word]=0
                cz[word]+=1
                
        for word in line_en.split(' '):
            word = word.lower()
            if word.isalpha():
                if not word in en:
                    en[word]=0
                en[word]+=1


# sort dictionaries
cz = dict(sorted(cz.items(), key=lambda item: item[1], reverse=True))
en = dict(sorted(en.items(), key=lambda item: item[1], reverse=True))

# save dictionaries
with open(OUTPUT_CZ, 'w', encoding='utf-8') as f:
    for key in cz.keys():
        f.write("%s,%s\n"%(key,cz[key]))
    
with open(OUTPUT_EN, 'w', encoding='utf-8') as f:
    for key in en.keys():
        f.write("%s,%s\n"%(key,en[key]))