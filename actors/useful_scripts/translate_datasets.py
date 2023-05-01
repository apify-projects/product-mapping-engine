import pandas as pd
import numpy as np
import requests
import json
import sys


inputfile = sys.argv[1]
lan = sys.argv[2]
data = pd.read_csv(inputfile)

def translate_cz(text):
    url = "https://lindat.mff.cuni.cz/services/translation/api/v2/languages/?src=cs&tgt=en" # Set destination URL here
    header = {"Content-type": "application/x-www-form-urlencoded","Accept": "text/plain", 'charset':'utf-8'}
    data = { "input_text":text }
    request = requests.post(url, headers = header, data=data)
    request.encoding = 'utf-8'
    r = request.text.rstrip()
    return r


def translate_en(text):
    url = "https://lindat.mff.cuni.cz/services/transformer/api/v2/models/en-cs?src=en&tgt=cs" # Set destination URL here
    header = {"Content-type": "application/x-www-form-urlencoded","Accept": "text/plain", 'charset':'utf-8'}
    data = { "input_text":text }
    request = requests.post(url, headers = header, data=data)
    request.encoding = 'utf-8'
    r = request.text.rstrip()
    return r

def translate(text):
    if lan=='cz':
        return translate_cz(text)
    else:
        return translate_en(text)
spec_cols = ['specification1', 'specification2']

for c in spec_cols:
    spec_parsed = []
    for specs in data[c].values:
        s = json.loads(specs)
        spec = []
        for val in s:
            val['key'] = translate(val['key'])
            val['value'] = translate(val['value'])
            spec.append(val)
        spec_parsed.append(spec)
    data[c] = spec_parsed
    print('DONE ' + c)


cols = ['name1', 'short_description1', 'long_description1', 'name2', 'short_description2',
       'long_description2']

col = cols[0]

for col in cols:
    translated = []
    for text in data[[col]].values:
        translated_text = translate(text)
        translated.append(translated_text)
    data[col] = translated
    print('DONE ' + col)

data.to_csv(inputfile[:-4] + '_translated.csv', index=False)