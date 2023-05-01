import pandas as pd
import numpy as np
import requests
import json
import sys


inputfile = sys.argv[1]
lan = sys.argv[2]
data = pd.read_csv(inputfile)


import re


def tokenize(text):
    rgx = re.compile("\w[\"-'×.,%]?\w*")
    word_list = rgx.findall(text)
    return word_list

def remove_useless_spaces_and_characters(text):
    text = re.sub(r'(?<=\d) - (?=\d)', r'-', text)
    text = re.sub(r'(?<=\d),(?=\d)', r'', text)
    text = re.sub(r'(?<=\d)"', r' inch', text)
    text = re.sub(r'(?<=\d)x(?=\d)', r'×', text)
    # text = re.sub(r'(?<=\d)x', r'', text)
    text = text.replace(' × ', '×')
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text

def split_units_and_values(word_list):
    word_list_split = []
    for word in word_list:
        # if word in the form: number+string (eg:12kb, 1.2kb)
        if re.match('^([0-9]*[.])?[0-9]+[a-zA-Z]+$', word) is not None:
            word_list_split.append(re.split('[a-zA-Z]+$', word)[0])
            split = re.split('^([0-9]*[.])?[0-9]+', word)
            word_list_split.append(split[len(split) - 1])
        # if word in the form: number+nonstring_unit (eg: 50°C,70.5°F, 14")
        elif re.match('^([0-9]*[.])?[0-9]+[{°C}{°F}°%£€$Ω\"\']+$', word) is not None:
            word_list_split.append(re.split('[{°C}{°F}°%£€$Ω\"\']+$', word)[0])
            split = re.split('^([0-9]*[.])?[0-9]+', word)
            word_list_split.append(split[len(split) - 1])
        # if word in the form: number-number+string (eg: 10-15h)
        elif re.match('^([0-9]*[.])?[0-9]+-([0-9]*[.])?[0-9]+[a-zA-Z]+$', word) is not None:
            word_list_split.append(re.split('[a-zA-Z]+$', word)[0])
            split = re.split('^([0-9]*[.])?[0-9]+-([0-9]*[.])?[0-9]+', word)
            word_list_split.append(split[len(split) - 1])
        # if word in the form: number×number×number+string (eg: 10×10×10cm)
        elif re.match('^([0-9]+×[0-9]+(×[0-9])*)[a-zA-Z]+$', word) is not None:
            word_list_split.append(re.split('[a-zA-Z]+$', word)[0])
            split = re.split('^([0-9]+×[0-9]+(×[0-9])*)', word)
            word_list_split.append(split[len(split) - 1])
        else:
            word_list_split.append(word)
    return word_list_split


def lower_case(word_list):
    lowercase_word_list = []
    for word in word_list:
        lowercase_word_list.append(word.lower())
    return lowercase_word_list


def prepro_text(data):
    new_data = []    
    for text in data:
        text = remove_useless_spaces_and_characters(text)
        word_list = tokenize(text)
        word_list = split_units_and_values(word_list)
        word_list = lower_case(word_list)
    new_data.append(word_list)
    return new_data



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


cols = ['name1', 'short_description1', 'long_description1', 'name2', 'short_description2','long_description2']

col = cols[0]
data = data.head(5)
for col in cols:
    translated = []
    for text in data[[col]].values:
        text = prepro_text(text)
        translated_text = translate(' '.join(text[0]))
        translated.append(translated_text)
        print(translated_text)
    data[col] = translated
    print('DONE ' + col)

data.to_csv(inputfile[:-4] + '_translated.csv', index=False)