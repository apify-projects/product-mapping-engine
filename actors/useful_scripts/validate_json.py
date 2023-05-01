import pandas as pd
import numpy as np
import requests
import json
import sys


inputfile = 'promapcz-train_data.csv' 
data = pd.read_csv(inputfile)


def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
        print(jsonData)
    return True


for c in data['specification1']:
    validateJSON(c)
