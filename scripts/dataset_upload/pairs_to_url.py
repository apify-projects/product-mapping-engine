import os
import pandas as pd
import json


def transform_pairs_dataset_to_json_lists_of_urls(pairs_dataset_path, url_json1_path, url_json2_path):
    pairs_dataset = pd.read_csv(pairs_dataset_path)
    extract_unique_urls_and_save_them_to_json(pairs_dataset, 'itemUrl', url_json1_path)
    extract_unique_urls_and_save_them_to_json(pairs_dataset, 'matchUrl', url_json2_path)

def extract_unique_urls_and_save_them_to_json(pairs_dataset, urls_attribute, url_json_path):
    urls_array = pd.Series(pairs_dataset[urls_attribute].dropna().unique())
    urls_array = urls_array.apply(
        lambda url: url if type(url) == str else url.strip()
    )
    array_to_jsonify = []
    for url in urls_array:
        array_to_jsonify.append({
            'url': url
        })

    with open(url_json_path, 'w') as url_json_file:
        json.dump(array_to_jsonify, url_json_file)

# Test TODO delete
transform_pairs_dataset_to_json_lists_of_urls(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "old_data/jirka_all_categories.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "old_data/urls1.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "old_data/urls2.json")
)
