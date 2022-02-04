import os
import pandas as pd
import json

def extract_unique_urls_and_save_them_to_json(pairs_dataset, urls_attribute, url_json_path):
    """
    Extracts unique product URLs from a dataset of product pairs to create a file that can be fed to a scraper
    @param pairs_dataset: dataset of product pairs
    @param urls_attribute: dataset attribute to extract URLs from
    @param url_json_path: path to save the extracted URL file to
    @return:
    """
    urls_array = pd.Series(pairs_dataset[urls_attribute].dropna().unique())
    urls_array = urls_array.apply(
        lambda url: url.strip() if type(url) == str and "http" in url else ""
    )
    array_to_jsonify = []
    for url in urls_array:
        if url != "":
            array_to_jsonify.append({
                'url': url
            })

    with open(url_json_path, 'w') as url_json_file:
        json.dump(array_to_jsonify, url_json_file)
        print("Saving {} URLs to {}".format(len(array_to_jsonify), url_json_path.split("/")[-1]))

def transform_pairs_dataset_to_json_lists_of_urls(pairs_datasets_folder, url_files_folder, pairs_datasets_filenames):
    """
    Transforms datasets of product pairs in a given folder to files of lists of URLs that can be fed to a scraper
    @param pairs_datasets_folder: folder where the datasets are located
    @param url_files_folder: folder where the resulting files of lists of URLs should be located
    @param pairs_datasets_filenames: file names of datasets that should be considered
    @return:
    """
    for pairs_dataset_filename in pairs_datasets_filenames:
        url_json1_path = os.path.join(url_files_folder, pairs_dataset_filename + "_urls1.json")
        url_json2_path = os.path.join(url_files_folder, pairs_dataset_filename + "_urls2.json")
        pairs_dataset_path = os.path.join(pairs_datasets_folder, pairs_dataset_filename)
        pairs_dataset = pd.read_csv(pairs_dataset_path)
        extract_unique_urls_and_save_them_to_json(pairs_dataset, 'source_url', url_json1_path)
        extract_unique_urls_and_save_them_to_json(pairs_dataset, 'target_url', url_json2_path)


initial_files_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
    "annotated_data",
    "initial_files"
)

transform_pairs_dataset_to_json_lists_of_urls(
    initial_files_folder,
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "url_files"),
    os.listdir(initial_files_folder)
)
