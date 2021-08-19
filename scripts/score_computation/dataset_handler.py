import json
import os
import subprocess
import sys

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.preprocessing.images.image_preprocessing import crop_images_contour_detection, create_output_directory
from scripts.preprocessing.names.names_preprocessing import detect_ids_brands_and_colors, to_list
from scripts.score_computation.images.compute_hashes_similarity import create_hash_sets, compute_distances
from scripts.score_computation.names.compute_names_similarity import lower_case, remove_colors, compute_tf_idf

current_directory = os.path.dirname(os.path.realpath(__file__))
# Adding the higher level directory (scripts/) to sys.path so that we can import from the other folders
sys.path.append(os.path.join(current_directory, ".."))


def load_and_parse_data(input_file):
    """
    Load input file and split name and hash into dictionary
    @param input_file: file with hashes and names
    @return: dictionary with name and has value of the image
    """
    data = {}
    with open(input_file) as json_file:
        loaded_data = json.load(json_file)

    for d in loaded_data:
        dsplit = d.split(';')
        data[dsplit[0]] = dsplit[1]
    return data


def save_to_csv(data_list, output_file, column_names=None):
    """
    Save data list to csv format
    @param data_list: data as list
    @param output_file: name of the output file
    @param column_names: names of columns
    @return:
    """
    data_dataframe = pd.DataFrame(data_list, columns=column_names)
    data_dataframe.fillna(0, inplace=True)
    data_dataframe.to_csv(output_file, index=False)


def load_file(name_file):
    """
    Load file with product names
    @param name_file: name of the input file
    @return: list of product names
    """
    names = []

    file = open(name_file, 'r', encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        names.append(line)
    return names


def preprocess_data(dataset_folder):
    """
    For each pair of products compute their image and name similarity
    @param dataset_folder: folder containing data to be preprocessed
    @return: preprocessed data
    """
    name_similarities_path = os.path.join(dataset_folder, "name_similarities.csv")
    image_similarities_path = os.path.join(dataset_folder, "image_similarities.csv")

    name_similarities_exist = os.path.isfile(name_similarities_path)
    image_similarities_exist = os.path.isfile(image_similarities_path)

    product_pairs = pd.read_csv(os.path.join(dataset_folder, "product_pairs.csv"))
    total_count = 0
    imaged_count = 0
    for pair in product_pairs.itertuples():
        total_count += 1
        if pair.image1 > 0 and pair.image2 > 0:
            imaged_count += 1

    if not name_similarities_exist or not image_similarities_exist:
        if not name_similarities_exist:
            names = []
            names_by_id = {}
            for pair in product_pairs.itertuples():
                names_by_id[pair.id1] = len(names)
                names.append(pair.name1)
                names_by_id[pair.id2] = len(names)
                names.append(pair.name2)

            names = to_list(names)
            names, _, _ = detect_ids_brands_and_colors(names, compare_words=False)
            names = [' '.join(name) for name in names]
            names = lower_case(names)
            names = remove_colors(names)
            tf_idfs = compute_tf_idf(names)

            name_similarities_list = []
            for pair in product_pairs.itertuples():
                name1_index = names_by_id[pair.id1]
                name2_index = names_by_id[pair.id2]
                name_similarities = compute_name_similarities(
                    names[name1_index],
                    names[name2_index],
                    name1_index,
                    name2_index,
                    tf_idfs
                )
                name_similarities_list.append(name_similarities)

            save_to_csv(name_similarities_list, )

        if not image_similarities_exist:
            img_source_dir = os.path.join(dataset_folder, 'images_cropped')
            img_dir = os.path.join(dataset_folder, 'images')
            create_output_directory(img_source_dir)
            crop_images_contour_detection(img_dir, img_source_dir)
            hashes_dir = os.path.join(dataset_folder, "hashes_cropped.json")
            script_dir = os.path.join(current_directory, "preprocessing/images/image_hash_creator/main.js")
            subprocess.call(f'node {script_dir} {img_source_dir} {hashes_dir}', shell=True)

            data = load_and_parse_data(hashes_dir)
            hashes, names = create_hash_sets(data)
            imaged_pairs_similarities = compute_distances(hashes, names, metric='binary',
                                                          filter_dist=True,
                                                          thresh=0.9)

            # Correctly order the similarities and fill in 0 similarities for pairs that don't have images
            image_similarities = []
            for x in range(total_count):
                image_similarities.append(0)
            for index, similarity in imaged_pairs_similarities:
                image_similarities[index] = similarity

            save_to_csv(image_similarities, )

    name_similarities = pd.read_csv(name_similarities_path)
    image_similarities = pd.read_csv(image_similarities_path)
    return pd.concat([name_similarities, image_similarities, product_pairs["match"]], axis=1)


def analyse_dataset(data):
    print('\n\nDataset analysis')
    print('----------------------------')
    data_size = data.shape[0]
    for column in data:
        values = data[column].to_numpy()
        len_nonzero = np.count_nonzero(values)
        ratio_of_nonzero_ids = len_nonzero / data_size
        print(f'Pairs with nonzero {column} match: {round(100 * ratio_of_nonzero_ids, 2)}%')


    corr = data.iloc[:, :-1].corr()
    print('\n\nCorrelation matrix of features')
    print(corr.values)
    print('----------------------------')

    plot_features(data)

def plot_features(data):
    for column in data.iloc[:, :-1]:
        subset = data[[column, 'match']]
        groups = subset.groupby('match')
        for name, group in groups:
            plt.plot(np.arange(0, len(group)), group[column], marker="o", linestyle="", label=name)
        plt.title(f'Data distribution according to the {column} match value')
        plt.xlabel('Data pair')
        plt.ylabel(f'{column} match value')
        plt.legend(['0', '1'])
        plt.show()

    subset = data[['words', 'cos', 'match']]
    groups = subset.groupby('match')
    for name, group in groups:
        plt.plot(group['words'], group['cos'], marker="o", linestyle="", label=name)
    plt.title(f'Data distribution according to the words and cos match value')
    plt.xlabel('Words match value')
    plt.ylabel(f'Cos match value')
    plt.legend(['0', '1'])
    plt.show()
