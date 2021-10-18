import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.preprocessing.images.image_preprocessing import crop_images_contour_detection, create_output_directory
from scripts.preprocessing.names.names_preprocessing import detect_ids_brands_and_colors, to_list
from scripts.score_computation.images.compute_hashes_similarity import create_hash_sets, compute_distances
from scripts.score_computation.names.compute_names_similarity import lower_case, remove_colors, compute_tf_idf, \
    compute_name_similarities


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


def preprocess_data_without_saving(dataset_folder='', dataset_dataframe=None, dataset_images_kvs=None):
    """
    For each pair of products compute their image and name similarity without saving anything
    @param dataset_folder: folder containing data to be preprocessed
    @return: preprocessed data
    """
    product_pairs = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(os.path.join(dataset_folder, "product_pairs.csv"))
    name_similarities = create_name_similarities_data(product_pairs)
    image_similarities = [0] * len(product_pairs)
    image_similarities = create_image_similarities_data(len(product_pairs), images_folder=dataset_folder, dataset_images_kvs=dataset_images_kvs)
    name_similarities = pd.DataFrame(name_similarities, columns=list(name_similarities[0].keys()))
    image_similarities = pd.DataFrame(image_similarities, columns=['hash_similarity'])
    return pd.concat([name_similarities, image_similarities, product_pairs["match"]], axis=1)


def create_image_similarities_data(total_count, images_folder='', dataset_images_kvs=None):
    """
    Compute images similarities and create dataset with hash similarity
    @param total_count: number of pairs of products to be compared in source dataset
    @param images_folder: images source folder of pairs of products
    @return: Similarity scores for the images
    """
    if images_folder == '':
        images_folder = '.'

    img_source_dir = os.path.join(images_folder, 'images_cropped')
    img_dir = os.path.join(images_folder, 'images')

    if dataset_images_kvs is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for item in dataset_images_kvs.list_keys()['items']:
            image_name = item['key']
            image_data = dataset_images_kvs.get_record(image_name)['value']
            with open(os.path.join(img_dir, image_name), 'wb') as image_file:
                image_file.write(image_data)

    create_output_directory(img_source_dir)
    crop_images_contour_detection(img_dir, img_source_dir)
    hashes_dir = os.path.join(images_folder, "hashes_cropped.json")
    script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../preprocessing/images/image_hash_creator/main.js")
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
    return image_similarities


def create_name_similarities_data(product_pairs):
    """
    Compute names similarities and create dataset with cos, id, tf idf and brand similarity
    @param product_pairs: product pairs data
    @return: Similarity scores for the names
    """
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
    return name_similarities_list


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
            name_similarities_list = create_name_similarities_data(product_pairs)
            save_to_csv(name_similarities_list, os.path.join(dataset_folder, "name_similarities.csv"))

        if not image_similarities_exist:
            image_similarities = create_image_similarities_data(total_count, dataset_folder)
            save_to_csv(image_similarities, os.path.join(dataset_folder, "image_similarities.csv"))

    name_similarities = pd.read_csv(name_similarities_path)
    image_similarities = pd.read_csv(image_similarities_path)
    return pd.concat([name_similarities, image_similarities, product_pairs["match"]], axis=1)


def analyse_dataset(data):
    """
    Compute percent of non zero values, create correlation matrix and plot data distribution for every feature
    @param data: Dataset to analyse
    @return:
    """
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
    """
    Plot graph of data distribution for every feature
    @param data: Datase to visualize
    @return:
    """
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
        plt.clf()

    subset = data[['words', 'cos', 'match']]
    groups = subset.groupby('match')
    for name, group in groups:
        plt.plot(group['words'], group['cos'], marker="o", linestyle="", label=name)
    plt.title(f'Data distribution according to the words and cos match value')
    plt.xlabel('Words match value')
    plt.ylabel(f'Cos match value')
    plt.legend(['0', '1'])
    plt.show()
