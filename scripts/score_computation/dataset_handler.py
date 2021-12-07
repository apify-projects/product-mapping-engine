import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.preprocessing.images.image_preprocessing import create_output_directory
from scripts.preprocessing.images.image_preprocessing import crop_images_contour_detection
from scripts.score_computation.images.compute_hashes_similarity import create_hash_sets, compute_distances
from scripts.score_computation.texts.compute_specifications_similarity import \
    preprocess_specifications_and_compute_similarity
from scripts.score_computation.texts.compute_texts_similarity import compute_similarity_of_texts


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


def preprocess_data_without_saving(
        dataset_folder='',
        dataset_dataframe=None,
        dataset_images_kvs1=None,
        dataset_images_kvs2=None
):
    """
    For each pair of products compute their image and name similarity without saving anything
    @param dataset_folder: folder containing data to be preprocessed
    @param dataset_dataframe: dataframe of pairs to be compared
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @return: preprocessed data
    """
    product_pairs = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
        os.path.join(dataset_folder, "product_pairs.csv"))
    name_similarities = create_text_similarities_data(product_pairs)

    image_similarities = [0] * len(product_pairs)
    image_similarities = create_image_similarities_data(
        product_pairs[['id1', 'image1', 'id2', 'image2']].to_dict(orient='records'),
        dataset_folder=dataset_folder,
        dataset_images_kvs1=dataset_images_kvs1,
        dataset_images_kvs2=dataset_images_kvs2
    )
    name_similarities = pd.DataFrame(name_similarities)
    image_similarities = pd.DataFrame(image_similarities, columns=['hash_similarity'])
    dataframes_to_concat = [name_similarities, image_similarities]
    if 'match' in product_pairs.columns:
        dataframes_to_concat.append(product_pairs['match'])

    return pd.concat(dataframes_to_concat, axis=1)


def download_images_from_kvs(
        img_dir,
        dataset_images_kvs,
        prefix
):
    """
    Downloads images from the given key-value-store and saves them into the specified folder, prefixing their name with
    the provided prefix.
    @param img_dir: folder to save the downloaded images to
    @param dataset_images_kvs: key-value-store containing the images
    @param prefix: prefix identifying the dataset the images come from, will be used in their file names
    @return: Similarity scores for the images
    """
    if dataset_images_kvs is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for item in dataset_images_kvs.list_keys()['items']:
            image_name = item['key']
            image_data = dataset_images_kvs.get_record(image_name)['value']
            with open(os.path.join(img_dir, prefix + '_' + image_name), 'wb') as image_file:
                image_file.write(image_data)


def create_image_similarities_data(
        pair_ids_and_counts_dataframe,
        dataset_folder='',
        dataset_images_kvs1=None,
        dataset_images_kvs2=None
):
    """
    Compute images similarities and create dataset with hash similarity
    @param pair_ids_and_counts_dataframe: dataframe containing ids and image counts for the pairs of products
    @param dataset_folder: folder to be used as dataset root, determining where the images will be stored
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @return: Similarity scores for the images
    """
    if dataset_folder == '':
        dataset_folder = '.'

    img_source_dir = os.path.join(dataset_folder, 'images_cropped')
    img_dir = os.path.join(dataset_folder, 'images')

    dataset_prefixes = ['dataset1', 'dataset2']
    download_images_from_kvs(img_dir, dataset_images_kvs1, dataset_prefixes[0])
    download_images_from_kvs(img_dir, dataset_images_kvs2, dataset_prefixes[1])

    create_output_directory(img_source_dir)
    crop_images_contour_detection(img_dir, img_source_dir)
    hashes_dir = os.path.join(dataset_folder, "hashes_cropped.json")
    script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "../preprocessing/images/image_hash_creator/main.js")
    subprocess.call(f'node {script_dir} {img_source_dir} {hashes_dir}', shell=True)
    data = load_and_parse_data(hashes_dir)
    hashes, names = create_hash_sets(data, pair_ids_and_counts_dataframe, dataset_prefixes)
    imaged_pairs_similarities = compute_distances(
        hashes,
        names,
        metric='binary',
        filter_dist=True,
        thresh=0.9
    )

    # Correctly order the similarities and fill in 0 similarities for pairs that don't have images
    image_similarities = []
    for x in range(len(pair_ids_and_counts_dataframe)):
        image_similarities.append(0)
    for index, similarity in imaged_pairs_similarities:
        image_similarities[index] = similarity
    return image_similarities


def create_text_similarities_data(product_pairs):
    """
    Compute all the text-based similarities for the product pairs
    @param product_pairs: product pairs data
    @return: Similarity scores for the product pairs
    """
    columns = ['name', 'short_description', 'long_description', 'specification']
    similarity_names = ['id', 'brand', 'words', 'cos', 'descriptives', 'units']
    df_all_similarities = create_emtpy_dataframe(columns, similarity_names)

    # all text types preprocessed as texts
    for column in columns:
        column1 = f'{column}1'
        column2 = f'{column}2'
        if column1 in product_pairs and column2 in product_pairs:
            columns_similarity = compute_similarity_of_texts(
                product_pairs[column1],
                product_pairs[column2],
                id_detection=True,
                color_detection=True,
                brand_detection=True,
                units_detection=True
            )
            columns_similarity = pd.DataFrame(columns_similarity)
            for similarity_name, similarity_value in columns_similarity.items():
                df_all_similarities[f'{column}_{similarity_name}'] = similarity_value
        else:
            for similarity_name in similarity_names:
                df_all_similarities[f'{column}_{similarity_name}'] = 0

    # specification with units and values preprocessed as specification
    specification_column_name1 = 'specification1'
    specification_column_name2 = 'specification2'
    df_all_similarities['specification_key_matches'] = 0
    df_all_similarities['specification_key_value_matches'] = 0
    if specification_column_name1 in product_pairs and specification_column_name2 in product_pairs:
        specification_similarity = preprocess_specifications_and_compute_similarity(specification_column_name1, specification_column_name2, separator=': ')
        specification_similarity = pd.DataFrame(specification_similarity)
        df_all_similarities['specification_key_matches'] = specification_similarity.iloc[:, [0]].values
        df_all_similarities['specification_key_value_matches'] = specification_similarity.iloc[:, [1]].values
    return df_all_similarities


def create_emtpy_dataframe(text_types, similarity_names):
    """
    Create empty dataframe for text similarity results
    @param text_types: names of compared types of the text
    @param similarity_names: names of measured similarities
    @return: empty dataframe with suitable column names for all measured text similarities
    """
    df_column_names = []
    for text_type in text_types:
        for similarity_name in similarity_names:
            df_column_names.append(f'{text_type}_{similarity_name}')
    df_all_similarities = pd.DataFrame(columns=df_column_names)
    return df_all_similarities


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
            name_similarities_list = create_text_similarities_data(product_pairs)
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
