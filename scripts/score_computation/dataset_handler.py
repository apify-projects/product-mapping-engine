import base64
import copy
import json
import os
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .images.compute_hashes_similarity import create_hash_sets, compute_distances
from .texts.compute_specifications_similarity import \
    compute_similarity_of_specifications
from .texts.compute_texts_similarity import compute_similarity_of_texts
from ..preprocessing.images.image_preprocessing import crop_images_contour_detection, create_output_directory, \
    compute_image_hashes
from ..preprocessing.texts.keywords_detection import detect_ids_brands_colors_and_units
from ..preprocessing.texts.specification_preprocessing import convert_specifications_to_texts, \
    parse_specifications, preprocess_specifications
from ..preprocessing.texts.text_preprocessing import preprocess_text

COLUMNS = ['name', 'short_description', 'long_description', 'specification_text', 'all_texts']
SIMILARITY_NAMES = ['id', 'brand', 'words', 'cos', 'descriptives', 'units']


def load_and_parse_data(input_files):
    """
    Load input file and split name and hash into dictionary
    @param input_files: files with hashes and names
    @return: dictionary with name and has value of the image
    """
    data = {}

    for input_file in input_files:
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
    data_dataframe.fillna(0, inplCOLUMNSace=True)
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


def preprocess_textual_data(dataset,
                            id_detection=True,
                            color_detection=True,
                            brand_detection=True,
                            units_detection=True):
    """
    Preprocessing of all textual data in dataset column by column
    @param dataset: dataset to be preprocessed
    @param id_detection: True if id should be detected
    @param color_detection: True if color should be detected
    @param brand_detection: True if brand should be detected
    @param units_detection: True if units should be detected
    @return preprocessed dataset
    """
    dataset['price'] = pd.to_numeric(dataset['price'])
    dataset = dataset.sort_values(by=['price'])
    dataset = parse_specifications_and_create_copies(dataset, 'specification')
    dataset = add_all_texts_columns(dataset)
    for column in COLUMNS:
        if column in dataset:
            dataset[column] = preprocess_text(dataset[column].values)
            dataset[column] = detect_ids_brands_colors_and_units(
                dataset[column],
                id_detection,
                color_detection,
                brand_detection,
                units_detection
            )
    if 'specification' in dataset.columns:
        dataset['specification'] = preprocess_specifications(dataset['specification'])
    return dataset


def preprocess_data_without_saving(dataset1, dataset2, tf_idfs, descriptive_words, pool, num_cpu,
                                   dataset_folder='',
                                   dataset_dataframe=None,
                                   dataset_images_kvs1=None,
                                   dataset_images_kvs2=None
                                   ):
    """
    For each pair of products compute their image and name similarity without saving anything
    @param dataset1: first dataframe with products
    @param dataset2: second dataframe with products
    @param tf_idfs: dictionary of tf.idfs for each text column in products
    @param descriptive_words:  dictionary of descriptive words for each text column in products
    @param pool: parallelising object
    @param num_cpu: number of processes
    @param dataset_folder: folder containing data to be preprocessed
    @param dataset_dataframe: dataframe of pairs to be compared
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @return: preprocessed data
    """
    product_pairs_idx = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
        os.path.join(dataset_folder, "product_pairs.csv"))

    print("Text similarities computation started")
    name_similarities = create_text_similarities_data(dataset1, dataset2, product_pairs_idx, tf_idfs, descriptive_words,
                                                      pool, num_cpu)

    print("Text similarities computation finished")

    pairs = []
    for source_id, target_ids in product_pairs_idx.items():
        for target_id in target_ids:
            pairs.append({
                'id1': dataset1['id'][source_id],
                'image1': dataset1['image'][source_id],
                'id2': dataset2['id'][target_id],
                'image2': dataset2['image'][target_id],
            })

    image_similarities = create_image_similarities_data(pool, num_cpu,
        pairs,
        dataset_folder=dataset_folder,
        dataset_images_kvs1=dataset_images_kvs1,
        dataset_images_kvs2=dataset_images_kvs2
    )
    name_similarities = pd.DataFrame(name_similarities)
    image_similarities = pd.DataFrame(image_similarities, columns=['hash_similarity'])
    dataframes_to_concat = [name_similarities, image_similarities]
    # if 'match' in product_pairs.columns:
    #    dataframes_to_concat.append(product_pairs['match'])

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

        for chunk_record in dataset_images_kvs.list_keys()['items']:
            chunk = json.loads(dataset_images_kvs.get_record(chunk_record['key'])['value'])
            for image_name, image_data in chunk.items():
                with open(os.path.join(img_dir, prefix + '_' + image_name), 'wb') as image_file:
                    image_file.write(base64.b64decode(bytes(image_data, 'utf-8')))


def multi_run_compute_image_hashes(args):
    """
    Wrapper for passing more arguments to create_hash_sets in parallel way
    @param args: Arguments of the function
    @return: call the create_hash_sets in parallel way
    """
    return compute_image_hashes(*args)

def multi_run_create_images_hash_wrapper(args):
    """
    Wrapper for passing more arguments to create_hash_sets in parallel way
    @param args: Arguments of the function
    @return: call the create_hash_sets in parallel way
    """
    return create_hash_sets(*args)

def multi_run_compute_distances_wrapper(args):
    """
    Wrapper for passing more arguments to compute_distances in parallel way
    @param args: Arguments of the function
    @return: call the compute_distances in parallel way
    """
    return compute_distances(*args)


def create_image_similarities_data(
        pool,
        num_cpu,
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

    img_dir = os.path.join(dataset_folder, 'images')

    print("Image download started")

    dataset_prefixes = ['dataset1', 'dataset2']

    download_images_from_kvs(img_dir, dataset_images_kvs1, dataset_prefixes[0])
    download_images_from_kvs(img_dir, dataset_images_kvs2, dataset_prefixes[1])

    print("Image download finished")

    print("Image preprocessing started")
    script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "../preprocessing/images/image_hash_creator/main.js")
    image_filenames = os.listdir(img_dir)
    image_filenames_chunks = np.array_split(image_filenames, num_cpu)
    hash_files = pool.map(
        multi_run_compute_image_hashes,
        [
            (index, dataset_folder, img_dir, image_filenames_chunk, script_dir)
            for index, image_filenames_chunk in enumerate(image_filenames_chunks)
        ]
    )

    data = load_and_parse_data(hash_files)

    pair_ids_and_counts_dataframe_parts = np.array_split(pair_ids_and_counts_dataframe, num_cpu)
    hashes_names_list = pool.map(multi_run_create_images_hash_wrapper,
                                        [(data, pair_ids_and_counts_dataframe_part, dataset_prefixes) for pair_ids_and_counts_dataframe_part in pair_ids_and_counts_dataframe_parts])
    print("Image preprocessing finished")

    print("Image similarities computation started")
    imaged_pairs_similarities_list = pool.map(multi_run_compute_distances_wrapper,
                                              [(item[0], item[1], 'binary', True, 0.9) for item in hashes_names_list])
    imaged_pairs_similarities = [item for sublist in imaged_pairs_similarities_list for item in sublist]
    print("Image similarities computation finished")


    # Correctly order the similarities and fill in 0 similarities for pairs that don't have images
    image_similarities = []
    for x in range(len(pair_ids_and_counts_dataframe)):
        image_similarities.append(0)
    for index, similarity in imaged_pairs_similarities:
        image_similarities[index] = similarity
    return image_similarities





def chunks(dictionary, dict_num):
    """
    Split dictionary into several same parts
    @param dictionary: dictionary to be split
    @param dict_num: number or parts
    @return: list of dict_num dictionaries of the same size
    """
    it = iter(dictionary)
    for i in range(0, len(dictionary), dict_num):
        yield {k: dictionary[k] for k in islice(it, dict_num)}


def multi_run_text_similarities_wrapper(args):
    """
    Wrapper for passing more arguments to compute_similarity_of_texts in parallel way
    @param args: Arguments of the function
    @return: call the compute_similarity_of_texts in parallel way
    """
    return compute_text_similarities_parallely(*args)


def create_text_similarities_data(dataset1, dataset2, product_pairs_idx, tf_idfs, descriptive_words, pool, num_cpu):
    """
    Compute all the text-based similarities for the product pairs
    @param dataset1: first dataset of all products
    @param dataset2: second dataset of all products
    @param product_pairs_idx: dict with indices of filtered possible matching pairs
    @param tf_idfs: tf.idfs of all words from both datasets
    @param descriptive_words: decsriptive words from both datasets
    @param pool: parallelising object
    @param num_cpu: number of processes
    @return: Similarity scores for the product pairs
    """
    df_all_similarities_list = pool.map(multi_run_text_similarities_wrapper,
                                        [(dataset1, dataset2, descriptive_words,
                                          product_pairs_idx_part, tf_idfs) for product_pairs_idx_part in
                                         chunks(product_pairs_idx, round(len(product_pairs_idx) / num_cpu))])
    df_all_similarities = pd.concat(df_all_similarities_list, ignore_index=True)

    # for each column compute the similarity of product pairs selected after filtering

    # specification comparison with units and values preprocessed as specification
    df_all_similarities['specification_key_matches'] = 0
    df_all_similarities['specification_key_value_matches'] = 0

    if 'specification' in dataset1.columns and 'specification' in dataset2.columns:
        specification_similarity = compute_similarity_of_specifications(dataset1['specification'],
                                                                        dataset2['specification'], product_pairs_idx)
        specification_similarity = pd.DataFrame(specification_similarity)
        df_all_similarities['specification_key_matches'] = specification_similarity['matching_keys']
        df_all_similarities['specification_key_value_matches'] = specification_similarity['matching_keys_values']

    return df_all_similarities


def compute_text_similarities_parallely(dataset1, dataset2, descriptive_words,
                                        product_pairs_idx, tf_idfs):
    """
    Compute similarity score of each pair in both datasets parallelly for each column
    @param dataset1: first list of texts where each is list of words
    @param dataset2: second list of texts where each is list of words
    @param descriptive_words: decsriptive words from both datasets
    @param product_pairs_idx: dict with indices of filtered possible matching pairs
    @param tf_idfs: tf.idfs of all words from both datasets
    @return: dataset of pair similarity scores
    """
    df_all_similarities = create_empty_dataframe(product_pairs_idx)
    for column in COLUMNS:
        if column in dataset1 and column in dataset2:
            columns_similarity = compute_similarity_of_texts(dataset1[column], dataset2[column], product_pairs_idx,
                                                             tf_idfs[column],
                                                             descriptive_words[column]
                                                             )

            columns_similarity = pd.DataFrame(columns_similarity)
            for similarity_name, similarity_value in columns_similarity.items():
                df_all_similarities[f'{column}_{similarity_name}'] = similarity_value
        else:
            for similarity_name in SIMILARITY_NAMES:
                df_all_similarities[f'{column}_{similarity_name}'] = 0
    return df_all_similarities


def parse_specifications_and_create_copies(dataset, specification_name):
    """
    Parse specification from json to dict and create copies of them converted to classical text
    @param dataset: dataframe with products
    @param specification_name: name of the specification column
    @return: dataframe with products with parsed specifications and new columns of specifications converted to text
    """
    if specification_name in dataset.columns:
        dataset[specification_name] = parse_specifications(dataset[specification_name])
        specification_name_text = f'{specification_name}_text'
        if specification_name[-1] in ['1', '2']:
            specification_name_text = f'{specification_name[:-1]}_text{specification_name[-1]}'

        dataset[specification_name_text] = convert_specifications_to_texts(
            copy.deepcopy(dataset[specification_name].values))
    return dataset


def add_all_texts_columns(dataset):
    """
    Add to the dataset column containing all joined texts columns
    @param dataset: dataframe in which to join text columns
    @return: dataframe with additional two columns containing all texts for each product
    """
    columns = list(dataset.columns)
    columns_to_remove = ['match', 'image', 'price', 'url', 'index', 'specification']
    for col in columns_to_remove:
        if col in columns:
            columns.remove(col)
    dataset['all_texts'] = dataset[columns].agg(','.join, axis=1)
    return dataset


def add_all_texts_columns_pairs(dataset):
    """
    Add to the dataset column containing all joined texts columns
    @param dataset: dataframe in which to join text columns
    @return: dataframe with additional two columns containing all texts for each product
    """
    columns = list(dataset.columns)
    columns_to_remove = ['match', 'image1', 'image2', 'price1', 'price2', 'url1', 'url2', 'index', 'specification1',
                         'specification2']
    for col in columns_to_remove:
        if col in columns:
            columns.remove(col)

    columns1 = [x for x in columns if not '2' in x]
    columns2 = [x for x in columns if not '1' in x]
    dataset['all_texts1'] = dataset[columns1].agg(','.join, axis=1)
    dataset['all_texts2'] = dataset[columns2].agg(','.join, axis=1)
    return dataset


def create_empty_dataframe(product_pairs_idx):
    """
    Create empty dataframe for text similarity results with indices of possible pairs after filtration
    @param product_pairs_idx: indices of filtered possible matching pairs
    @return: empty dataframe with suitable column names for all measured text similarities
    """
    idxs_array = []
    for idx1, idxs2 in product_pairs_idx.items():
        for idx2 in idxs2:
            idxs_array.append([idx1, idx2])
    df_column_names = pd.DataFrame(idxs_array,
                                   columns=['index1',
                                            'index2'])

    return df_column_names


def create_empty_dataframe_for_training(text_types, similarity_names):
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
