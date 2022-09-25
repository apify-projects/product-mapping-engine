import base64
import hashlib
import json
import os

import numpy as np
import pandas as pd
from slugify import slugify

from ....configuration import HEX_GROUPS_FOR_IMAGE_HASHES, IS_ON_PLATFORM, IMAGE_FILTERING, IMAGE_FILTERING_THRESH
from ...preprocessing.images.image_preprocessing import compute_image_hashes, load_and_parse_data


def hex_to_dec(hex_val):
    """
    Takes every group of BIT_GROUPS characters and converts them from hex to dec
    @param hex_val: hex value of the hash
    @return: decimal value of the hash
    """
    fourths = [hex_val[i:i + HEX_GROUPS_FOR_IMAGE_HASHES] for i in range(0, len(hex_val), HEX_GROUPS_FOR_IMAGE_HASHES)]
    return [int(f, 16) for f in fourths]


def hex_to_bin(val):
    """
    Convert string hash hex value to binary hash value char by char
    @param val: hex value of the hash
    @return: binary value of the hash
    """
    hash_bin = []
    for i in val:
        hash_bin.append(bin(int(i, 16))[2:].zfill(4))
    return str.join('', [val for sub in hash_bin for val in sub])


def dec_similarity(list1, list2):
    """
    Compute difference of two list of hashes of images of two given products
    @param list1: hash 1 in decimal
    @param list2: hash 2 in decimal
    @return: distance of hashes
    """
    diff = 0
    for i, j in zip(list1, list2):
        diff += (abs(i - j))
    return diff


def bit_similarity(hash1, hash2):
    """
    Compute number of different bits in two hashes
    @param hash1: image hash 1
    @param hash2: image hash 2
    @return: ratio of different bits
    """
    matches = 0
    for i1, i2 in zip(hash1, hash2):
        if i1 == i2:
            matches += 1
    return matches / len(hash1)


def compute_distances(hashes, names, metric, filter_dist, thresh):
    """
    Convert hashes to dec or binary and compute distances between every two set of images of products
    @param hashes: set of hashes of images
    @param names: set of image names
    @param metric: metric of computing the distance
    @param filter_dist: whether the images that are too far should be filtered or not
    @param thresh: thresh value above which the images should be filtered
    @return: distances of all images of pairs of products
    """
    pair_similarities = []
    for hash_set, names_set in zip(hashes, names):
        pair_index = int(names_set[0].split('_')[0].split('pair')[1])
        first_product = []
        second_product = []
        first_name = names_set[0].split('_')[1]
        for h, n in zip(hash_set, names_set):
            if n.split('_')[1] == first_name:
                first_product.append([h, n])
            else:
                second_product.append([h, n])
        if len(first_product) == 0 or len(second_product) == 0:
            pair_similarities.append((pair_index, 0))
            continue
        similarities = []
        for (first_hash, first_name) in first_product:
            for (second_hash, second_name) in second_product:

                if metric == 'binary':
                    first_hash_transferred = [hex_to_bin(k) for k in first_hash]
                    second_hash_transferred = [hex_to_bin(k) for k in second_hash]
                    sim = bit_similarity(first_hash_transferred, second_hash_transferred)
                else:
                    first_hash_transferred = hex_to_dec(first_hash)
                    second_hash_transferred = hex_to_dec(second_hash)
                    sim = dec_similarity(first_hash_transferred, second_hash_transferred)

                if filter_dist and sim > thresh:
                    sim = None

                similarities.append(sim)
        # for each image from the first set find the most similar in the second set
        # and check whether is the distance below threshold
        sim = sum([float(s) for s in similarities if s is not None])
        pair_similarities.append((pair_index, sim))

    return pair_similarities


def create_hash_sets(
        data,
        pair_ids_and_counts_dataframe,
        dataset_prefixes
):
    """
    Create list of lists of image hashes and names for each product
    @param data: input dict of image name and hash value
    @param pair_ids_and_counts_dataframe: dataframe containing ids and image counts for the pairs of products
    @param dataset_prefixes: prefixes of images identifying them as parts of a specific dataset
    @return: list of hashes and list on names of images
            (returns as a list for easier further multi thread parallel processing)
    """
    hashes = []
    names = []

    pair_index = 0
    for pair in pair_ids_and_counts_dataframe:
        pair_hashes = []
        pair_names = []
        for dataset_index in range(2):
            id_key = 'id{}'.format(dataset_index + 1)
            image_count_key = 'image{}'.format(dataset_index + 1)
            for image_index in range(int(pair[image_count_key])):
                # TODO equalize indexing (images for instance have indexing starting from 0 and from 1)
                image_name = dataset_prefixes[dataset_index] + '_' + \
                             hashlib.sha224(slugify(pair[id_key] + '_image_{}'.format(image_index)).encode(
                                 'utf-8')).hexdigest() + '.jpg'
                pair_image_identification = 'pair{}_product{}_image{}'.format(pair_index, dataset_index + 1,
                                                                              image_index + 1)

                if image_name in data:
                    one_hash = data[image_name]
                    pair_hashes.append(one_hash)
                    pair_names.append(pair_image_identification)

        if len(pair_hashes) > 0:
            hashes.append(pair_hashes)
            names.append(pair_names)

        pair_index += 1

    print("Hashes")
    print(len(hashes))
    print(len(names))
    return [hashes, names]


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
                with open(os.path.join(img_dir, prefix + '_' + hashlib.sha224(image_name.encode('utf-8')).hexdigest()),
                          'wb') as image_file:
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
        dataset_images_kvs2=None,
        is_on_platform=IS_ON_PLATFORM
):
    """
    Compute images similarities and create dataset with hash similarity
    @param pool: pool of Python processes (from the multiprocessing library)
    @param num_cpu: the amount of processes the CPU can handle at once
    @param pair_ids_and_counts_dataframe: dataframe containing ids and image counts for the pairs of products
    @param dataset_folder: folder to be used as dataset root, determining where the images will be stored
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @param is_on_platform: True if this is running on the platform
    @return: Similarity scores for the images
    """
    if dataset_folder == '':
        dataset_folder = '.'

    img_dir = os.path.join(dataset_folder, 'images')
    hashes_file_path = os.path.join(dataset_folder, 'precomputed_hashes.json')

    dataset_prefixes = ['dataset1', 'dataset2']

    if is_on_platform and os.path.exists(hashes_file_path):
        with open(hashes_file_path, 'r') as hashes_file:
            hashes_data = json.load(hashes_file)
    else:
        print("Image download started")

        download_images_from_kvs(img_dir, dataset_images_kvs1, dataset_prefixes[0])
        download_images_from_kvs(img_dir, dataset_images_kvs2, dataset_prefixes[1])

        print("Image download finished")
        print("Image preprocessing started")
        script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../../preprocessing/images/image_hash_creator/main.js")
        image_filenames = os.listdir(img_dir)
        image_filenames_chunks = np.array_split(image_filenames, num_cpu)
        hash_files = pool.map(
            multi_run_compute_image_hashes,
            [
                (index, dataset_folder, img_dir, image_filenames_chunk, script_dir)
                for index, image_filenames_chunk in enumerate(image_filenames_chunks)
            ]
        )

        hashes_data = load_and_parse_data(hash_files)
        if is_on_platform:
            with open(hashes_file_path, 'w') as hashes_file:
                json.dump(hashes_data, hashes_file)

    pair_ids_and_counts_dataframe_parts = np.array_split(pair_ids_and_counts_dataframe, num_cpu)
    hashes_names_list = pool.map(multi_run_create_images_hash_wrapper,
                                 [(hashes_data, pair_ids_and_counts_dataframe_part, dataset_prefixes) for
                                  pair_ids_and_counts_dataframe_part in pair_ids_and_counts_dataframe_parts])
    print("Image preprocessing finished")

    print("Image similarities computation started")
    imaged_pairs_similarities_list = pool.map(multi_run_compute_distances_wrapper,
                                              [(item[0], item[1], 'binary', IMAGE_FILTERING, IMAGE_FILTERING_THRESH) for
                                               item in hashes_names_list])
    imaged_pairs_similarities = [item for sublist in imaged_pairs_similarities_list for item in sublist]
    print("Image similarities computation finished")

    # Correctly order the similarities and fill in 0 similarities for pairs that don't have images
    image_similarities = []
    image_similarities_df = pd.DataFrame(columns=['id1', 'id2', 'hash_similarity'])
    image_similarities_df['id1'] = [item['id1'] for item in pair_ids_and_counts_dataframe]
    image_similarities_df['id2'] = [item['id2'] for item in pair_ids_and_counts_dataframe]
    for x in range(len(pair_ids_and_counts_dataframe)):
        image_similarities.append(0)
    for index, similarity in imaged_pairs_similarities:
        image_similarities[index] = similarity
    image_similarities_df['hash_similarity'] = image_similarities
    return image_similarities_df
