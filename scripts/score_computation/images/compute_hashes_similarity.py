import json
import sys

NAME_CHAR_SUBSET = 3
BIT_GROUPS = 4


def load_and_parse_data(input_file):
    """
    Load input file and split name and hash into dictionary
    @param input_file: file with hasehs and names
    @return: dictionary with name and has value of the image
    """
    data = {}
    with open(input_file) as json_file:
        loaded_data = json.load(json_file)

    for d in loaded_data:
        dsplit = d.split(';')
        data[dsplit[0]] = dsplit[1]
    return data


def create_hash_sets(data):
    """
    Create list of lists of image hashes and names for each product
    @param data: input dict of image name and hash value
    @return: list of hashes and list on names of images
    """
    hashes = []
    names = []
    last_name = list(data.keys())[0][:NAME_CHAR_SUBSET]
    hash_set = []
    img_names = []
    for name, hashval in data.items():
        if name[:NAME_CHAR_SUBSET] == last_name:
            hash_set.append(hashval)
            img_names.append(name)
        else:
            hashes.append(hash_set)
            names.append(img_names)
            hash_set = []
            img_names = []
            hash_set.append(hashval)
            img_names.append(name)
            last_name = name[:NAME_CHAR_SUBSET]
    names.append(img_names)
    hashes.append(hash_set)
    return hashes, names


def create_hash_sets_wdc(data):
    NAME_CHAR_SUBSET = 6
    hashes = []
    names = []
    last_name = list(data.keys())[0].split('_')
    last_name = last_name[0] + last_name[1]
    img_names = []
    hash_set = []
    for orig_name, hashval in data.items():
        orig_name = orig_name.split('_')
        name = orig_name[0] + orig_name[1]
        orig_name = f'{orig_name[0]}{orig_name[1]}_{orig_name[2]}{orig_name[3]}_{orig_name[4]}{orig_name[5]}'
        if name == last_name:
            hash_set.append(hashval)
            img_names.append(orig_name)
        else:
            hashes.append(hash_set)
            names.append(img_names)
            hash_set = []
            img_names = []
            hash_set.append(hashval)
            img_names.append(orig_name)
            last_name = name
    names.append(img_names)
    hashes.append(hash_set)
    return hashes, names


def hex_to_dec(hex_val):
    """
    Takes every groups of BIT_GROUPS characters and converts them from hex to dec
    @param hex_val: hex value of the hash
    @return: decimal value of the hash
    """
    fourths = [hex_val[i:i + BIT_GROUPS] for i in range(0, len(hex_val), BIT_GROUPS)]
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


def compute_difference(list1, list2):
    """
    Compute difference of two list of hashes of images of two given products
    @param list1: hash 1
    @param list2: hash 2
    @return: distance of hashes
    """
    diff = 0
    for i, j in zip(list1, list2):
        diff += (abs(i - j))
    return diff


def get_nearest_image(img, name, imgset, names):
    """
    Find nearest image to given one from the set of images
    @param img: image hash values
    @param name: image name
    @param imgset: set of image hashes to find the nearest on
    @param names: set of image names
    @return: distance and nearest image name
    """
    dist = sys.maxsize
    nearest_name = None
    for i, (img2, name2) in enumerate(zip(imgset, names)):
        diff = compute_difference(img, img2)
        if diff < dist:
            dist = diff
            nearest_name = name2
    return dist, nearest_name


def bit_distance_old(img, name, imgset, names):
    """
    Find one image in the image set for given image with the highest number of matching bits
    @param img: image to be compared
    @param name: image name
    @param imgset: set of images to find the most similar image
    @param names: names of the images
    @return: matching score and name of the nearest image
    """
    match = 0
    nearest_name = None
    for img2, name2 in zip(imgset, names):
        matches = 0
        for i1, i2 in zip(img, img2):
            if i1 == i2:
                matches += 1
        if matches > match:
            match = matches
            nearest_name = name2
    return match / len(img), nearest_name

def bit_distance(hash1, hash2):
    matches = 0
    for i1, i2 in zip(hash1, hash2):
        if i1 == i2:
            matches += 1
    return matches / len(hash1)

def compute_distances_wdc(hashes, names, metric, filter_dist, thresh):
    """
    Convert hashes to dec or binary and compute distances between every two set of images of products
    @param hashes: set of hashes of images
    @param names: set of image names
    @param metric: metric of computing the distance
    @param filter_dist: whether the images that are too far should be filtered or not
    @param thresh: thresh value above which the images should be filtered
    @return: distances of all images
    """
    pair_similarities = []
    for hash_set, names_set in zip(hashes, names):
        first_product = []
        second_product = []
        first_name = names_set[0].split('_')[1]
        for h, n in zip(hash_set, names_set):
            if n.split('_')[1] == first_name:
                first_product.append([h, n])
            else:
                second_product.append([h, n])
        if len(first_product) == 0 or len(second_product) == 0:
            pair_similarities.append(0)
            continue
        for (first_hash, first_name) in first_product:
            for (second_hash, second_name) in second_product:
                similarities = []

                if metric == 'binary':
                    first_hash_tranf = [hex_to_bin(k) for k in first_hash]
                    second_hash_transf = [hex_to_bin(k) for k in second_hash]
                else:
                    first_hash_tranf = [hex_to_dec(k) for k in first_hash]
                    second_hash_transf = [hex_to_dec(k) for k in second_hash]

                if metric == 'binary':
                    sim = bit_distance(first_hash_tranf, second_hash_transf)
                else:
                    # TODO: opravit
                    sim, nearest_name = get_nearest_image(first_hash_tranf, first_name, second_hash_transf, second_name)
                if filter_dist and sim > thresh:
                    sim = None

                similarities.append(sim)
        # for each image from the first set find the most similar in the second set and check whether is the distance below threshold
        sim = sum([float(s) for s in similarities if s != None])
        pair_similarities.append(sim)

    return pair_similarities


def compute_distances(hashes, names, metric, filter_dist, thresh):
    """
    Convert hashes to dec or binary and compute distances between every two set of images of products
    @param hashes: set of hashes of images
    @param names: set of image names
    @param metric: metric of computing the distance
    @param filter_dist: whether the images that are too far should be filtered or not
    @param thresh: thresh value above which the images should be filtered
    @return: distances of all images
    """
    all_images_distances = []
    for i, (first_hash, first_name) in enumerate(zip(hashes, names)):
        image_distances = []
        total_distances = []
        for j, (second_hash, second_name) in enumerate(zip(hashes[i + 1::], names[i + 1::])):
            j += i + 1
            distances = []

            if metric == 'binary':
                first_hash_tranf = [hex_to_bin(k) for k in first_hash]
                second_hash_transf = [hex_to_bin(k) for k in second_hash]
            else:
                first_hash_tranf = [hex_to_dec(k) for k in first_hash]
                second_hash_transf = [hex_to_dec(k) for k in second_hash]

            # for each image from the first set find the most similar in the second set and check whether is the distance below threshold
            for num, name in zip(first_hash_tranf, first_name):
                if metric == 'binary':
                    dist, nearest_name = bit_distance(num, name, second_hash_transf, second_name)
                    if filter_dist and dist > thresh:
                        dist, nearest_name = None, None
                else:
                    dist, nearest_name = get_nearest_image(num, name, second_hash_transf, second_name)
                    if filter_dist and dist > thresh:
                        dist, nearest_name = None, None

                image_distances.append([name, nearest_name, dist])
                distances.append(dist)

            dst = sum([float(v) for v in distances if v != None])
            total_distances.append([i, j, dst])
        all_images_distances.append(total_distances)
    return all_images_distances


def save_to_txt(data, output_file):
    """
    Save data to txt file
    @param data: data to save
    @param output_file: file to save data
    @return:
    """
    with open(output_file, 'w') as f:
        for img_sim_set in data:
            for one_set in img_sim_set:
                f.write(f'{one_set[0]}, {one_set[1]}, {one_set[2]}\n')

def save_to_txt_wdc(data, output_file):
    """
    Save data to txt file
    @param data: data to save
    @param output_file: file to save data
    @return:
    """
    with open(output_file, 'w') as f:
        for d in data:
            f.write(f'{d}\n')