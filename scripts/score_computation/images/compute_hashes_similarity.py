from slugify import slugify

NAME_CHAR_SUBSET = 6
BIT_GROUPS = 4
HEX_GROUPS = 1


def hex_to_dec(hex_val):
    """
    Takes every groups of BIT_GROUPS characters and converts them from hex to dec
    @param hex_val: hex value of the hash
    @return: decimal value of the hash
    """
    fourths = [hex_val[i:i + HEX_GROUPS] for i in range(0, len(hex_val), HEX_GROUPS)]
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
        for (first_hash, first_name) in first_product:
            for (second_hash, second_name) in second_product:
                similarities = []

                if metric == 'binary':
                    first_hash_tranf = [hex_to_bin(k) for k in first_hash]
                    second_hash_transf = [hex_to_bin(k) for k in second_hash]
                    sim = bit_similarity(first_hash_tranf, second_hash_transf)
                else:
                    first_hash_tranf = hex_to_dec(first_hash)
                    second_hash_transf = hex_to_dec(second_hash)
                    sim = dec_similarity(first_hash_tranf, second_hash_transf)

                if filter_dist and sim > thresh:
                    sim = None

                similarities.append(sim)
        # for each image from the first set find the most similar in the second set and check whether is the distance below threshold
        sim = sum([float(s) for s in similarities if s is not None])
        pair_similarities.append((pair_index, sim))

    return pair_similarities


def create_hash_sets(data, pair_ids_and_counts, dataset_prefixes):
    """
    Create list of lists of image hashes and names for each product
    @param data: input dict of image name and hash value
    @return: list of hashes and list on names of images
    """
    hashes = []
    names = []

    pair_index = 0
    for pair in pair_ids_and_counts:
        pair_hashes = []
        pair_names = []
        for dataset_index in range(2):
            id_key = 'id{}'.format(dataset_index+1)
            image_count_key = 'image{}'.format(dataset_index + 1)
            for image_index in range(int(pair[image_count_key])):
                # TODO equalize indexing (images for instance have indexing starting from 0 and from 1)
                image_name = dataset_prefixes[dataset_index] + '_' + slugify(pair[id_key] + '_image_{}'.format(image_index)) + '.jpg'
                pair_image_identification = 'pair{}_product{}_image{}'.format(pair_index, dataset_index + 1, image_index + 1)
                hash = data[image_name]

                pair_hashes.append(hash)
                pair_names.append(pair_image_identification)

        hashes.append(pair_hashes)
        names.append(pair_names)
        pair_index += 1

    return hashes, names
