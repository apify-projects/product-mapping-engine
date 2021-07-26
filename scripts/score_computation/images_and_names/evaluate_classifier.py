from scripts.score_computation.names.compute_names_similarity import ID_MARK, BND_MARK
from scripts.score_computation.names.compute_names_similarity import remove_markers, cosine_similarity, lower_case, \
    remove_colors

PRINT_STATS = False


def preprocess_name(names_list):
    """
    Preprocess name - lowercase and remove colors
    @param names_list: list of names
    @return: preprocessed list of names
    """
    names_list = lower_case(names_list)
    names_list = remove_colors(names_list)
    return names_list


def compute_similarity_score(classifier, n1, n2, i, j, images_similarity, weights, tf_idfs, thresh_img):
    """
    Compute similarity score of two products comparing images and names
    @param classifier: classifier to be used to train weights of parameters
    @param n1: first name
    @param n2: second name
    @param i: first name index
    @param j: second name index
    @param images_similarity: similarity of images
    @param weights: weights of parameters as id, cosine similarity, brand, same wards match
    @param tf_idfs: tf ids of names
    @param thresh_img: thresh of image above which the images are not considered to be similar
    @return: similarity score of names and images
    """
    name1 = n1.split(' ')
    name2 = n2.split(' ')
    match_ratios = {}

    # detect and compare ids
    id1 = [word for word in name1 if ID_MARK in word]
    id2 = [word for word in name2 if ID_MARK in word]
    match_ratios['id'] = len(set(id1) & set(id2)) / len(id1) if not id1 == [] else 0

    # detect and compare brands
    bnd1 = [word for word in name1 if BND_MARK in word]
    bnd2 = [word for word in name2 if BND_MARK in word]
    match_ratios['brand'] = len(set(bnd1) & set(bnd2)) / len(bnd1) if not bnd1 == [] and bnd1 == bnd2 else 0

    # ratio of the similar words
    name1 = remove_markers(name1)
    name2 = remove_markers(name2)
    list1 = set(name1)
    intersection = list1.intersection(name2)
    intersection_list = list(intersection)
    match_ratios['words'] = len(intersection_list) / len(name1)

    # cosine similarity of vectors from tf.idf
    match_ratios['cos'] = cosine_similarity([tf_idfs.iloc[i].values, tf_idfs.iloc[j].values])[0][1]

    # images similarity
    match_ratios['image'] = 0 if images_similarity > thresh_img else (thresh_img - images_similarity) / thresh_img * 100

    if classifier == 'linear':
        return compute_similarity_linear(match_ratios, weights)
    return 0


def compute_similarity_linear(match_ratios, weights):
    """
    Compute linear similarity of names and images
    @param match_ratios: match score of parameters as brand, id, cosine and similar words
    @param weights: weights of params to be trained
    @return:
    """
    name_sim = weights['brand'] * match_ratios['brand']
    name_sim += weights['words'] * match_ratios['words']
    name_sim += weights['id'] * match_ratios['id']
    name_sim += weights['cos'] * match_ratios['cos']
    similarity_score = weights['name'] * name_sim + weights['image'] * match_ratios['image']
    return similarity_score
