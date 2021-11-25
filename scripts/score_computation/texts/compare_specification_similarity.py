import copy

from sklearn.metrics.pairwise import cosine_similarity

from scripts.score_computation.texts.compute_texts_similarity import remove_markers, create_tf_idf, \
    compare_units_and_values


def compare_specifications(dataset1, dataset2):
    """
    Compare two specifications by units and values and compute cos similarity of texts
    @param dataset1: list of products and each contains list of parameter names and values
    @param dataset2: list of products and each contains list of parameter names and values
    @return: cosine similarity and ratio of the same units and values
    """
    scores = []

    dataset1_tfidf = prepare_dataset_for_tf_idfs(copy.deepcopy(dataset1))
    dataset2_tfidf = prepare_dataset_for_tf_idfs(copy.deepcopy(dataset2))

    tf_idfs = create_tf_idf(dataset1_tfidf, dataset2_tfidf)
    for specification1 in dataset1:
        specification1 = [[word for words in param for word in words.split(' ')] for param in specification1]
        specification1 = [item for sublist in specification1 for item in sublist]
        for specification2 in dataset2:
            specification2 = [[word for words in param for word in words.split(' ')] for param in specification2]
            specification2 = [item for sublist in specification2 for item in sublist]
            score = compare_units_and_values(specification1, specification2)
            similarity = cosine_similarity([tf_idfs.iloc[0].values, tf_idfs.iloc[1].values])[0][1]
            scores.append([score, similarity])
    return scores


def prepare_dataset_for_tf_idfs(dataset):
    new_dataset = []
    for product_specification in dataset:
        product_specification = remove_markers(product_specification)
        product_specification = [' '.join(d) for d in product_specification]
        new_dataset.append(product_specification)
    return new_dataset
