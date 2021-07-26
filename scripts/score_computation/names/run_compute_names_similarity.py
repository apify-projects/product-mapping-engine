import click

from compute_names_similarity import remove_output_file_is_necessary, load_file, lower_case, remove_colors, \
    compute_tf_idf, compute_similarity_score, save_to_file, evaluate_dataset


@click.command()
@click.option('--input_file', '-i',
              default='data/preprocessed/10_products/names/names_10a_prepro.txt',
              required=False,
              help='Input file with product names dictionary file to preprocess')
@click.option('--output_file', '-o',
              default='results/similarity_score/10_products/names/name_similarity.txt',
              required=False, help='Output file with similarity scores')
@click.option('--filter_distance', '-f', default=False, type=bool,
              help='Whether filter distance above given thresh')
@click.option('--id_weight', '-iw', type=int, default=100, help='Weight for id similarity')
@click.option('--brand_weight', '-bw', type=int, default=10, help='Weight for brand similarity')
@click.option('--cos_weight', '-cw', type=int, default=10, help='Weight for cosine similarity')
@click.option('--word_weight', '-ww', type=int, default=1, help='Weight for same words similarity')
@click.option('--thresh', '-t', default=[5, 10, 20, 30, 50, 60, 70, 100], type=list,
              help='Threshold to evaluate accuracy of similarities')
@click.option('--print_stats', '-p', default=False, type=bool,
              help='Whether print statistical values')
# Load product names and compute their similarity
def main(**kwargs):
    scores = []
    remove_output_file_is_necessary(kwargs['output_file'])

    ''' FOR COMPARISON OF NAMES FROM 2 FILES '''
    names_list1 = load_file(
        'data/preprocessed/10_products/names/10_products_prepro_a.txt')
    names_list2 = load_file(
        'data/preprocessed/10_products/names/10_products_prepro_b.txt')
    names_list1 = lower_case(names_list1)
    names_list2 = lower_case(names_list2)
    names_voc1 = remove_colors(names_list1)
    names_voc2 = remove_colors(names_list2)

    # tf.idf for creation of vectors of words
    names_voc = names_voc1 + names_voc2
    tf_idfs = compute_tf_idf(names_voc)
    weights = {'id': kwargs['id_weight'], 'brand': kwargs['brand_weight'], 'cos': kwargs['cos_weight'],
               'words': kwargs['word_weight']}
    for i, n1 in enumerate(names_list1):
        for j, n2 in enumerate(names_list2):
            similarity_score = compute_similarity_score(n1, i, n2, j, tf_idfs, weights)
            are_names_same = 1 if i == j else 0
            scores.append([n1, n2, are_names_same, similarity_score])
            save_to_file(kwargs['output_file'], n1, n2, similarity_score, are_names_same)
    evaluate_dataset(scores, kwargs['thresh'])

    ''' FOR COMPARISON OF NAMES FROM 1 FILE  
    names_list = load_file(kwargs['input_file'])
    names_list = lower_case(names_list)
    names_list = remove_colors(names_list)
    print(kwargs['id_weight'])
    weights = {'id': kwargs['id_weight'], 'brand': kwargs['brand_weight'], 'cos': kwargs['cos_weight'],
               'words': kwargs['word_weight']}
    tf_idfs = compute_tf_idf(names_list)
    for i, n1 in enumerate(names_list):
        for j, n2 in enumerate(names_list[i + 1::]):
            j += i + 1
            similarity_score = compute_similarity_score(n1, i, n2, j, tf_idfs, weights)
            are_names_same = are_idxs_same(i, j)
            scores.append([n1, n2, are_names_same, similarity_score])
            save_to_file(kwargs['output_file'], n1, n2, similarity_score, are_names_same)
    evaluate_dataset(scores, kwargs['thresh'])
'''


if __name__ == "__main__":
    main()
