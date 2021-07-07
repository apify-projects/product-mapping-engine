import click

from evaluate_classifier import preprocess_name, compute_similarity_score
from scripts.score_computation.names.compute_names_similarity import load_file, save_to_file, compute_tf_idf, \
    are_idxs_same, evaluate_dataset


@click.command()
@click.option('--names_file', '-n',
              default='C:/Users/kater/PycharmProjects/product-mapping/results/similarity_score/10_products/names/name_similarity.txt',
              required=False,
              help='Input file with product names to compute similarity')
@click.option('--images_file', '-i',
              default='C:/Users/kater/PycharmProjects/product-mapping/results/similarity_score/10_products/images/hash_distances.txt',
              required=False,
              help='Input file with product images to compute similarity')
@click.option('--output_file', '-o',
              default='C:/Users/kater/PycharmProjects/product-mapping/results/similarity_score/10_products/names_and_images/scores.txt',
              required=False, help='Output file with similarity scores of products')
@click.option('--classifier', '-c', default='linear',
              type=click.Choice(['linear', 'svg']))
@click.option('--filter_distance', '-f', default=False, type=bool,
              help='Whether filter distance above given thresh')
@click.option('--name_weight', '-nw', type=int, default=1, help='Weight for names')
@click.option('--image_weight', '-iw', type=int, default=1, help='Weight for images')
@click.option('--id_weight', '-niw', type=int, default=100, help='Weight for id similarity in names')
@click.option('--brand_weight', '-nbw', type=int, default=10, help='Weight for brand similarity in names')
@click.option('--cos_weight', '-ncw', type=int, default=10, help='Weight for cosine similarity in names')
@click.option('--word_weight', '-nww', type=int, default=1, help='Weight for same words similarity in names')
@click.option('--thresh', '-t', default=[50, 70, 90, 100, 120], type=list,
              help='Threshold to evaluate accuracy of similarities')
@click.option('--print_stats', '-p', default=False, type=bool,
              help='Whether print statistical values')
# Load product names and images compute their similarity
def main(**kwargs):
    images_data = load_file(kwargs['images_file'])
    names_list = load_file(kwargs['names_file'])

    names_list = preprocess_name(names_list)
    scores = []

    weights = {'name': kwargs['name_weight'], 'image': kwargs['image_weight'], 'id': kwargs['id_weight'],
               'brand': kwargs['brand_weight'], 'cos': kwargs['cos_weight'],
               'words': kwargs['word_weight']}

    tf_idfs = compute_tf_idf(names_list)
    thresh_img = max([float(i.split(',')[2]) for i in images_data])
    for i, n1 in enumerate(names_list):
        for j, n2 in enumerate(names_list[i + 1::]):
            j += i + 1
            similarity_score = compute_similarity_score(kwargs['classifier'], n1, n2, i, j, images_data[i],
                                                        weights, tf_idfs, thresh_img)
            are_products_same = are_idxs_same(i, j)
            scores.append([n1, n2, are_products_same, similarity_score])

            save_to_file(kwargs['output_file'], n1, n2, similarity_score, are_products_same)
    evaluate_dataset(scores, kwargs['thresh'])


if __name__ == "__main__":
    main()
