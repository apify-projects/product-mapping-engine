import click

from compute_total_score import load_file, save_to_file, evaluate_dataset, compute_distance

names_file = '../names/data/results/scores_ab.txt'
images_file = '../img_hash/data/results/all_dist_bin_cropped.txt'
output_file = 'results/name_img_similarity.txt'


@click.command()
@click.option('--names_file', '-n',
              default='C:/Users/kater/PycharmProjects/ProductMapping/results/similarity_score/10_products/names/name_similarity.txt',
              required=False,
              help='Input file with product names to compute similarity')
@click.option('--images_file', '-i',
              default='C:/Users/kater/PycharmProjects/ProductMapping/results/similarity_score/10_products/images/hash_distances.txt',
              required=False,
              help='Input file with product images to compute similarity')
@click.option('--output_file', '-o',
              default='C:/Users/kater/PycharmProjects/ProductMapping/results/similarity_score/10_products/names_and_images/scores.txt',
              required=False, help='Output file with similarity scores of products')
@click.option('--name_weight', '-nw', type=int, default=1, help='Weight for names')
@click.option('--image_weight', '-iw', type=int, default=1, help='Weight for images')
@click.option('--thresh', '-t', default=[50, 70, 90, 100, 120], type=list,
              help='Threshold to evaluate accuracy of similarities')
@click.option('--print_stats', '-p', default=False, type=bool,
              help='Whether print statistical values')
# Load product names and images compute their similarity
def main(**kwargs):
    names_data = load_file(kwargs['names_file'])
    images_data = load_file(kwargs['images_file'])
    distances = compute_distance(images_data, names_data, kwargs['name_weight'],
                                 kwargs['image_weight'], kwargs['print_stats'])
    save_to_file(distances, kwargs['output_file'])
    evaluate_dataset(distances, kwargs['thresh'], kwargs['print_stats'])


if __name__ == "__main__":
    main()
