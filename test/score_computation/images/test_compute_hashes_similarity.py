import click

from scripts.score_computation.dataset_handler import save_to_csv, load_and_parse_data
from scripts.score_computation.images.compute_hashes_similarity import create_hash_sets, compute_distances


@click.command()
@click.option('--input_file', '-i',
              default='test/data/10_products/dataset/preprocessed/hashes_cropped.json',
              required=False,
              help='File with input hashes')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/hash_similarities.csv',
              required=False, help='File to store distances of hashes')
@click.option('--metric', '-m', default='binary', type=click.Choice(['mean', 'binary', 'thresh']),
              help='Metric of hash values distance computation')
@click.option('--filter_distance', '-f', default=False, type=bool,
              help='Whether filter distance above given thresh')
# Load folder with image hashes and compute their distance
def main(**kwargs):
    data = load_and_parse_data(kwargs['input_file'])
    hashes, names = create_hash_sets(data)

    name_similarities = compute_distances(hashes, names, metric=kwargs['metric'], filter_dist=kwargs['filter_distance'],
                                          thresh=kwargs['filter_distance'])
    save_to_csv(name_similarities, kwargs['output_file'])


if __name__ == "__main__":
    main()
