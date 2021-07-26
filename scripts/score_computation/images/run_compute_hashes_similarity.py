import click

from compute_hashes_similarity import load_and_parse_data, create_hash_sets, save_to_txt, compute_distances


@click.command()
@click.option('--input_file', '-i',
              default='data/preprocessed/10_products/images/hashes/hashes_cropped.json',
              required=False,
              help='File with input hashes')
@click.option('--output_file', '-o',
              default='results/similarity_score/10_products/images/hash_distances.txt',
              required=False, help='File to store distances of hashes')
@click.option('--metric', '-m', default='binary', type=click.Choice(['mean', 'binary', 'thresh']),
              help='Metric of hash values distance computation')
@click.option('--filter_distance', '-f', default=False, type=bool,
              help='Whether filter distance above given thresh')
@click.option('--thresh', '-t', default=0.9, type=int,
              help='Threshold to filter data')  # mean=20000
# Load folder with image hashes and compute their distance
def main(**kwargs):
    data = load_and_parse_data(kwargs['input_file'])
    hashes, names = create_hash_sets(data)

    distance_set = compute_distances(hashes, names, metric=kwargs['metric'], filter_dist=kwargs['filter_distance'],
                                     thresh=kwargs['filter_distance'])
    save_to_txt(distance_set, kwargs['output_file'])


if __name__ == "__main__":
    main()
