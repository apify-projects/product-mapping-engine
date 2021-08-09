import click

from scripts.score_computation.images.compute_hashes_similarity import load_and_parse_data, create_hash_sets, \
    compute_distances


@click.command()
@click.option('--input_file', '-i',
              default='test/data/10_products/dataset/preprocessed/hashes_cropped.json',
              required=False,
              help='File with input hashes')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/hash_similarities.txt',
              required=False, help='File to store distances of hashes')
@click.option('--metric', '-m', default='binary', type=click.Choice(['mean', 'binary', 'thresh']),
              help='Metric of hash values distance computation')
@click.option('--filter_distance', '-f', default=False, type=bool,
              help='Whether filter distance above given thresh')
# Load folder with image hashes and compute their distance
def main(**kwargs):
    data = load_and_parse_data(kwargs['input_file'])
    hashes, names = create_hash_sets(data)

    distance_set = compute_distances(hashes, names, metric=kwargs['metric'], filter_dist=kwargs['filter_distance'],
                                     thresh=kwargs['filter_distance'])
    with open(kwargs['output_file'], 'w') as f:
        for d in distance_set:
            f.write(f'{d}\n')


if __name__ == "__main__":
    main()
