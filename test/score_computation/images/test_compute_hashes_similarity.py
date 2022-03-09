import click
import pandas as pd

from scripts.dataset_handler.preprocessing.images.image_preprocessing import load_and_parse_data
from scripts.dataset_handler.similarity_computation.images.compute_hashes_similarity import create_hash_sets, \
    compute_distances


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
    hashes, names = create_hash_sets(data, None, None)

    name_similarities = compute_distances(hashes, names, metric=kwargs['metric'], filter_dist=kwargs['filter_distance'],
                                          thresh=kwargs['filter_distance'])
    save_to_csv(name_similarities, kwargs['output_file'])


if __name__ == "__main__":
    main()


def save_to_csv(data_list, output_file, column_names=None):
    """
    Save data list to csv format
    @param data_list: data as list
    @param output_file: name of the output file
    @param column_names: names of columns
    @return:
    """
    data_dataframe = pd.DataFrame(data_list, columns=column_names)
    data_dataframe.fillna(0, inplace=True)
    data_dataframe.to_csv(output_file, index=False)
