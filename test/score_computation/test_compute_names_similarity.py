import click
import pandas as pd

from scripts.score_computation.names.compute_names_similarity import load_file, lower_case, remove_colors, \
    compute_tf_idf, compute_name_similarities


@click.command()
@click.option('--input_file_1', '-i1',
              default='test/data/10_products/dataset/preprocessed/names/names_products_1_prepro.csv',
              required=False,
              help='Input file 1 with product names dictionary file to preprocess')
@click.option('--input_file_2', '-i2',
              default='test/data/10_products/dataset/preprocessed/names/names_products_2_prepro.csv',
              required=False,
              help='Input file 2 with product names dictionary file to preprocess')
@click.option('--output_file', '-o',
              default='test/data/10_products/dataset/preprocessed/name_similarities.txt',
              required=False, help='Output file with similarity scores')
# Load product names and compute their similarities
def main(**kwargs):
    names_list1 = load_file(kwargs['input_file_1'])
    names_list2 = load_file(kwargs['input_file_2'])
    names_list1 = lower_case(names_list1)
    names_list2 = lower_case(names_list2)
    names_voc1 = remove_colors(names_list1)
    names_voc2 = remove_colors(names_list2)

    # tf.idf for creation of vectors of words
    names_voc = names_voc1 + names_voc2
    tf_idfs = compute_tf_idf(names_voc)
    name_similarities_list = []
    for i, (n1, n2) in enumerate(zip(names_list1, names_list2)):
        name_similarities = compute_name_similarities(n1, n2, i, i, tf_idfs)
        name_similarities_list.append(name_similarities)

    name_similarities_dataframe = pd.DataFrame(name_similarities_list)
    name_similarities_dataframe.fillna(0, inplace=True)
    name_similarities_dataframe.to_csv(kwargs['output_file'], index=False)


if __name__ == "__main__":
    main()
