import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    data = pd.read_csv('data/wdc_dataset/dataset/preprocessed/product_pairs.csv')
    directory = 'results/mismatches/data'
    product_indices_list = load_mismatches_from_classificators(directory)
    product_indices_dict = create_mismatched_product_indices_dict(product_indices_list)
    plot_mismatched_products_counts(product_indices_dict)
    find_mismatching_product_names(product_indices_dict, data, 6)


def load_mismatches_from_classificators(directory):
    """
    Load wrongly predicted products for each classificator
    @param directory: directory with wrongly predicted product files
    @return:  list of wrongly predicted products for each classificator
    """
    product_indices_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            product_indices_list.append(df['index'].values)
    return product_indices_list


def create_mismatched_product_indices_dict(product_indices_list):
    """
    Create dictionary with indices and number of classifiers that predicted them wrongly
    @param product_indices_list: list of wrongly predicted products for each classificator
    @return: dictionary with indices and number of classifiers that predicted them wrongly
    """
    product_indices_dict = {}
    for product_indices in product_indices_list:
        for values in product_indices:
            if not values in product_indices_dict:
                product_indices_dict[values] = 0
            product_indices_dict[values] += 1
    return product_indices_dict


def plot_mismatched_products_counts(product_indices_dict):
    """
    Plot counts of products that were wrongly predicted according to the number of classifiers that predicted them wrongly
    @param product_indices_dict: indices of products and number of classifiers predicted them wrongly
    @return:
    """
    values_occurrences = dict(
        (x, list(product_indices_dict.values()).count(x)) for x in list(product_indices_dict.values()))
    plt.bar(values_occurrences.keys(), values_occurrences.values(), color='green')
    plt.title('Comparison of frequencies of misclassified products among classificators')
    plt.xlabel('Frequencies of misclassified products among all classificators')
    plt.ylabel('Number of misclassified products')


def find_mismatching_product_names(product_indices_dict, data, mismatches):
    """
    Print product pairs which were predicted wrongly by given number of classificators
    @param product_indices_dict: indices of products and number of classifiers predicted them wrongly
    @param data: original dataset with product pairs
    @param mismatches: number of classificators, that predicted wrongly a given products
    @return:
    """
    product_indices_allwrong_idxs = [k for k, v in product_indices_dict.items() if v == mismatches]
    product_indices_allwrong_idxs.sort()
    indices = np.array(product_indices_allwrong_idxs)
    names = np.asarray(data[['name1', 'name2', 'match']])
    mismatched_names = np.array([names[i] for i in indices])
    print(f'Number of mismatched pairs is: {len(mismatched_names)}')
    output_file_path = 'results/mismatches/'
    df = pd.DataFrame(mismatched_names, columns=['name1', 'name2', 'match'])
    df = df.replace(to_replace='^\[ ', value='', regex=True)
    df = df.replace(to_replace='\ ]$', value='', regex=True)
    df_fp = df[(df['match'] == 0)]
    df_fn = df[(df['match'] == 1)]
    df_fp.to_csv(os.path.join(output_file_path, 'mismatched_product_pairs_fp.csv'))
    df_fn.to_csv(os.path.join(output_file_path, 'mismatched_product_pairs_fn.csv'))


if __name__ == "__main__":
    main()
