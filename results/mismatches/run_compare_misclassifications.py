import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    directory = 'results/mismatches/data'
    product_indices_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            product_indices_list.append(df['index'].values)

    product_indices_dict = {}
    for product_indices in product_indices_list:
        for values in product_indices:
            if not values in product_indices_dict:
                product_indices_dict[values] = 0
            product_indices_dict[values] += 1

    values_occurrences = dict(
        (x, list(product_indices_dict.values()).count(x)) for x in list(product_indices_dict.values()))

    plt.bar(values_occurrences.keys(), values_occurrences.values(), color='green')
    plt.title('Comparison of frequencies of misclassified products among classificators')
    plt.xlabel('Frequencies of misclassified products among all classificators')
    plt.ylabel('Number of misclassified products')
    # plt.show()

    # plot product pairs which were predicted wrongly by all classificators
    product_indices_allwrong_idxs = [k for k, v in product_indices_dict.items() if v == 6]
    product_indices_allwrong_idxs.sort()
    indices = np.array(product_indices_allwrong_idxs)
    data = pd.read_csv('data/wdc_dataset/dataset/preprocessed/product_pairs.csv')
    names = np.asarray(data[['name1', 'name2', 'match']])

    mismatched_names = np.array([names[i] for i in indices])

    #print(mismatched_names)
    print(f'Number of mismatched pairs is: {len(mismatched_names)}')

    output_file = 'results/mismatches/mismatched_product_pairs.csv'
    df = pd.DataFrame(mismatched_names, columns=['name1','name2', 'match'])
    df.to_csv(output_file)

if __name__ == "__main__":
    main()
