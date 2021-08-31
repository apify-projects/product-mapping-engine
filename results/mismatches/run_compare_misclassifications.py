import os

import pandas as pd
from matplotlib import pyplot as plt


def main():
    directory = 'results/mismatches'
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
    plt.show()


if __name__ == "__main__":
    main()
