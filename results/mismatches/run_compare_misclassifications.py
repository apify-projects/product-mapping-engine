import os

import pandas as pd
from matplotlib import pyplot as plt


def main():
    directory = 'results/mismatches'
    product_indices_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            product_indices_list.append(df['Unnamed: 0'].values)  # TODO: prejmenovat na neco smysluplneho pri ukladani

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
    plt.xlabel('Frequencies of misclassified products')
    plt.ylabel('NUmber of misclassified products')
    plt.show()


if __name__ == "__main__":
    main()
