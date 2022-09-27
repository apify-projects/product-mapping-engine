import bisect

import numpy as np
import pandas as pd

from ..configuration import MIN_MATCH_PRICE_RATIO, MAX_MATCH_PRICE_RATIO, MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH, \
    MIN_DESCRIPTIVE_WORDS_FOR_MATCH, MIN_LONG_PRODUCT_NAME_SIMILARITY_FOR_MATCH
from .similarity_computation.texts.compute_texts_similarity import \
    compute_descriptive_words_similarity


def multi_run_filter_wrapper(args):
    """
    Wrapper for passing more arguments to filter_possible_product_pairs_parallelly in parallel way
    @param args: Arguments of the function
    @return: call the filter function in parallel way
    """
    return filter_possible_product_pairs_parallelly(*args)


def filter_possible_product_pairs(dataset1, dataset2, descriptive_words, pool, num_cpu):
    """
    Filter possible pairs of two datasets using price similar words and descriptive words filter
    @param pool: parallelling object
    @param num_cpu: number of processes
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return dict with key as indices of products from the first dataset and
            values as indices of filtered possible matching products from second dataset
    """
    dataset2_no_price_idx = dataset2.index[(dataset2['price'] == 0) | (dataset2['price'] == '') | (dataset2['price'].isnull())].tolist()
    dataset2_no_price_products = dataset2.iloc[dataset2_no_price_idx]

    dataset1 = dataset1.sort_values(by=['price'])
    dataset2 = dataset2.sort_values(by=['price'])
    dataset_start_index = len(dataset1)

    dataset_list = np.array_split(dataset1, num_cpu)
    filtered_indices_dicts = pool.map(multi_run_filter_wrapper,
                                      [(dataset_subset, dataset2, dataset2_no_price_products, dataset_start_index,
                                        descriptive_words) for dataset_subset in dataset_list])
    pairs_dataset_idx = {}
    for filtered_dict in filtered_indices_dicts:
        pairs_dataset_idx.update(filtered_dict)
    return pairs_dataset_idx


def filter_possible_product_pairs_parallelly(dataset1, dataset2, dataset2_no_price_products, dataset_start_index,
                                             descriptive_words):
    """
    Filter possible pairs of two datasets using price similar words and descriptive words filter in parallel way
    @param dataset_start_index: starting index to index the products from second dataset in descriptive words
    @param dataset2_no_price_products: products from second dataset without specified prices
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return dict with key as indices of products from the first dataset and
            values as indices of filtered possible matching products from second dataset
    """
    dataset1.info()
    product = dataset1.iloc[0, :]
    idx_start, idx_to = 0, 0

    if 'price' in product.index.values and 'price' in dataset2:
        idx_start = bisect.bisect_left(dataset2['price'].values, MIN_MATCH_PRICE_RATIO * product['price'])
        idx_to = bisect.bisect(dataset2['price'].values, product['price'] * MAX_MATCH_PRICE_RATIO)
        if idx_to == len(dataset2):
            idx_to -= 1

    pairs_dataset_idx = {}
    for idx, product in dataset1.iterrows():
        data_subset_idx, idx_start, idx_to = filter_products(product, descriptive_words['all_texts'].iloc[idx].values,
                                                             dataset2, idx_start, idx_to, dataset_start_index,
                                                             dataset2_no_price_products, descriptive_words)
        if len(data_subset_idx) == 0:
            print(f'No corresponding product for product "{product["name"]}" at index {idx}')
            data_subset_idx = []
        else:
            data_subset_idx = data_subset_idx.tolist()

        pairs_dataset_idx[idx] = data_subset_idx

    return pairs_dataset_idx


def filter_products(product, product_descriptive_words, dataset, idx_from, idx_to, dataset_start_index,
                    no_price_products, descriptive_words):
    """
    Filter products in dataset according to the price, category and word similarity to reduce number of comparisons
    @param product: given product for which we want to filter dataset
    @param product_descriptive_words: descriptive words of the product
    @param dataset:  dataset of products to be filtered sorted according to the price
    @param idx_from: starting index for searching for product with similar price in dataset
    @param idx_to: ending index for searching for product with similar price in dataset
    @param dataset_start_index: starting index to index the products from second dataset in descriptive words
    @param no_price_products: dataframe of products with no specified price
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return: Filtered dataset of products that are possibly the same as given product
    """
    if 'price' not in product.index.values or 'price' not in dataset:
        data_filtered = dataset
    else:
        last_price = dataset.iloc[idx_from]['price']

        min_price = product['price'] * MIN_MATCH_PRICE_RATIO
        while last_price < min_price and idx_from < len(dataset) - 1:
            idx_from += 1
            last_price = dataset.iloc[idx_from]['price']

        last_price = dataset.iloc[idx_to]['price']
        max_price = product['price'] * MAX_MATCH_PRICE_RATIO
        while last_price <= max_price and idx_to < len(dataset) - 1:
            idx_to += 1
            last_price = dataset.iloc[idx_to]['price']

        if idx_to == len(dataset) - 1:
            data_filtered = dataset.iloc[idx_from:]
        else:
            data_filtered = dataset.iloc[idx_from:idx_to]

        if not no_price_products.empty:
            data_filtered = pd.concat([data_filtered, no_price_products])

    '''
    if 'category' in product.index.values and 'category' in dataset:
        data_filtered = data_filtered[
            data_filtered['category'] == product['category'] or data_filtered['category'] is None]
    '''

    data_filtered = filter_products_with_no_similar_words(product, product_descriptive_words, data_filtered,
                                                          dataset_start_index, descriptive_words['all_texts'])
    return data_filtered.index.values, idx_from, idx_to


def filter_products_with_no_similar_words(product, product_descriptive_words, dataset, dataset_start_index,
                                          descriptive_words):
    """
    Filter products from the dataset of products with no same words and with low ratio of descriptive words
    @param product: product used as filter
    @param product_descriptive_words: descriptive words of the product
    @param dataset: dataset of products to be filtered
    @param dataset_start_index: starting index to index the products from second dataset in descriptive words
    @param descriptive_words: dictionary of descriptive words for each text column in products
    @return: dataset with products containing at least one same word as source product
    """
    data_subsets = []
    for idx, second_product in dataset.iterrows():
        second_product_descriptive_words = descriptive_words.iloc[idx + dataset_start_index].values
        descriptive_words_sim = compute_descriptive_words_similarity(
            product_descriptive_words,
            second_product_descriptive_words
        )

        codes_matching = True
        if "code" in product:
            if len(product["code"]) > 0 and len(second_product["code"]) > 0 \
                    and len(second_product["code"]) + len(product["code"]) > 2:
                matches = 0
                for code1 in product["code"]:
                    for code2 in second_product["code"]:
                        if code1 == code2:
                            matches += 1
                            break

                if matches == 0:
                    codes_matching = False

        # TODO delete
        codes_matching = True

        matching_words = len(set(product['name']) & set(second_product['name']))
        if codes_matching and (matching_words >= MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH) \
            and (len(product['name']) < 4 or len(second_product['name']) < 4 or matching_words >= MIN_LONG_PRODUCT_NAME_SIMILARITY_FOR_MATCH) \
            and descriptive_words_sim >= MIN_DESCRIPTIVE_WORDS_FOR_MATCH:
            data_subsets.append(second_product)

    return pd.DataFrame(data_subsets) if len(data_subsets) > 0 else pd.DataFrame(columns=dataset.columns.tolist())
