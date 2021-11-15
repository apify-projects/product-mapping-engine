import os
import pandas as pd


def transform_scraped_datasets_to_full_pairs_dataset(
    url_pairs_dataset_path,
    scraped_dataset1_path,
    scraped_dataset2_path,
    full_pairs_dataset_path
):
    url_pairs_dataset = pd.read_csv(url_pairs_dataset_path)
    scraped_dataset1 = pd.read_csv(scraped_dataset1_path)
    scraped_dataset2 = pd.read_csv(scraped_dataset2_path)
    pairs_dataset = url_pairs_dataset.merge(scraped_dataset1, how='left', left_on='itemUrl', right_on='url')
    full_pairs_dataset = pairs_dataset.merge(scraped_dataset2, how='left', left_on='matchUrl', right_on='url')
    selected_columns_pairs_dataset = full_pairs_dataset
    selected_columns_pairs_dataset.to_csv(full_pairs_dataset_path, index=False)


transform_scraped_datasets_to_full_pairs_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "sampled-data.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "scraped-alza.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "scraped-mall.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "pairs_dataset.csv")
)