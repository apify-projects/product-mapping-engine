import cv2
import json
import numpy as np
import os
import pandas as pd
import requests
import imghdr

import importlib.machinery

configuration = importlib.machinery.SourceFileLoader(
    'configuration',
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..", "configuration.py"),
).load_module()


def is_url(potential_url):
    return "http" in potential_url or ".com" in potential_url


def download_images(images_path, pair_index, product_index, pair):
    """
    Download images for a product from pair of products from the dataset
    @param images_path: what folder to save the images to
    @param pair_index: index of the pair
    @param product_index: either 1 or 2, specifying which product from the pair should images be downloaded for
    @param pair: pair of products from the dataset
    @return: the amount of images that have been downloaded
    """
    downloaded_images = 0
    for potential_url in pair["image{}".format(product_index)]:
        if is_url(potential_url):
            try:
                response = requests.get(potential_url, timeout=5)
            except:
                print(response)
                continue

            print(response)
            if response.ok:
                if imghdr.what("", h=response.content) is None:
                    continue

                image_file_path = os.path.join(
                    images_path,
                    'pair_{}_product_{}_image_{}.jpg'.format(pair_index, product_index, downloaded_images + 1)
                )

                image = cv2.imdecode(np.asarray(bytearray(response.content), dtype="uint8"), cv2.IMREAD_COLOR)

                # decreasing the size of the images when needed to increase speed and preserve memory
                width_resize_ratio = configuration.IMAGE_RESIZE_WIDTH / image.shape[1]
                height_resize_ratio = configuration.IMAGE_RESIZE_HEIGHT / image.shape[0]
                resize_ratio = min(width_resize_ratio, height_resize_ratio)
                if resize_ratio < 1:
                    image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)

                cv2.imwrite(image_file_path, image)
                downloaded_images += 1

    return downloaded_images


def transform_scraped_datasets_to_full_pairs_dataset(
        url_pairs_dataset_paths,
        scraped_dataset1_paths,
        scraped_dataset2_paths,
        full_pairs_dataset_path,
        images_path
):
    """
    Uses scraped data to construct a labeled dataset than can be uploaded to the platform
    @param url_pairs_dataset_paths: array of datasets of pairs of product URLs along with information about their match status
    @param scraped_dataset1_paths: array of scraped source datasets
    @param scraped_dataset2_paths: array of scraped target datasets
    @param full_pairs_dataset_path: path to save the completed dataset to
    @param images_path: what folder to save the images to
    @return:
    """
    url_pairs_datasets = []
    for path in url_pairs_dataset_paths:
        url_pairs_datasets.append(pd.read_csv(path))
    url_pairs_dataset = pd.concat(url_pairs_datasets)

    url_pairs_dataset.info()

    scraped_datasets1 = []
    for path in scraped_dataset1_paths:
        scraped_datasets1.append(pd.read_json(path))
    scraped_dataset1 = pd.concat(scraped_datasets1)

    '''
    scraped_dataset1 = scraped_dataset1[
        ["name", "shortDescription", "longDescription", "specification", "images", "price", "url"]]
    '''
    scraped_dataset1 = scraped_dataset1.rename(
        columns={
            "images": "image",
            "shortDescription": "short_description",
            "longDescription": "long_description",
        }
    )

    scraped_dataset1 = scraped_dataset1.rename(
        columns={
            "name": "name1",
            "short_description": "short_description1",
            "long_description": "long_description1",
            "specification": "specification1",
            "image": "image1",
            "price": "price1",
            "url": "url1",
            "code": "code1"
        }
    )

    scraped_dataset1 = scraped_dataset1.astype({"price1": str}, errors='raise')
    scraped_dataset1["price1"] = scraped_dataset1["price1"].str.replace(".", ",", regex=False)
    scraped_dataset1["price1"] = scraped_dataset1["price1"].str.replace("[^0-9,]", "", regex=True)
    scraped_dataset1["price1"] = scraped_dataset1["price1"].str.replace(",", ".", regex=False)
    scraped_dataset1["id1"] = scraped_dataset1["url1"]

    scraped_datasets2 = []
    for path in scraped_dataset2_paths:
        scraped_datasets2.append(pd.read_json(path))
    scraped_dataset2 = pd.concat(scraped_datasets2)
    scraped_dataset2[["brand"]].drop_duplicates()["brand"].to_json("brands.json", orient="records")

    '''
    scraped_dataset2 = scraped_dataset2[["productName", "shortDescription", "longDescription", "specifications", "images", "price", "productUrl"]]
    '''
    '''scraped_dataset2 = scraped_dataset2.rename(
        columns={
            "shortDescription": "short_description",
            "longDescription": "long_description"
        }
    )'''

    scraped_dataset2 = scraped_dataset2.rename(
        columns={
            "name": "name2",
            "short_description": "short_description2",
            "long_description": "long_description2",
            "specification": "specification2",
            "image": "image2",
            "price": "price2",
            "url": "url2",
            "code": "code2"
        }
    )


    scraped_dataset2 = scraped_dataset2.astype({"price2": str}, errors='raise')
    scraped_dataset2["price2"] = scraped_dataset2["price2"].str.replace(".", ",", regex=False)
    scraped_dataset2["price2"] = scraped_dataset2["price2"].str.replace("[^0-9,]", "", regex=True)
    scraped_dataset2["price2"] = scraped_dataset2["price2"].str.replace(",", ".", regex=False)
    scraped_dataset2["id2"] = scraped_dataset2["url2"]

    pairs_dataset = url_pairs_dataset.merge(scraped_dataset1, how='inner', left_on='source_url', right_on='url1')
    full_pairs_dataset = pairs_dataset.merge(scraped_dataset2, how='inner', left_on='target_url', right_on='url2')

    full_pairs_dataset.info()

    full_pairs_dataset["match"] = full_pairs_dataset["match_type"].apply(
        lambda match_type: 1 if match_type == "match" else 0
    )
    full_pairs_dataset = full_pairs_dataset.filter(regex=("(.*(1|2)$)|^match$"), axis=1)
    #full_pairs_dataset = full_pairs_dataset[(full_pairs_dataset["price1"] != "") & (full_pairs_dataset["price2"] != "")]
    full_pairs_dataset.reset_index(inplace=True)

    full_pairs_dataset['specification1'] = full_pairs_dataset['specification1'].apply(
        lambda specification: json.dumps(specification)
    )

    full_pairs_dataset['specification2'] = full_pairs_dataset['specification2'].apply(
        lambda specification: json.dumps(specification)
    )

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for index, row in full_pairs_dataset.iterrows():
        print(row)
        print(index)
        full_pairs_dataset["image1"][index] = download_images(images_path, index, 1, row)
        full_pairs_dataset["image2"][index] = download_images(images_path, index, 2, row)

    full_pairs_dataset.to_csv(full_pairs_dataset_path, index=False)
    print(full_pairs_dataset.shape)


'''
transform_scraped_datasets_to_full_pairs_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files", "Hotovo - Televize - Karolína Bečvářová.xlsx - Sheet1.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data", "alza_televize.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data", "mall_televize.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "televize.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "televize_images"),
)
'''

'''
transform_scraped_datasets_to_full_pairs_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files", "aggregated.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data", "alza_aggregated.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data", "mall_aggregated.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "aggregated.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "aggregated_images"),
)
'''
'''
transform_scraped_datasets_to_full_pairs_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files",
                 "complete_cz.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "alza_complete_cz.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "mall_complete_cz.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "complete_cz.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "complete_cz_images"),
)
'''
'''
transform_scraped_datasets_to_full_pairs_dataset(
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files",
                     "amazon.csv"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files",
                     "xcite.csv"),
    ],
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "extra.json"),
    ],
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "xcite.json"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "amazon.json"),
    ],
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "extra_xcite_amazon.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "extra_xcite_amazon_images_finalized"),
)
'''


transform_scraped_datasets_to_full_pairs_dataset(
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "initial_files",
                     "xcite.csv"),
    ],
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "extra.json"),
    ],
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "scraped_data",
                 "xcite.json"),
    ],
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "extra_xcite.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "extra_xcite_images_finalized"),
)


