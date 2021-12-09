import os
import pandas as pd
import requests
import imghdr

IMAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/images")

def is_url(potential_url):
    return "http" in potential_url or ".com" in potential_url

def download_images(pair_index, product_index, pair):
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
                    IMAGE_DIRECTORY,
                    'pair_{}_product_{}_image_{}'.format(pair_index, product_index, downloaded_images + 1)
                )
                with open(image_file_path, 'wb') as image_file:
                    image_file.write(response.content)

                downloaded_images += 1

    return downloaded_images


def transform_scraped_datasets_to_full_pairs_dataset(
    url_pairs_dataset_path,
    scraped_dataset1_path,
    scraped_dataset2_path,
    full_pairs_dataset_path
):
    '''
    Alza:
    name = name
    short_description = shortDescription
    long_description = longDescription
    specification = parameters
    images = images
    price = price
    url = url
    '''

    '''
    Mall:
    name = name
    short_description = shortDescription
    long_description = longDescription
    specification = params
    images = images
    price = price
    url = url
    '''
    url_pairs_dataset = pd.read_csv(url_pairs_dataset_path)

    scraped_dataset1 = pd.read_json(scraped_dataset1_path)
    scraped_dataset1 = scraped_dataset1[["name", "shortDescription", "longDescription", "parameters", "images", "price", "url"]]
    scraped_dataset1 = scraped_dataset1.rename(
        columns = {
            "shortDescription": "short_description",
            "longDescription": "long_description",
            "parameters": "specification",
            "images": "image"
        }
    )
    scraped_dataset1 = scraped_dataset1.rename(
        columns = {
            "name": "name1",
            "short_description": "short_description1",
            "long_description": "long_description1",
            "specification": "specification1",
            "image": "image1",
            "price": "price1",
            "url": "url1"
        }
    )
    scraped_dataset1["price1"] = scraped_dataset1["price1"].str.replace("[^0-9]", "", regex=True)

    scraped_dataset2 = pd.read_json(scraped_dataset2_path)
    scraped_dataset2 = scraped_dataset2[["name", "shortDescription", "longDescription", "params", "images", "price", "url"]]
    scraped_dataset2 = scraped_dataset2.rename(
        columns = {
            "shortDescription": "short_description",
            "longDescription": "long_description",
            "params": "specification",
            "images": "image"
        }
    )
    scraped_dataset2 = scraped_dataset2.rename(
        columns = {
            "name": "name2",
            "short_description": "short_description2",
            "long_description": "long_description2",
            "specification": "specification2",
            "image": "image2",
            "price": "price2",
            "url": "url2"
        }
    )
    scraped_dataset2["price2"] = scraped_dataset2["price2"].str.replace("[^0-9]", "", regex=True)

    pairs_dataset = url_pairs_dataset.merge(scraped_dataset1, how='inner', left_on='itemUrl', right_on='url1')
    full_pairs_dataset = pairs_dataset.merge(scraped_dataset2, how='inner', left_on='matchUrl', right_on='url2')
    full_pairs_dataset = full_pairs_dataset.rename(
        columns = { "match_status": "match" }
    )
    full_pairs_dataset = full_pairs_dataset.filter(regex=("(.*(1|2)$)|^match$"), axis=1)
    full_pairs_dataset = full_pairs_dataset[(full_pairs_dataset["price1"] != "") & (full_pairs_dataset["price2"] != "")]
    full_pairs_dataset.reset_index(inplace=True)
    print(full_pairs_dataset["price1"])
    print(full_pairs_dataset["price2"])
    print(full_pairs_dataset)
    print(full_pairs_dataset.columns)
    #selected_columns_pairs_dataset = full_pairs_dataset
    #selected_columns_pairs_dataset.to_csv(full_pairs_dataset_path, index=False)
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    for index, row in full_pairs_dataset.iterrows():
        print(row)
        print(index)
        full_pairs_dataset["image1"][index] = download_images(index, 1, row)
        full_pairs_dataset["image2"][index] = download_images(index, 2, row)

    print(full_pairs_dataset["image1"])
    full_pairs_dataset.to_csv(full_pairs_dataset_path, index=False)


transform_scraped_datasets_to_full_pairs_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/sampled_data.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/scraped_alza.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/scraped_mall.json"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/pairs_dataset.csv")
)