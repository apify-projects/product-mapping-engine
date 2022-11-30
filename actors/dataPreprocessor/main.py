import json
import os
from math import ceil

import pandas as pd
from apify_client import ApifyClient
from price_parser import Price

from product_mapping_engine.scripts.dataset_handler.dataset_upload.dataset_preprocessor import download_images, upload_images_to_kvs

def fix_price(price_string):
    price = Price.fromstring(price_string)
    price_amount = price.amount_float
    return price_amount

def fix_specification(specification):
    fixed_specification = []
    for parameter, value in specification.items():
        fixed_specification.append({
            'key': parameter,
            'value': value
        })

    return fixed_specification

def squishAttributesIntoAListAttribute(row):
    result = []
    for value in row:
        if value != "":
            result.append(value)

    return result

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    if not is_on_platform:
        # Set default input if not on platform
        default_kvs_client.set_record(
            'INPUT',
            {
                "run_executor": True,
                "task_id": "Extra-Xcite",
                "competitor_scrape_run_id": "tstIvNcnhKJ98XFjg",
                "executor_task_id": "aqX9cgYmSfpVWRGhg",
                'source': {
                    'eshop_name': "extra",
                    'dataset_id': 'J0MpblfbcF5jEQ0OI',
                    'attribute_names': {
                        'id': 'url',
                        'name': 'name',
                        # TODO extra dataset should contain brand
                        #'brand': 'brand',
                        'short_description': 'shortDescription',
                        'long_description': 'longDescription',
                        'specification': 'specification',
                        'image': 'images',
                        'price': 'price',
                        'url': 'url',
                        'code': ['productSpecificId', 'shopSpecificId']
                    },
                    'specification_format': 'correct',
                    'fix_prices': False,
                    'connecting_attribute': "url"
                },
                'target': {
                    'eshop_name': "xcite",
                    'dataset_id': 'g35C7bU89t4IvAlJP',
                    'attribute_names': {
                        'id': 'productUrl',
                        'name': 'name',
                        'brand': 'brand',
                        'short_description': 'shortDescription',
                        'long_description': 'longDescription',
                        'specification': 'specifications',
                        'image': 'images',
                        'price': 'price',
                        'url': 'productUrl',
                        'code': ['productSpecificId', 'shopSpecificId'],
                        'extraUrl': 'extraUrl'
                    },
                    'specification_format': 'parameter-value',
                    'fix_prices': True,
                    'source_connecting_attribute': "extraUrl"
                },
            }
        )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    competitor_scrape_run_client = client.run(parameters["competitor_scrape_run_id"])
    competitor_scrape_run_info = competitor_scrape_run_client.get()

    print(competitor_scrape_run_info)

    scraped_dataset_id = competitor_scrape_run_info["defaultDatasetId"]
    parameters["target"]["dataset_id"] = scraped_dataset_id

    scraper_kvs_client = client.key_value_store(competitor_scrape_run_info["defaultKeyValueStoreId"])
    scraper_input = scraper_kvs_client.get_record("INPUT")["value"]
    scrape_info_kvs_id = scraper_input["scrape_info_kvs_id"]
    competitor_name = scraper_input["competitor_name"]

    source_dataset = {}
    target_dataset = pd.DataFrame()
    for dataset in ['source', 'target']:
        dataset_parameters = parameters[dataset]
        # Read the dataset
        scraped_dataset_id = dataset_parameters['dataset_id']
        scraped_dataset_client = client.dataset(scraped_dataset_id)
        scraped_dataset = pd.DataFrame(scraped_dataset_client.list_items().items).fillna("") # PM system expects empty strings, not nulls

        # Change attributes of the dataset to the ones needed by the Product Mapping system
        partial_product_mapping_dataset = pd.DataFrame()
        attribute_names = dataset_parameters['attribute_names']
        for product_mapping_attribute, scraped_attribute in attribute_names.items():
            if type(scraped_attribute) is list:
                partial_product_mapping_dataset[product_mapping_attribute] = scraped_dataset[scraped_attribute].apply(squishAttributesIntoAListAttribute, axis=1)
            else:
                partial_product_mapping_dataset[product_mapping_attribute] = scraped_dataset[scraped_attribute]

        if dataset_parameters['specification_format'] == 'parameter-value':
            partial_product_mapping_dataset['specification'] = partial_product_mapping_dataset['specification'].apply(fix_specification)

        if dataset_parameters['fix_prices']:
            partial_product_mapping_dataset['price'] = partial_product_mapping_dataset['price'].apply(fix_price)

        # Download images
        '''
        product_mapping_dataset['image'] = download_images(product_mapping_dataset)
    
        target_kvs_name = 'PM-Prepro-Images-' + parameters['task_id'] + '-' + scraped_dataset_id
        target_kvs_id = client.key_value_stores().get_or_create(name=target_kvs_name)['id']
        target_kvs_client = client.key_value_store(target_kvs_id)
    
        upload_images_to_kvs(product_mapping_dataset, target_kvs_client)
        '''

        if dataset == 'target':
            target_dataset = partial_product_mapping_dataset
        else:
            connecting_attribute = parameters[dataset]['connecting_attribute']
            for index, product_row in partial_product_mapping_dataset.iterrows():
                product = dict(product_row)
                source_dataset[product[connecting_attribute]] = product

    target_connecting_attribute = parameters['target']['source_connecting_attribute']
    dataset_to_upload = []
    for index, target_product_row in target_dataset.iterrows():
        target_product = dict(target_product_row)
        corresponding_source_product = source_dataset[target_product[target_connecting_attribute]]
        del target_product[target_connecting_attribute]

        corresponding_source_product = {f"{key}1": val for key, val in corresponding_source_product.items()}
        target_product = {f"{key}2": val for key, val in target_product.items()}

        candidate_pair = {**corresponding_source_product, **target_product}
        dataset_to_upload.append(candidate_pair)

    '''
    # Save to file for debugging
    product_mapping_dataset.to_json('xcite_extra.json', orient='records')
    '''

    scrape_info_kvs_client = client.key_value_store(scrape_info_kvs_id)
    product_mapping_model_name = scrape_info_kvs_client.get_record("product_mapping_model_name")["value"]

    # Upload the preprocessed dataset
    preprocessed_dataset_name = 'PM-Prepro-Data-' + product_mapping_model_name + '-' + scraped_dataset_id
    preprocessed_dataset_id = client.datasets().get_or_create(name=preprocessed_dataset_name)['id']
    preprocessed_dataset_client = client.dataset(preprocessed_dataset_id)

    for e in range(ceil(len(dataset_to_upload) / 500)):
        chunk_to_upload = dataset_to_upload[e * 500: (e+1) * 500]
        preprocessed_dataset_client.push_items(chunk_to_upload)

    print(f"{len(dataset_to_upload)} items uploaded to dataset {preprocessed_dataset_id}")

    competitor_record = scrape_info_kvs_client.get_record(competitor_name)["value"]
    competitor_record["scraped_dataset_id"] = scraped_dataset_id
    competitor_record["preprocessed_dataset_id"] = preprocessed_dataset_id
    scrape_info_kvs_client.set_record(competitor_name, competitor_record)

    if parameters["run_executor"]:
        executor_task_client = client.task(parameters["executor_task_id"])
        executor_task_client.start(task_input={
            "scrape_info_kvs_id": scrape_info_kvs_id,
            "competitor_name": competitor_name
        })

