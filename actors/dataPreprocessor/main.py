import json
import os
from math import ceil

import pandas as pd
from apify_client import ApifyClient
from price_parser import Price

from product_mapping_engine.scripts.dataset_handler.dataset_upload.dataset_preprocessor import download_images, \
    upload_images_to_kvs

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
                'task_id': 'Final-Xcite',
                'dataset_to_preprocess': 'R2yxBEbgCOA8hyO9u',
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
                    'code': ['productSpecificId', 'shopSpecificId']
                },
                'specification_format': 'parameter-value'
            }
        )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    # Read the dataset
    scraped_dataset_id = parameters['dataset_to_preprocess']
    scraped_dataset_client = client.dataset(scraped_dataset_id)
    scraped_dataset = pd.DataFrame(scraped_dataset_client.list_items().items).fillna("") # PM system expects empty strings, not nulls
    print(scraped_dataset.info())

    # Change attributes of the dataset to the ones needed by the Product Mapping system
    product_mapping_dataset = pd.DataFrame()
    attribute_names = parameters['attribute_names']
    for product_mapping_attribute, scraped_attribute in attribute_names.items():
        if type(scraped_attribute) is list:
            product_mapping_dataset[product_mapping_attribute] = scraped_dataset[scraped_attribute].apply(squishAttributesIntoAListAttribute, axis=1)
        else:
            product_mapping_dataset[product_mapping_attribute] = scraped_dataset[scraped_attribute]

    if parameters['specification_format'] == 'parameter-value':
        product_mapping_dataset['specification'] = product_mapping_dataset['specification'].apply(fix_specification)

    product_mapping_dataset['price'] = product_mapping_dataset['price'].apply(fix_price)

    # Download images
    '''
    product_mapping_dataset['image'] = download_images(product_mapping_dataset)

    target_kvs_name = 'PM-Prepro-Images-' + parameters['task_id'] + '-' + scraped_dataset_id
    target_kvs_id = client.key_value_stores().get_or_create(name=target_kvs_name)['id']
    target_kvs_client = client.key_value_store(target_kvs_id)

    upload_images_to_kvs(product_mapping_dataset, target_kvs_client)
    '''

    # Save to file for debugging
    product_mapping_dataset.to_json('finalizedXcite.json', orient='records')

    # Upload the changed dataset
    target_dataset_name = 'PM-Prepro-Data-' + parameters['task_id'] + '-' + scraped_dataset_id
    target_dataset_id = client.datasets().get_or_create(name=target_dataset_name)['id']
    target_dataset_client = client.dataset(target_dataset_id)
    target_dataset_client.push_items(product_mapping_dataset.to_dict('records'))
