import json
import os
from math import ceil

import pandas as pd
from apify_client import ApifyClient

from product_mapping_engine.scripts.dataset_handler.dataset_upload.dataset_preprocessor import download_images, \
    upload_images_to_kvs

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
                "task_id": "FirstExtraExperiments",
                "dataset_to_preprocess": "smtu9PfcyVU4iYM62",
            }
        )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    # Read the dataset
    dataset_id = parameters['dataset_to_preprocess']
    dataset_client = client.dataset(dataset_id)
    dataset = pd.DataFrame(dataset_client.list_items().items)

    # Download images
    images = download_images(dataset)

    images_kvs_id = client.key_value_stores().get_or_create(name=task_id + "-" + dataset_id + '-image-kvs')['id']
    images_kvs_client = client.key_value_store(images_kvs_id)
    upload_images_to_kvs(images, kvs_client)

    # Change attributes of the dataset
    


    # Upload the changed dataset


