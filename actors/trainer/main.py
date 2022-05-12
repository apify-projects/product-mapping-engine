import json
import os

import pandas as pd
from apify_client import ApifyClient

from product_mapping_engine.scripts.actor_model_interface import load_data_and_train_model

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    if not is_on_platform:
        full_dataset = False
        if full_dataset:
            default_kvs_client.set_record(
                'INPUT',
                {
                    "task_id": "Alpha-Complete-CZ",
                    "classifier_type": "LogisticRegression",
                    "dataset_id": "cfKxr20fm88KfhBDg",
                    "images_kvs_1": "iNNZxJhjAatupQSV0",
                    "images_kvs_2": "NNZ40CQnWh4KofXJB"
                }
            )
        else:
            default_kvs_client.set_record(
                    'INPUT',
                    {
                        "task_id": "Alpha-Complete-CZ-middle-sample",
                        "classifier_type": "LogisticRegression",
                        "dataset_id": "bI50cdPwQXmcvhfcZ",
                        "images_kvs_1": "coKhaQXSyj2kPqzyM",
                        "images_kvs_2": "wb001Kh9JAGpFvpaG"
                     }
            )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    task_id = parameters['task_id']
    classifier_type = parameters['classifier_type']

    # Prepare storages and read data
    labeled_dataset_client = client.dataset(parameters['dataset_id'])
    images_kvs_1_client = client.key_value_store(parameters['images_kvs_1'])
    images_kvs_2_client = client.key_value_store(parameters['images_kvs_2'])
    output_key_value_store_info = client.key_value_stores().get_or_create(
        name=task_id + '-product-mapping-model-output'
    )
    output_key_value_store_client = client.key_value_store(output_key_value_store_info['id'])
    output_key_value_store_client.set_record('parameters', parameters)
    labeled_dataset = pd.DataFrame(labeled_dataset_client.list_items().items)

    stats = load_data_and_train_model(
        classifier_type,
        dataset_dataframe=labeled_dataset,
        images_kvs1_client=images_kvs_1_client,
        images_kvs2_client=images_kvs_2_client,
        output_key_value_store_client=output_key_value_store_client,
        task_id=task_id,
        is_on_platform=is_on_platform
    )
    output_key_value_store_client.set_record('stats', stats)
