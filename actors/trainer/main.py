import os
import warnings

import pandas as pd
from apify_client import ApifyClient

from product_mapping_engine.scripts.actor_model_interface import load_data_and_train_model
from product_mapping_engine.scripts.configuration import DATA_FOLDER, TASK_ID, CLASSIFIER_TYPE

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"
    load_dataset_locally = True
    if not is_on_platform:
        czech_dataset = True
        if czech_dataset:
            default_kvs_client.set_record(
                'INPUT',
                {
                    "task_id": "full-cs-dataset",
                    "classifier_type": "LogisticRegression",
                    "dataset_id": "hnSwb2SaERXcvbXQ6",
                    "images_kvs_1": "dEoB1XWso0B0cY6AC",
                    "images_kvs_2": "lBhezRArqcep8rMER"
                }
            )
        else:
            default_kvs_client.set_record(
                'INPUT',
                {
                    "task_id": "fixed-v4-extra-xcite-mapping",
                    "classifier_type": "LogisticRegression",
                    "dataset_id": "TyXf5pvH3eg7AES8g",
                    "images_kvs_1": "OFXD6JAgZJ8XvFzfA",
                    "images_kvs_2": "SLsfIZYZjjHzoQNtb"
                }
            )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    # print('Actor input:')
    # print(json.dumps(parameters, indent=2))

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

    if load_dataset_locally:
        task_id = TASK_ID
        classifier_type = CLASSIFIER_TYPE
        print('Task id: ' + task_id)
        print('Classifier type: ' + classifier_type)
        try:
            labeled_dataset = pd.read_csv(f'{DATA_FOLDER}/{task_id}.csv')
        except:
            warnings.warn('No source dataset found')
            labeled_dataset = pd.DataFrame()
        labeled_dataset = labeled_dataset.fillna('')

    # labeled_dataset = labeled_dataset.drop(columns={'match_type','image_url1','image_url2', 'category'})
    stats = load_data_and_train_model(
        classifier_type,
        dataset_dataframe=labeled_dataset,
        images_kvs1_client=images_kvs_1_client,
        images_kvs2_client=images_kvs_2_client,
        output_key_value_store_client=output_key_value_store_client,
        task_id=task_id,
        is_on_platform=is_on_platform
    )
    if stats is not None:
        output_key_value_store_client.set_record('stats', stats)
