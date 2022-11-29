import json
import os
from math import ceil

import pandas as pd
from apify_client import ApifyClient

from product_mapping_engine.scripts.actor_model_interface import load_model_create_dataset_and_predict_matches
from product_mapping_engine.scripts.configuration import LOAD_PRECOMPUTED_MATCHES, SAVE_PRECOMPUTED_MATCHES

CHUNK_SIZE = 500
LAST_PROCESSED_CHUNK_KEY = 'last_processed_chunk'

def calculate_dataset_changes(dataset_output_attributes):
    columns = []
    renames = {}
    for old_column, new_column in dataset_output_attributes.items():
        columns.append(new_column)
        renames[old_column] = new_column

    return columns, renames


def output_results (
    output_dataset_client,
    default_kvs_client,
    predicted_matching_pairs,
    is_on_platform,
    current_chunk,
):
    output_data = predicted_matching_pairs[["id1", "url1", "id2", "url2", "predicted_scores"]]
    output_dataset_client.push_items(
        output_data.to_dict(orient='records')
    )

    print(f"Chunk {current_chunk} processed")
    print(f"Found {predicted_matching_pairs.shape[0]} matches")

    if is_on_platform:
        default_kvs_client.set_record(
            LAST_PROCESSED_CHUNK_KEY,
            current_chunk
        )

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    if not is_on_platform:
        czech_dataset = False
        if czech_dataset:
            default_kvs_client.set_record(
                'INPUT',
                {
                    "task_id": "Alpha-Complete-CZ",
                    "dataset_1": "9mk56pDWdfDZoCMiR",
                    "images_kvs_1": "iNNZxJhjAatupQSV0",
                    "dataset_2": "axCYJHLt6cmb1gbNJ",
                    "images_kvs_2": "NNZ40CQnWh4KofXJB",
                    "augment_outgoing_data": False
                }
            )
        else:
            default_kvs_client.set_record(
                'INPUT',
                {
                    "task_id": "fixed-v4-extra-xcite-mapping",
                    "dataset_1": "YRJd6DPu3Cbd9SrjZ",
                    "images_kvs_1": "OFXD6JAgZJ8XvFzfA",
                    "dataset_2": "lTVsLhiQXQoFIo52D",
                    "images_kvs_2": "SLsfIZYZjjHzoQNtb"
                }
            )

        # TODO delete
        default_kvs_client.set_record(
            'INPUT',
            {
                "task_id": "fixed-v4-extra-xcite-mapping",
                'pair_dataset': "OwYNOL0srq5urThEE"
            }
        )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    task_id = parameters['task_id']

    # Load precomputed matches
    dataset_precomputed_matches = None
    if LOAD_PRECOMPUTED_MATCHES:
        dataset_collection_client = client.datasets()
        precomputed_matches_collection = dataset_collection_client.get_or_create(
            name=task_id + '-precomputed-matches')
        precomputed_matches_client = client.dataset(precomputed_matches_collection['id'])
        dataset_precomputed_matches = pd.DataFrame(precomputed_matches_client.list_items().items)
        if not is_on_platform and os.path.isfile(task_id + '-precomputed-matches' + '.csv'):
            print(task_id + '-precomputed-matches' + '.csv')
            dataset_precomputed_matches = pd.read_csv(task_id + '-precomputed-matches' + '.csv')

    pair_dataset = None
    dataset1 = None
    dataset2 = None
    images_kvs_1_client = None
    images_kvs_2_client = None

    # Prepare storages and read data
    if 'pair_dataset' in parameters:
        pair_dataset_client = client.dataset(parameters['pair_dataset'])
        dataset_items = pair_dataset_client.list_items().items
        if dataset_items[0] == {}:
            dataset_items = dataset_items[1:]

        pair_dataset = pd.DataFrame(dataset_items)
        dataset_shape = pair_dataset.shape
        print(f"Working on dataset of shape: {dataset_shape[0]}x{dataset_shape[1]}")
    else:
        dataset_1_client = client.dataset(parameters['dataset_1'])
        dataset_2_client = client.dataset(parameters['dataset_2'])
        images_kvs_1_client = client.key_value_store(parameters['images_kvs_1'])
        images_kvs_2_client = client.key_value_store(parameters['images_kvs_2'])

        dataset1 = pd.DataFrame(dataset_1_client.list_items().items)
        dataset2 = pd.DataFrame(dataset_2_client.list_items().items)

        dataset1 = dataset1.drop_duplicates(subset=['url'], ignore_index=True)
        dataset2 = dataset2.drop_duplicates(subset=['url'], ignore_index=True)
        print(dataset1.shape)
        print(dataset2.shape)

    model_key_value_store_info = client.key_value_stores().get_or_create(
        name=task_id + '-product-mapping-model-output'
    )
    model_key_value_store = client.key_value_store(model_key_value_store_info['id'])

    first_chunk = 0
    if is_on_platform:
        start_from_chunk = default_kvs_client.get_record(LAST_PROCESSED_CHUNK_KEY)
        if start_from_chunk:
            first_chunk = start_from_chunk['value'] + 1

    if pair_dataset is not None:
        data_count = pair_dataset.shape[0]
    else:
        data_count = dataset1.shape[0]

    if not is_on_platform:
        CHUNK_SIZE = data_count

    #dataset_collection_client = client.datasets()
    #output_dataset_info = dataset_collection_client.get_or_create(name=f"{}")
    output_dataset_client = client.dataset(os.environ['APIFY_DEFAULT_DATASET_ID'])

    for current_chunk in range(first_chunk, ceil(data_count / CHUNK_SIZE)):
        if is_on_platform:
            print('Searching matches for products {}:{}'.format(current_chunk * CHUNK_SIZE,
                                                                (current_chunk + 1) * CHUNK_SIZE - 1))
            print('---------------------------------------------\n\n')

        pair_dataset_chunk = None
        dataset1_chunk = None

        if pair_dataset is not None:
            pair_dataset_chunk = pair_dataset.iloc[current_chunk * CHUNK_SIZE: (current_chunk + 1) * CHUNK_SIZE].reset_index()
        else:
            dataset1_chunk = dataset1.iloc[current_chunk * CHUNK_SIZE: (current_chunk + 1) * CHUNK_SIZE].reset_index()

        predicted_matching_pairs, all_product_pairs_matching_scores, new_product_pairs_matching_scores = \
            load_model_create_dataset_and_predict_matches(
                pair_dataset=pair_dataset_chunk,
                dataset1=dataset1_chunk,
                dataset2=dataset2,
                images_kvs1_client=images_kvs_1_client,
                images_kvs2_client=images_kvs_2_client,
                precomputed_pairs_matching_scores=dataset_precomputed_matches,
                model_key_value_store_client=model_key_value_store,
                task_id=task_id,
                is_on_platform=is_on_platform
            )

        all_product_pairs_matching_scores.to_csv("all_product_pairs_matching_scores.csv")

        if pair_dataset_chunk is not None:
            predicted_matching_pairs = predicted_matching_pairs.merge(pair_dataset_chunk, how='left', on=['id1', 'id2'])
        else:
            predicted_matching_pairs = predicted_matching_pairs.merge(dataset1_chunk.rename(columns={"id": "id1"}),
                                                                      on='id1', how='left') \
                .merge(dataset2.rename(columns={"id": "id2"}), on='id2', how='left', suffixes=('1', '2'))
            # TODO remove upon resolution
            predicted_matching_pairs = predicted_matching_pairs.drop_duplicates(subset=['url1', 'url2'])

        if SAVE_PRECOMPUTED_MATCHES:
            if not is_on_platform:
                if len(new_product_pairs_matching_scores) != 0:
                    new_product_pairs_matching_scores.to_csv(task_id + '-precomputed-matches' + '.csv', index=False)
            else:
                precomputed_matches_client.push_items(new_product_pairs_matching_scores.to_dict(orient='records'))

        # TODO investigate
        predicted_matching_pairs = predicted_matching_pairs[predicted_matching_pairs['url1'].notna()]
        predicted_matching_pairs.to_csv("predicted_matches.csv", index=False)

        output_results(
            output_dataset_client,
            default_kvs_client,
            predicted_matching_pairs,
            is_on_platform,
            current_chunk
        )

    aggregator_task_client = client.task(parameters["aggregator_task_id"])
    aggregator_task_client.start(task_input={
        "scrape_id": parameters["scrape_id"]
    })

    print("Done\n")
