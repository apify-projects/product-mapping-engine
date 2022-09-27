import json
import os
from math import ceil

import pandas as pd
from apify_client import ApifyClient

from product_mapping_engine.scripts.actor_model_interface import load_model_create_dataset_and_predict_matches
from product_mapping_engine.scripts.configuration import LOAD_PRECOMPUTED_MATCHES, SAVE_PRECOMPUTED_MATCHES

CHUNK_SIZE = 1000
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
    dataset1_output_attributes,
    dataset2_output_attributes,
    augment_outgoing_data,
    augmenting_dataset1,
    augmenting_dataset2
):
    if augment_outgoing_data:
        output_data = predicted_matching_pairs[['id1', 'id2']]
        output_data = output_data.merge(augmenting_dataset1, how='left', left_on="id1", right_on="url1")
        augmenting_dataset2.info()
        # TODO parametrize instead of "productUrl2"
        output_data = output_data.merge(augmenting_dataset2, how='left', left_on="id2", right_on="productUrl2")

        columns1, renames1 = calculate_dataset_changes(dataset1_output_attributes)
        output_data = output_data.rename(columns=renames1)
        columns2, renames2 = calculate_dataset_changes(dataset2_output_attributes)
        output_data = output_data.rename(columns=renames2)

        output_data = output_data[columns1 + columns2]
    else:
        output_data = predicted_matching_pairs

    output_dataset_client.push_items(
        output_data.to_dict(orient='records')
    )

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
                    "images_kvs_2": "SLsfIZYZjjHzoQNtb",
                    "dataset_1_output_attributes": {
                        "shopSpecificId1": "extraSku",
                        "url1": "extraUrl"
                    },
                    "dataset_2_output_attributes": {
                        "SKU2": "competitorSku",
                        "productUrl2": "competitorUrl",
                        "competitor2": "competitor"
                    },
                    "augment_outgoing_data": True,
                    "augmented_dataset_1": "J0MpblfbcF5jEQ0OI",
                    "augmented_dataset_2": "R2yxBEbgCOA8hyO9u",
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

    # Prepare storages and read data
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

    default_dataset_client = client.dataset(os.environ['APIFY_DEFAULT_DATASET_ID'])

    augment_outgoing_data = parameters['augment_outgoing_data']
    if augment_outgoing_data:
        augmented_dataset1 = pd.DataFrame(
            client.dataset(parameters['augmented_dataset_1']).list_items().items
        ).add_suffix("1")

        augmented_dataset2 = pd.DataFrame(
            client.dataset(parameters['augmented_dataset_2']).list_items().items
        ).add_suffix("2")
    else:
        augmented_dataset1 = augmented_dataset2 = None

    first_chunk = 0
    if is_on_platform:
        start_from_chunk = default_kvs_client.get_record(LAST_PROCESSED_CHUNK_KEY)
        if start_from_chunk:
            first_chunk = start_from_chunk['value'] + 1

    data_count = dataset1.shape[0]
    if not is_on_platform:
        CHUNK_SIZE = data_count

    for current_chunk in range(first_chunk, ceil(data_count / CHUNK_SIZE)):
        if is_on_platform:
            print('Searching matches for products {}:{}'.format(current_chunk * CHUNK_SIZE + 1,
                                                                (current_chunk + 1) * CHUNK_SIZE))
            print('---------------------------------------------\n\n')

        dataset1_chunk = dataset1.iloc[current_chunk * CHUNK_SIZE: (current_chunk + 1) * CHUNK_SIZE]

        predicted_matching_pairs, all_product_pairs_matching_scores, new_product_pairs_matching_scores = \
            load_model_create_dataset_and_predict_matches(
                dataset1_chunk,
                dataset2,
                dataset_precomputed_matches,
                images_kvs_1_client,
                images_kvs_2_client,
                model_key_value_store_client=model_key_value_store,
                task_id=task_id,
                is_on_platform=is_on_platform
            )

        all_product_pairs_matching_scores.to_csv("all_product_pairs_matching_scores.csv")

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

        dataset_collection_client = client.datasets()
        apify_dataset_info = dataset_collection_client.get_or_create(name="sample-pm-results")
        apify_dataset_client = client.dataset(apify_dataset_info['id'])

        output_results(
            apify_dataset_client,
            default_kvs_client,
            predicted_matching_pairs,
            is_on_platform,
            current_chunk,
            parameters["dataset_1_output_attributes"],
            parameters["dataset_2_output_attributes"],
            augment_outgoing_data,
            augmented_dataset1,
            augmented_dataset2
        )

        print("Done\n")
