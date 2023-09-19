import json
import os
from math import ceil, sqrt

import pandas as pd
from apify_client import ApifyClient

# A hack dealing with "attempted relative import with no known parent package" when executor is run directly,
# compared to when it is run from the public actor.
if __name__ == '__main__':
    from product_mapping_engine.scripts.actor_model_interface import load_model_create_dataset_and_predict_matches
    from product_mapping_engine.scripts.configuration import LOAD_PRECOMPUTED_MATCHES, SAVE_PRECOMPUTED_MATCHES
else:
    from .product_mapping_engine.scripts.actor_model_interface import load_model_create_dataset_and_predict_matches
    from .product_mapping_engine.scripts.configuration import LOAD_PRECOMPUTED_MATCHES, SAVE_PRECOMPUTED_MATCHES

CHUNK_SIZE = 500
LAST_PROCESSED_CHUNK_KEY = 'last_processed_chunk'

def squishAttributesIntoAListAttribute(row):
    result = []
    for value in row:
        if value != "" and value is not None:
            result.append(value)

    return result

def identify_id_attributes_from_input_mapping(input_mapping):
    return input_mapping["eshop1"]["id"], input_mapping["eshop2"]["id"]

def calculate_dataset_changes(dataset, attributes_mapping, dataset_postfix=None):
    columns_found = set()
    columns = []
    renames = {}
    encountered_old_columns = {}
    for new_column, old_column in attributes_mapping.items():
        new_column_with_postfix = f"{new_column}{dataset_postfix}" if dataset_postfix is not None else new_column
        columns.append(new_column_with_postfix)
        columns_found.add(new_column)
        if type(old_column) == list:
            dataset[new_column_with_postfix] = dataset[old_column].apply(squishAttributesIntoAListAttribute, axis=1)
        else:
            old_column_name = old_column
            if old_column not in encountered_old_columns:
                encountered_old_columns[old_column] = 1
            else:
                encountered_old_columns[old_column] += 1
                old_column_name = f"{old_column}_{encountered_old_columns[old_column]}"
                dataset[old_column_name] = dataset[old_column].copy()

            renames[old_column_name] = new_column_with_postfix

    omittable_columns = [
        "name",
        "price",
        "short_description",
        "long_description",
        "specification"
    ]

    default_values = {
        "name": "column_did_not_exist_in_scraper",
        "price": None,
        "short_description": "column_did_not_exist_in_scraper",
        "long_description": "column_did_not_exist_in_scraper",
        "specification": [[] for _ in range(dataset.shape[0])]
    }

    # Needed to make sure that the system works even if some attributes are omitted
    for omittable_column in omittable_columns:
        if omittable_column not in columns_found:
            print(f"Adding {omittable_column}")
            new_column_with_postfix = f"{omittable_column}{dataset_postfix}" if dataset_postfix is not None else omittable_column
            columns.append(new_column_with_postfix)
            dataset[new_column_with_postfix] = default_values[omittable_column]

    return dataset, columns, renames

def perform_input_mapping(input_mapping, pair_dataset=None, singular_dataset=None, singular_dataset_index=None):
    if pair_dataset is not None:
        all_columns = []
        for e in range(2):
            eshop_mapping = input_mapping[f"eshop{e+1}"]
            pair_dataset, columns, renames = calculate_dataset_changes(pair_dataset, eshop_mapping, e+1)
            pair_dataset = pair_dataset.rename(columns=renames)
            all_columns.extend(columns)

        return pair_dataset[all_columns]
    else:
        eshop_mapping = input_mapping[f"eshop{singular_dataset_index}"]
        singular_dataset, columns, renames = calculate_dataset_changes(singular_dataset, eshop_mapping)
        singular_dataset = singular_dataset.rename(columns=renames)
        return singular_dataset[columns]

def output_results (
    output_dataset_client,
    default_kvs_client,
    predicted_matching_pairs,
    rejected_pairs,
    is_on_platform,
    current_chunk,
    output_mapping,
    id_attribute1,
    id_attribute2,
    original_pair_dataset=None,
    original_dataset1=None,
    original_dataset2=None
):
    output_data = predicted_matching_pairs[["id1", "id2", "predicted_scores"]]
    output_data["predicted_match"] = 1
    output_data = pd.concat([
        output_data,
        rejected_pairs[["id1", "id2", "predicted_scores", "predicted_match"]]
    ], ignore_index=True)

    if output_mapping:
        if original_pair_dataset is not None:
            raw_output_data_df = output_data[["id1", "id2"]].merge(original_pair_dataset, left_on=["id1", "id2"], right_on=[id_attribute1, id_attribute2], suffixes=("_pm_attr", None))
        else:
            raw_output_data_dfs = []
            raw_output_data_dfs.append(output_data[["id1"]].merge(original_dataset1, how="left", left_on="id1", right_on=id_attribute1, suffixes=("_pm_attr", None)))
            raw_output_data_dfs.append(output_data[["id2"]].merge(original_dataset2, how="left", left_on="id2", right_on=id_attribute2, suffixes=("_pm_attr", None)))

        output_data = output_data[["predicted_scores", "predicted_match"]]

        for e in range(2):
            eshop_mapping = output_mapping[f"eshop{e+1}"]
            raw_output_data = raw_output_data_dfs[e] if original_pair_dataset is None else raw_output_data_df
            for output_attribute, original_attribute in eshop_mapping.items():
                output_data[output_attribute] = raw_output_data[original_attribute]

    output_dataset_client.push_items(
        output_data.to_dict(orient='records')
    )

    if is_on_platform:
        default_kvs_client.set_record(
            LAST_PROCESSED_CHUNK_KEY,
            current_chunk
        )

def assemble_dataset_from_partial_dataset_ids(data_client, partial_dataset_ids):
    partial_datasets = []
    for partial_dataset_id in partial_dataset_ids:
        partial_dataset_client = data_client.dataset(partial_dataset_id)
        dataset_items = partial_dataset_client.list_items(clean=True).items
        if dataset_items[0] == {}:
            dataset_items = dataset_items[1:]

        partial_datasets.append(pd.DataFrame(dataset_items))

    return pd.concat(partial_datasets)

def perform_mapping (
    parameters,
    output_dataset_client,
    default_kvs_client,
    data_client,
    is_on_platform,
    task_id,
    return_all_considered_pairs=False,
    max_items_to_process=None
):
    # Load precomputed matches
    dataset_precomputed_matches = None
    if LOAD_PRECOMPUTED_MATCHES:
        dataset_collection_client = data_client.datasets()
        precomputed_matches_collection = dataset_collection_client.get_or_create(
            name=task_id + '-precomputed-matches')
        precomputed_matches_client = data_client.dataset(precomputed_matches_collection['id'])
        dataset_precomputed_matches = pd.DataFrame(precomputed_matches_client.list_items().items)
        if not is_on_platform and os.path.isfile(task_id + '-precomputed-matches' + '.csv'):
            print(task_id + '-precomputed-matches' + '.csv')
            dataset_precomputed_matches = pd.read_csv(task_id + '-precomputed-matches' + '.csv')

    pair_dataset = None
    dataset1 = None
    dataset2 = None
    images_kvs_1_client = None
    images_kvs_2_client = None
    input_mapping = parameters.get("input_mapping")
    output_mapping = parameters.get("output_mapping")
    original_pair_dataset = None
    original_dataset1 = None
    original_dataset2 = None

    global CHUNK_SIZE

    # Prepare storages and read data
    if parameters.get('pair_dataset_ids'):
        pair_dataset_ids = parameters['pair_dataset_ids']
        pair_dataset = assemble_dataset_from_partial_dataset_ids(data_client, pair_dataset_ids)

        dataset_shape = pair_dataset.shape
        print(f"Working on dataset of shape: {dataset_shape[0]}x{dataset_shape[1]}")

        if max_items_to_process is not None:
            pair_dataset = pair_dataset.head(max_items_to_process)
            dataset_shape = pair_dataset.shape
            print(f"Restricted to shape (due to the maximum amount of items to process): {dataset_shape[0]}x{dataset_shape[1]}")

        original_pair_dataset = pair_dataset

        CHUNK_SIZE = 500
    else:
        # TODO deal with this (there are multiple dataset ids, but only one images kvs)
        images_kvs_1_client = None
        images_kvs_2_client = None
        if parameters.get('images_kvs_1') and parameters.get('images_kvs_2'):
            images_kvs_1_client = data_client.key_value_store(parameters['images_kvs_1'])
            images_kvs_2_client = data_client.key_value_store(parameters['images_kvs_2'])

        dataset1 = assemble_dataset_from_partial_dataset_ids(data_client, parameters["dataset1_ids"])
        dataset2 = assemble_dataset_from_partial_dataset_ids(data_client, parameters["dataset2_ids"])

        original_dataset1 = dataset1
        original_dataset2 = dataset2

        if input_mapping:
            dataset1 = perform_input_mapping(input_mapping, singular_dataset=dataset1, singular_dataset_index=1)
            dataset2 = perform_input_mapping(input_mapping, singular_dataset=dataset2, singular_dataset_index=2)

        if not return_all_considered_pairs:
            dataset1 = dataset1.drop_duplicates(subset=['id'], ignore_index=True)
            dataset2 = dataset2.drop_duplicates(subset=['id'], ignore_index=True)

        print()
        print(f"Working on two datasets:")
        print(f"Dataset 1 of shape: {dataset1.shape[0]}x{dataset1.shape[1]}")
        print(f"Dataset 2 of shape: {dataset2.shape[0]}x{dataset2.shape[1]}")
        print()

        if max_items_to_process is not None:
            dataset1_rows = dataset1.shape[0]
            dataset2_rows = dataset2.shape[0]

            dataset1_rows_needed = dataset2_rows_needed = ceil(sqrt(max_items_to_process))
            if dataset1_rows < dataset1_rows_needed:
                dataset2_rows_needed = ceil(max_items_to_process / dataset1_rows)
            elif dataset2_rows < dataset2_rows_needed:
                dataset1_rows_needed = ceil(max_items_to_process / dataset2_rows)

            dataset1 = dataset1.head(dataset1_rows_needed)
            dataset2 = dataset2.head(dataset2_rows_needed)

            print()
            print(f"Restricted the datasets to shape (due to the maximum amount of items to process):")
            print(f"Dataset 1 of shape: {dataset1.shape[0]}x{dataset1.shape[1]}")
            print(f"Dataset 2 of shape: {dataset2.shape[0]}x{dataset2.shape[1]}")
            print()

        CHUNK_SIZE = 50
    if not task_id.startswith("__local__"):
        model_key_value_store_info = data_client.key_value_stores().get_or_create(
            name=task_id + '-product-mapping-model-output'
        )
        model_key_value_store = data_client.key_value_store(model_key_value_store_info['id'])
    else:
        model_key_value_store = None
        if pair_dataset is not None:
            if "code1" in pair_dataset and "code2" in pair_dataset:
                task_id += "_codes"
            else:
                task_id += "_no_codes"
        else:
            if "code" in dataset1 and "code" in dataset2:
                task_id += "_codes"
            else:
                task_id += "_no_codes"

    first_chunk = 0
    # TODO fix
    if False and is_on_platform:
        start_from_chunk = default_kvs_client.get_record(LAST_PROCESSED_CHUNK_KEY)
        if start_from_chunk:
            first_chunk = start_from_chunk['value'] + 1

    if pair_dataset is not None:
        data_count = pair_dataset.shape[0]
        chunk_count = ceil(data_count / CHUNK_SIZE)
        chunks = [(chunk_number * CHUNK_SIZE, (chunk_number + 1) * CHUNK_SIZE) for chunk_number in range(chunk_count)]
    else:
        dataset1_chunk_count = ceil(dataset1.shape[0] / CHUNK_SIZE)
        dataset2_chunk_count = ceil(dataset2.shape[0] / CHUNK_SIZE)
        chunk_count = dataset1_chunk_count * dataset2_chunk_count
        chunks = []
        for dataset1_chunk_number in range(dataset1_chunk_count):
            for dataset2_chunk_number in range(dataset2_chunk_count):
                chunks.append((
                    (dataset1_chunk_number * CHUNK_SIZE, (dataset1_chunk_number + 1) * CHUNK_SIZE),
                    (dataset2_chunk_number * CHUNK_SIZE, (dataset2_chunk_number + 1) * CHUNK_SIZE)
                ))

    if not is_on_platform:
        CHUNK_SIZE = data_count

    for current_chunk in range(first_chunk, chunk_count):
        pair_dataset_chunk = None
        dataset1_chunk = None
        dataset2_chunk = None

        print('\n\n------------------------------------------------------------------------------------------')

        if pair_dataset is not None:
            if is_on_platform:
                print('Searching matches for products {}:{}'.format(chunks[current_chunk][0],
                                                                    chunks[current_chunk][1] - 1))

            pair_dataset_chunk = pair_dataset.iloc[
                chunks[current_chunk][0]: chunks[current_chunk][1]
            ].reset_index()

            if input_mapping:
                pair_dataset_chunk = perform_input_mapping(input_mapping, pair_dataset=pair_dataset_chunk)
        else:
            if is_on_platform:
                print('Searching matches for products {}:{} from dataset1 and {}:{} from dataset2'.format(
                    chunks[current_chunk][0][0],
                    chunks[current_chunk][0][1] - 1,
                    chunks[current_chunk][1][0],
                    chunks[current_chunk][1][1] - 1
                ))

            dataset1_chunk = dataset1.iloc[chunks[current_chunk][0][0]: chunks[current_chunk][0][1]].reset_index()
            dataset2_chunk = dataset2.iloc[chunks[current_chunk][1][0]: chunks[current_chunk][1][1]].reset_index()

        print('------------------------------------------------------------------------------------------\n\n')

        predicted_matching_pairs, rejected_pairs, all_product_pairs_matching_scores, new_product_pairs_matching_scores = \
            load_model_create_dataset_and_predict_matches(
                pair_dataset=pair_dataset_chunk,
                dataset1=dataset1_chunk,
                dataset2=dataset2_chunk,
                images_kvs1_client=images_kvs_1_client,
                images_kvs2_client=images_kvs_2_client,
                precomputed_pairs_matching_scores=dataset_precomputed_matches,
                model_key_value_store_client=model_key_value_store,
                task_id=task_id,
                is_on_platform=is_on_platform,
                return_all_considered_pairs=return_all_considered_pairs
            )

        matching_pairs_count = 0
        if predicted_matching_pairs is not None:
            all_product_pairs_matching_scores.to_csv("all_product_pairs_matching_scores.csv")

            if pair_dataset_chunk is not None:
                predicted_matching_pairs = predicted_matching_pairs.merge(pair_dataset_chunk, how='left', on=['id1', 'id2'])
            else:
                print(predicted_matching_pairs)
                print(dataset1_chunk)
                predicted_matching_pairs = predicted_matching_pairs.merge(dataset1_chunk.rename(columns={"id": "id1"}),
                                                                          on='id1', how='left') \
                    .merge(dataset2_chunk.rename(columns={"id": "id2"}), on='id2', how='left', suffixes=('1', '2'))

            if not return_all_considered_pairs:
                # TODO remove upon resolution
                predicted_matching_pairs = predicted_matching_pairs.drop_duplicates(subset=['id1', 'id2'])

            if SAVE_PRECOMPUTED_MATCHES:
                if not is_on_platform:
                    if len(new_product_pairs_matching_scores) != 0:
                        new_product_pairs_matching_scores.to_csv(task_id + '-precomputed-matches' + '.csv', index=False)
                else:
                    precomputed_matches_client.push_items(new_product_pairs_matching_scores.to_dict(orient='records'))

            # TODO investigate
            predicted_matching_pairs = predicted_matching_pairs[predicted_matching_pairs['id1'].notna()]

            if input_mapping:
                original_id_attributes = identify_id_attributes_from_input_mapping(input_mapping)

            output_results(
                output_dataset_client,
                default_kvs_client,
                predicted_matching_pairs,
                rejected_pairs,
                is_on_platform,
                current_chunk,
                output_mapping,
                original_id_attributes[0] if input_mapping else None,
                original_id_attributes[1] if input_mapping else None,
                original_pair_dataset,
                original_dataset1,
                original_dataset2,
            )

            matching_pairs_count = predicted_matching_pairs.shape[0]

        print(f"Chunk {current_chunk} processed")
        print(f"Found {matching_pairs_count} matches")


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
                "aggregator_task_id": "QuaYjF9KbNyVhLLfV",
                "competitor_name": "eddy",
                "scrape_info_kvs_id": "wcEII0IJOf1pBiBMr"
            }
        )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    output_dataset_id = os.environ['APIFY_DEFAULT_DATASET_ID']
    output_dataset_client = client.dataset(output_dataset_id)

    # TODO delete
    if "different_user_token" in parameters:
        client = ApifyClient(parameters["different_user_token"], api_url=os.environ['APIFY_API_BASE_URL'])

    scrape_info_kvs_id = parameters["scrape_info_kvs_id"]
    scrape_info_kvs_client = client.key_value_store(scrape_info_kvs_id)

    task_id = scrape_info_kvs_client.get_record("product_mapping_model_name")["value"]

    competitor_name = parameters["competitor_name"]
    competitor_record = scrape_info_kvs_client.get_record(competitor_name)["value"]

    parameters["pair_dataset_ids"] = [competitor_record["preprocessed_dataset_id"]]

    perform_mapping(
        parameters,
        output_dataset_client,
        default_kvs_client,
        client,
        is_on_platform,
        task_id
    )

    competitor_record["mapped_dataset_id"] = output_dataset_id
    scrape_info_kvs_client.set_record(competitor_name, competitor_record)

    if is_on_platform:
        aggregator_task_client = client.task(parameters["aggregator_task_id"])
        aggregator_task_client.start(task_input={
            "scrape_info_kvs_id": scrape_info_kvs_id,
            "competitor_name": competitor_name
        })

    print("Done\n")
