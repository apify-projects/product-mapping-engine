import json
import os
from math import ceil

import pandas as pd
from apify_client import ApifyClient

from product_mapping_engine.scripts.actor_model_interface import load_model_create_dataset_and_predict_matches
from product_mapping_engine.scripts.configuration import LOAD_PRECOMPUTED_MATCHES, SAVE_PRECOMPUTED_MATCHES

CHUNK_SIZE = 1000
LAST_PROCESSED_CHUNK_KEY = 'last_processed_chunk'

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    one_dataset = False

    if one_dataset:
        if not is_on_platform:
            full_dataset = True
            if full_dataset:
                default_kvs_client.set_record(
                    'INPUT',
                    {
                        "task_id": "Alpha-Complete-CZ",
                        "classifier_type": "LogisticRegression",
                        "dataset_1": "9mk56pDWdfDZoCMiR",
                        "images_kvs_1": "iNNZxJhjAatupQSV0",
                        "dataset_2": "axCYJHLt6cmb1gbNJ",
                        "images_kvs_2": "NNZ40CQnWh4KofXJB"
                    }
                )
            else:
                default_kvs_client.set_record(
                    'INPUT',
                    {
                        "task_id": "extra-xcite-mapping",
                        "classifier_type": "LogisticRegression",
                        "dataset_1": "ajZuzoWIkpUSbRqWP",
                        "images_kvs_1": "iCdo7OawbdUx8MJVk",
                        "dataset_2": "MKqJMRLNFXpebNj2X",
                        "images_kvs_2": "cBi3fhJ7xAc9jl5HI"
                    }
                )

        parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
        print('Actor input:')
        print(json.dumps(parameters, indent=2))

        task_id = parameters['task_id']
        classifier_type = parameters['classifier_type']

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
                    classifier_type,
                    model_key_value_store_client=model_key_value_store,
                    task_id=task_id,
                    is_on_platform=is_on_platform
                )

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
            default_dataset_client.push_items(
                predicted_matching_pairs.to_dict(orient='records'))

            if is_on_platform:
                default_kvs_client.set_record(
                    LAST_PROCESSED_CHUNK_KEY,
                    current_chunk
                )

                print("Done\n")
    else:
        """
        records = [['INPUT',
            {
                "task_id": "Alpha-Complete-CZ",
                "classifier_type": "LogisticRegression",
                "dataset_1": "9mk56pDWdfDZoCMiR",
                "images_kvs_1": "iNNZxJhjAatupQSV0",
                "dataset_2": "axCYJHLt6cmb1gbNJ",
                "images_kvs_2": "NNZ40CQnWh4KofXJB"
            }]]
        similarity_scores_all_datasets = []
        for record in records:
            default_kvs_client.set_record(record[0], record[1])

            parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
            print('Actor input:')
            print(json.dumps(parameters, indent=2))

            task_id = parameters['task_id']
            classifier_type = parameters['classifier_type']

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
                        classifier_type,
                        model_key_value_store_client=model_key_value_store,
                        task_id=task_id,
                        is_on_platform=is_on_platform
                    )
                similarity_scores_all_datasets.append([parameters['dataset_1'], parameters['dataset_2'], all_product_pairs_matching_scores])
            """

        df12 = pd.DataFrame([['p1', 'p2', 0.9], ['p1', 'p4', 0.8], ['p6', 'p7', 0.3]],
                            columns=['id1', 'id2', 'predicted_scores'])
        df13 = pd.DataFrame([['p1', 'p3', 0.8], ['p1', 'p5', 0.8], ['p8', 'p9', 0.3]],
                            columns=['id1', 'id2', 'predicted_scores'])
        df23 = pd.DataFrame([['p2', 'p3', 0.9], ['p7', 'p9', 0.8]], columns=['id1', 'id2', 'predicted_scores'])

        df14 = pd.DataFrame([['p6', 'p0', 0.8]], columns=['id1', 'id2', 'predicted_scores'])
        df24 = pd.DataFrame([['p7', 'p0', 0.9]], columns=['id1', 'id2', 'predicted_scores'])
        similarity_scores_all_datasets = {('d1', 'd2'): df12, ('d1', 'd3'): df13, ('d2', 'd3'): df23,
                                          ('d1', 'd4'): df14, ('d2', 'd4'): df24}
        source_dataset_id = 'd1'  # parameters['dataset_1']
        target_dataset_id = 'd2'  # parameters['dataset_2']
        similarity_scores_source_target_datasets = similarity_scores_all_datasets[
            (source_dataset_id, target_dataset_id)]
        similarity_scores_merged_data = similarity_scores_source_target_datasets

        third_datasets_ids = [datasets_ids[1] for datasets_ids in similarity_scores_all_datasets.keys() if
                              datasets_ids[0] == source_dataset_id and datasets_ids[1] != target_dataset_id]
        for third_dataset_id in third_datasets_ids:
            similarity_scores_dataset1 = similarity_scores_all_datasets[(source_dataset_id, third_dataset_id)]
            similarity_scores_dataset2 = similarity_scores_all_datasets[(target_dataset_id, third_dataset_id)]

            similarity_scores_merged_data = pd.merge(similarity_scores_merged_data,
                                                     similarity_scores_dataset1, how='left',
                                                     left_on=['id1'], right_on=['id1']).fillna(0)
            similarity_scores_merged_data = pd.merge(similarity_scores_merged_data,
                                                     similarity_scores_dataset2, how='left',
                                                     left_on=['id2_x', 'id2_y'],
                                                     right_on=['id1', 'id2']).fillna(0)
            similarity_scores_merged_data = similarity_scores_merged_data.drop(columns=['id1_y', 'id2'])
            similarity_scores_merged_data = similarity_scores_merged_data.rename(
                columns={'id1_x': 'id1', 'id2_x': 'id2', 'id2_y': 'id3',
                         'predicted_scores_y': 'predicted_scores_13',
                         'predicted_scores': 'predicted_scores_23',
                         'predicted_scores_x': 'predicted_scores'})
            similarity_scores_merged_data['predicted_scores'] += \
                similarity_scores_merged_data['predicted_scores_13'] * \
                similarity_scores_merged_data['predicted_scores_23']
            similarity_scores_merged_data = similarity_scores_merged_data.drop(
                columns=['predicted_scores_13', 'predicted_scores_23'])

        similarity_scores_merged_data.loc[similarity_scores_merged_data['predicted_scores'] >= 1, 'predicted_match'] = 1
        similarity_scores_merged_data.loc[similarity_scores_merged_data['predicted_scores'] < 1, 'predicted_match'] = 0
        similarity_scores_merged_data = similarity_scores_merged_data.drop(columns=['id3'])
        print("Done\n")
