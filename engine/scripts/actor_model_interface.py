import hashlib
import os
import sys
from multiprocessing import Pool

import pandas as pd

from .dataset_handler.pairs_filtering import filter_possible_product_pairs
from .dataset_handler.similarity_computation.images.compute_hashes_similarity import \
    create_image_similarities_data

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ""))

from .dataset_handler.preprocessing.texts.text_preprocessing import parallel_text_preprocessing
from .dataset_handler.similarity_computation.texts.compute_texts_similarity import \
    create_tf_idfs_and_descriptive_words, create_text_similarities_data
from .configuration import IS_ON_PLATFORM, PERFORM_ID_DETECTION, \
    PERFORM_COLOR_DETECTION, PERFORM_BRAND_DETECTION, PERFORM_UNITS_DETECTION, \
    SAVE_PRECOMPUTED_SIMILARITIES, PERFORM_NUMBERS_DETECTION, COMPUTE_IMAGE_SIMILARITIES, \
    COMPUTE_TEXT_SIMILARITIES, TEXT_HASH_SIZE, LOAD_PRECOMPUTED_SIMILARITIES, PERFORMED_PARAMETERS_SEARCH, \
    NUMBER_OF_TRAINING_RUNS, PRINT_FEATURE_IMPORTANCE, DATA_FOLDER, EXCLUDE_CATEGORIES, TASK_ID, SAVE_PREPROCESSED_DATA, \
    LOAD_PREPROCESSED_DATA
from .classifier_handler.evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier, \
    parameters_search_and_best_model_training, ensemble_models_training, select_best_classifier, \
    train_excluding_categories


def split_dataframes(dataset):
    """
    Split preprocessed dataframe into dataframe with detected keywords and without them
    @param dataset: preprocessed dataframe
    @return: two dataframes with detected keywords and without them
    """
    codes_column_if_present = []
    if 'code' in dataset.columns:
        codes_column_if_present = ['code']

    columns_without_marks = [col for col in dataset.columns if 'no_detection' in col] + [
        'all_texts'] + codes_column_if_present
    dataset_without_marks = dataset[[col for col in columns_without_marks + ['price']]]
    dataset_without_marks.columns = dataset_without_marks.columns.str.replace('_no_detection', '')
    dataset = dataset[[col for col in dataset.columns if col not in columns_without_marks] + codes_column_if_present]
    return dataset, dataset_without_marks


def create_image_and_text_similarities(dataset1,
                                       dataset2,
                                       tf_idfs,
                                       descriptive_words,
                                       dataset2_starting_index,
                                       pool,
                                       num_cpu,
                                       product_pairs_idxs_dict,
                                       dataset_folder='',
                                       dataset_images_kvs1=None,
                                       dataset_images_kvs2=None
                                       ):
    """
    For each pair of products compute their image and name similarity
    @param dataset1: first dataframe with products
    @param dataset2: second dataframe with products
    @param tf_idfs: dictionary of tf.idfs for each text column in products
    @param descriptive_words:  dictionary of descriptive words for each text column in products
    @param pool: parallelling object
    @param num_cpu: number of processes
    @param dataset_folder: folder containing data to be preprocessed
    @param product_pairs_idxs_dict: dictionary of indices of product pairs to be compared
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @param dataset2_starting_index: size of the dataset1 used for indexing values of second dataset in tf_idfs and
                                    descriptive_words which contains joint dataset1 nad dataset2 into one dataset
    @return: list of dataframes with image and text similarities
    """

    if not COMPUTE_IMAGE_SIMILARITIES and not COMPUTE_TEXT_SIMILARITIES:
        print(
            'No similarities to be computed. Check value of COMPUTE_IMAGE_SIMILARITIES and COMPUTE_TEXT_SIMILARITIES '
            'in your configuration file.')
        exit()

    if COMPUTE_TEXT_SIMILARITIES:
        print("Text similarities computation started")
        name_similarities = create_text_similarities_data(dataset1, dataset2, product_pairs_idxs_dict, tf_idfs,
                                                          descriptive_words, dataset2_starting_index,
                                                          pool, num_cpu)
        print("Text similarities computation finished")
    else:
        name_similarities = pd.DataFrame()

    if COMPUTE_IMAGE_SIMILARITIES:
        print("Image similarities computation started")
        pair_identifications = []
        for source_id, target_ids in product_pairs_idxs_dict.items():
            for target_id in target_ids:
                pair_identifications.append({
                    'id1': dataset1['id'][source_id],
                    'image1': dataset1['image'][source_id],
                    'id2': dataset2['id'][target_id],
                    'image2': dataset2['image'][target_id],
                })
        image_similarities = create_image_similarities_data(pool, num_cpu, pair_identifications,
                                                            dataset_folder=dataset_folder,
                                                            dataset_images_kvs1=dataset_images_kvs1,
                                                            dataset_images_kvs2=dataset_images_kvs2
                                                            )
        print("Image similarities computation finished")
    else:
        image_similarities = pd.DataFrame()

    if len(name_similarities) == 0 and len(image_similarities) == 0:
        return name_similarities
    if len(name_similarities) == 0:
        return image_similarities
    if len(image_similarities) == 0:
        return name_similarities
    return pd.concat([name_similarities, image_similarities['hash_similarity']], axis=1)


def hash_text_using_sha256(text):
    """
    Hash text using sha256 hash
    @param text: text to be hashed
    @return: hashed text
    """
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), TEXT_HASH_SIZE)


def flatten(list_of_lists):
    """
    Flattens list of lists into one list
    @param list_of_lists: List to be flattened
    @return: flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]


def prepare_data_for_classifier(
        dataset1,
        dataset2,
        dataset_precomputed_matches,
        images_kvs1_client,
        images_kvs2_client,
        filter_data
):
    """
    Preprocess data, possibly filter data pairs and compute similarities
    @param dataset1: Source dataframe of products
    @param dataset2: Target dataframe with products to be searched in for the same products
    @param dataset_precomputed_matches: Dataframe with already precomputed matching pairs
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param filter_data: True whether filtering during similarity computations should be performed
    @return: dataframe with image and text similarities, dataset with precomputed similarities (for executor only)
    """
    # setup parallelling stuff
    pool = Pool()
    num_cpu =os.cpu_count() - 1

    # preprocess data
    if LOAD_PREPROCESSED_DATA:
        dataset1 = pd.read_csv(f'{DATA_FOLDER}/{TASK_ID}_preprocessed_dataset1.csv')
        dataset2 = pd.read_csv(f'{DATA_FOLDER}/{TASK_ID}_preprocessed_dataset2.csv')
        dataset1['specification'] = [eval(x) for x in dataset1['specification'].values]
        dataset2['specification'] = [eval(x) for x in dataset2['specification'].values]
        dataset1_without_marks = pd.read_csv(f'{DATA_FOLDER}/{TASK_ID}_dataset1_without_marks.csv')
        dataset2_without_marks = pd.read_csv(f'{DATA_FOLDER}/{TASK_ID}_dataset2_without_marks.csv')
        pairs_dataset_idx = pd.read_csv(f'{DATA_FOLDER}/{TASK_ID}_pairs_dataset_idx.csv').values
        pairs_dataset_idx = {x[0]: eval(x[1]) for x in pairs_dataset_idx}
        tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_without_marks,
                                                                          dataset2_without_marks)
        dataset2_starting_index = len(dataset1)
    else:
        print("Text preprocessing started")
        dataset1 = parallel_text_preprocessing(pool, num_cpu, dataset1,
                                               PERFORM_ID_DETECTION,
                                               PERFORM_COLOR_DETECTION,
                                               PERFORM_BRAND_DETECTION,
                                               PERFORM_UNITS_DETECTION,
                                               PERFORM_NUMBERS_DETECTION)
        dataset2 = parallel_text_preprocessing(pool, num_cpu, dataset2,
                                               PERFORM_ID_DETECTION,
                                               PERFORM_COLOR_DETECTION,
                                               PERFORM_BRAND_DETECTION,
                                               PERFORM_UNITS_DETECTION,
                                               PERFORM_NUMBERS_DETECTION)
        dataset1, dataset1_without_marks = split_dataframes(dataset1)
        dataset2, dataset2_without_marks = split_dataframes(dataset2)
        dataset2_starting_index = len(dataset1_without_marks)
        # create tf_idfs
        tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_without_marks,
                                                                          dataset2_without_marks)
        print("Text preprocessing finished")

        # create hashes from all texts
        dataset1.drop(columns=['price'])
        dataset2.drop(columns=['price'])

        if filter_data:
            # filter product pairs
            print("Filtering started")
            pairs_dataset_idx = filter_possible_product_pairs(dataset1_without_marks, dataset2_without_marks,
                                                              descriptive_words, pool, num_cpu)
            pairs_count = 0
            for key, target_ids in pairs_dataset_idx.items():
                pairs_count += len(target_ids)

            print(f"Filtered to {pairs_count} pairs")
            print("Filtering ended")
        else:
            pairs_dataset_idx = {}
            for i in range(0, len(dataset1)):
                pairs_dataset_idx[i] = [i]

    if SAVE_PREPROCESSED_DATA:
        dataset1.to_csv(f'{DATA_FOLDER}/{TASK_ID}_preprocessed_dataset1.csv', index=False)
        dataset2.to_csv(f'{DATA_FOLDER}/{TASK_ID}_preprocessed_dataset2.csv', index=False)
        dataset1_without_marks.to_csv(f'{DATA_FOLDER}/{TASK_ID}_dataset1_without_marks.csv', index=False)
        dataset2_without_marks.to_csv(f'{DATA_FOLDER}/{TASK_ID}_dataset2_without_marks.csv', index=False)
        pairs_dataset_idx_df = pd.Series(pairs_dataset_idx, index=pairs_dataset_idx.keys())
        pairs_dataset_idx_df.to_csv(f'{DATA_FOLDER}/{TASK_ID}_pairs_dataset_idx.csv')

    # create image and text similarities
    print("Similarities creation started")
    image_and_text_similarities = create_image_and_text_similarities(dataset1, dataset2, tf_idfs, descriptive_words,
                                                                     dataset2_starting_index, pool, num_cpu,
                                                                     pairs_dataset_idx,
                                                                     dataset_folder='.',
                                                                     dataset_images_kvs1=images_kvs1_client,
                                                                     dataset_images_kvs2=images_kvs2_client
                                                                     )

    print("Similarities creation ended")
    return image_and_text_similarities, dataset_precomputed_matches


def evaluate_executor_results(classifier, preprocessed_pairs, task_id, data_type, data_to_remove):
    """
    Evaluate results of executors predictions and filtering
    @param classifier: classifier used for predicting pairs
    @param preprocessed_pairs: dataframe with predicted and filtered pairs
    @param task_id: unique identification of the currently evaluated Product Mapping task
    @param data_type: string value specifying the evaluated data type
    @param data_to_remove: dataframe with pairs to be removed from labeled_dataset
    """
    print('{}_unlabeled_data.csv'.format(task_id))
    labeled_dataset = None
    try:
        labeled_dataset = pd.read_csv(f'{DATA_FOLDER}/{task_id}_unlabeled_data.csv')
    except OSError as e:
        print(e)
        print(
            'To solve this error run trainer and before calling load_data_and_train_model function save labeled dataset'
            ', eg by running: labeled_dataset.to_csv(dataset_name)'
        )
        exit(e.errno)

    # remove data_to_remove from labeled_dataset
    if data_to_remove is not None and len(data_to_remove) != 0:
        labeled_dataset = labeled_dataset.merge(data_to_remove[['id1', 'id2']], on=['id1', 'id2'], how='left',
                                                indicator=True)
        labeled_dataset = labeled_dataset[labeled_dataset['_merge'] == 'left_only'].drop(columns='_merge')

    predicted_pairs = preprocessed_pairs[['id1', 'id2', 'predicted_scores', 'predicted_match']]

    print("Predicted pairs")
    print(predicted_pairs[predicted_pairs['predicted_match'] == 1].shape)

    merged_data = predicted_pairs.merge(labeled_dataset[['id1', 'id2', 'match', 'price1', 'price2', 'name1', 'name2']],
                                        on=['id1', 'id2'], how='outer')
    merged_data.info()
    merged_data.to_csv(f'{DATA_FOLDER}/filtered.csv')

    predicted_pairs[predicted_pairs['predicted_match'] == 1][['id1', 'id2']].to_csv(f'{DATA_FOLDER}/predicted.csv')

    merged_data['match'] = merged_data['match'].fillna(0)
    merged_data['predicted_scores'] = merged_data['predicted_scores'].fillna(0)
    merged_data['predicted_match'] = merged_data['predicted_match'].fillna(0)

    merged_data_to_save = merged_data[merged_data['match'] == 1]
    merged_data_to_save = merged_data_to_save[merged_data_to_save['predicted_match'] == 0]
    merged_data_to_save.to_csv(f'{DATA_FOLDER}/merged.csv')

    columns_to_drop = ['id1', 'id2', 'url1', 'url2', 'price1', 'price2']
    for column in columns_to_drop:
        if column in merged_data:
            merged_data = merged_data.drop(column, axis=1)
    stats = evaluate_classifier(classifier, merged_data, merged_data, data_type)
    print(data_type)
    print(stats)


def load_model_create_dataset_and_predict_matches(
        dataset1,
        dataset2,
        precomputed_pairs_matching_scores,
        images_kvs1_client,
        images_kvs2_client,
        model_key_value_store_client=None,
        task_id="basic",
        is_on_platform=IS_ON_PLATFORM
):
    """
    For each product in first dataset find same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param precomputed_pairs_matching_scores: Dataframe with already precomputed matching pairs
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param model_key_value_store_client: key-value-store client where the classifier model is stored
    @param task_id: unique identification of the current Product Mapping task
    @param is_on_platform: True if this is running on the platform
    @return: dataframe with matching products for every given product
             dataframe with all precomputed and newly computed product pairs matching scores
             dataframe with newly computed product pairs matching scores
    """
    training_parameters = model_key_value_store_client.get_record('parameters')['value']
    classifier_type = training_parameters['classifier_type']

    classifier, _ = setup_classifier(classifier_type)
    classifier.load(key_value_store=model_key_value_store_client)

    preprocessed_pairs_file_path = f'{DATA_FOLDER}/preprocessed_pairs_{task_id}.csv'
    preprocessed_pairs_file_exists = os.path.exists(preprocessed_pairs_file_path)

    if LOAD_PRECOMPUTED_SIMILARITIES and preprocessed_pairs_file_exists:
        preprocessed_pairs = pd.read_csv(preprocessed_pairs_file_path)
    else:
        preprocessed_pairs, precomputed_pairs_matching_scores = prepare_data_for_classifier(
            dataset1,
            dataset2,
            precomputed_pairs_matching_scores,
            images_kvs1_client,
            images_kvs2_client,
            filter_data=True
        )

    if not is_on_platform and SAVE_PRECOMPUTED_SIMILARITIES and len(preprocessed_pairs) != 0:
        preprocessed_pairs.to_csv(preprocessed_pairs_file_path, index=False)

    if 'index1' in preprocessed_pairs.columns and 'index2' in preprocessed_pairs.columns:
        preprocessed_pairs = preprocessed_pairs.drop(['index1', 'index2'], axis=1)

    preprocessed_pairs_to_predict = preprocessed_pairs.drop(
        ['id1', 'id2'], axis=1
    )

    if len(preprocessed_pairs_to_predict) != 0:
        preprocessed_pairs['predicted_match'], preprocessed_pairs['predicted_scores'] = classifier.predict(
            preprocessed_pairs_to_predict
        )
        preprocessed_pairs.to_csv(f'{DATA_FOLDER}/predictions.csv')

        if not is_on_platform:
            evaluate_executor_results(classifier, preprocessed_pairs, task_id, 'new executor data',
                                      precomputed_pairs_matching_scores)

        predicted_matching_pairs = preprocessed_pairs[preprocessed_pairs['predicted_match'] == 1][
            ['id1', 'id2', 'predicted_scores', 'predicted_match']
        ]

        new_product_pairs_matching_scores = preprocessed_pairs[
            ['id1', 'id2', 'predicted_scores', 'predicted_match']
        ]
    else:
        predicted_matching_pairs = pd.DataFrame(
            columns=['id1', 'id2', 'predicted_scores', 'predicted_match']
        )
        new_product_pairs_matching_scores = pd.DataFrame(
            columns=['id1', 'id2', 'predicted_scores', 'predicted_match']
        )
    # Append dataset_precomputed_matches to predicted_matching_pairs
    if precomputed_pairs_matching_scores is not None and len(precomputed_pairs_matching_scores) != 0:
        precomputed_pairs_matching_scores['predicted_match'] = [0] * len(precomputed_pairs_matching_scores)
        precomputed_pairs_matching_scores['predicted_match'] = precomputed_pairs_matching_scores[
            'predicted_match'].mask(
            precomputed_pairs_matching_scores['predicted_scores'] >= classifier.weights['threshold'], 1
        )
        all_product_pairs_matching_scores = pd.concat(
            [new_product_pairs_matching_scores, precomputed_pairs_matching_scores],
            ignore_index=True)

        predicted_matching_pairs = pd.concat(
            [predicted_matching_pairs,
             precomputed_pairs_matching_scores[precomputed_pairs_matching_scores['predicted_match'] == 1]],
            ignore_index=True)
    else:
        all_product_pairs_matching_scores = new_product_pairs_matching_scores

    if not is_on_platform:
        evaluate_executor_results(classifier, all_product_pairs_matching_scores, task_id, 'all executor data', None)

    return predicted_matching_pairs.drop('predicted_match', axis=1), \
           all_product_pairs_matching_scores.drop('predicted_match', axis=1), \
           new_product_pairs_matching_scores.drop('predicted_match', axis=1),


def load_data_and_train_model(
        classifier_type,
        dataset_dataframe=None,
        images_kvs1_client=None,
        images_kvs2_client=None,
        output_key_value_store_client=None,
        task_id="basic",
        is_on_platform=IS_ON_PLATFORM
):
    """
    Load dataset and train and save model
    @param classifier_type: classifier type
    @param dataset_dataframe: dataframe of pairs to be compared
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param output_key_value_store_client: key-value-store client where the trained model should be stored
    @param task_id: unique identification of the current Product Mapping task
    @param is_on_platform: True if this is running on the platform
    @return: train and test stats after training
    """
    # Loading and preprocessing part
    similarities_file_path = f'{DATA_FOLDER}/{task_id}_similarities.csv'
    similarities_file_exists = os.path.exists(similarities_file_path)

    if LOAD_PRECOMPUTED_SIMILARITIES:
        if similarities_file_exists:
            similarities = pd.read_csv(similarities_file_path)
        else:
            similarities = None
    else:
        product_pairs = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
            f'{DATA_FOLDER}/{TASK_ID}.csv')
        if 'category' in product_pairs:
            product_pairs = product_pairs.drop(columns={'category'})
        if 'match_type' in product_pairs:
            product_pairs = product_pairs.drop(columns={'match_type'})
        if 'image_url1' in product_pairs:
            product_pairs = product_pairs.drop(columns={'image_url1'})
        if 'image_url2' in product_pairs:
            product_pairs = product_pairs.drop(columns={'image_url2'})
        product_pairs1 = product_pairs.filter(regex='1')
        product_pairs1.columns = product_pairs1.columns.str.replace("1", "")
        product_pairs2 = product_pairs.filter(regex='2')
        product_pairs2.columns = product_pairs2.columns.str.replace("2", "")
        preprocessed_pairs, _ = prepare_data_for_classifier(product_pairs1, product_pairs2, None,
                                                            images_kvs1_client,
                                                            images_kvs2_client, filter_data=False)
        if 'index1' in preprocessed_pairs.columns and 'index2' in preprocessed_pairs.columns:
            preprocessed_pairs = preprocessed_pairs.drop(columns=['index1', 'index2'])
        similarities_to_concat = [preprocessed_pairs]
        if 'match' in product_pairs.columns:
            similarities_to_concat.append(product_pairs['match'])
        similarities = pd.concat(similarities_to_concat, axis=1)
        if not is_on_platform and SAVE_PRECOMPUTED_SIMILARITIES:
            similarities.to_csv(similarities_file_path, index=False)

    # Training part
    classifiers = []
    if EXCLUDE_CATEGORIES:
        classifier, _ = setup_classifier(classifier_type)
        print(classifier)
        train_excluding_categories(classifier, similarities, task_id)
        return None

    if PERFORMED_PARAMETERS_SEARCH is not 'none':
        classifier, train_stats, test_stats = parameters_search_and_best_model_training(
            similarities,
            classifier_type,
            task_id
        )
        classifiers.append({'classifier': classifier, 'train_stats': train_stats, 'test_stats': test_stats})
    else:
        for _ in range(NUMBER_OF_TRAINING_RUNS):
            if classifier_type in ['Bagging', 'Boosting']:
                classifier, train_stats, test_stats = ensemble_models_training(similarities, classifier_type, task_id)
            else:
                classifier, _ = setup_classifier(classifier_type)
                print(classifier)
                train_stats, test_stats = train_classifier(classifier, similarities, task_id)
            classifiers.append({'classifier': classifier, 'train_stats': train_stats, 'test_stats': test_stats})
    best_classifier, best_train_stats, best_test_stats = select_best_classifier(classifiers)

    if PRINT_FEATURE_IMPORTANCE:
        feature_names = [col for col in similarities.columns if col not in ['id1', 'id2', 'match']]
        best_classifier.print_feature_importance(feature_names)

    return best_train_stats, best_test_stats
