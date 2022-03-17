import os
import sys
from multiprocessing import Pool

import pandas as pd

# DO NOT REMOVE
# Adding the higher level directories to sys.path so that we can import from the other folders
from .dataset_handler.similarity_computation.images.compute_hashes_similarity import \
    create_image_similarities_data
from .dataset_handler.pairs_filtering import filter_possible_product_pairs

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ""))

from .dataset_handler.preprocessing.texts.text_preprocessing import parallel_text_preprocessing
from .dataset_handler.similarity_computation.texts.compute_texts_similarity import \
    create_tf_idfs_and_descriptive_words, create_text_similarities_data
from .configuration import IS_ON_PLATFORM, LOAD_PREPROCESSED_DATA, PERFORM_ID_DETECTION, \
    PERFORM_COLOR_DETECTION, PERFORM_BRAND_DETECTION, PERFORM_UNITS_DETECTION, SIMILARITIES_TO_IGNORE, \
    SAVE_PREPROCESSED_DATA, SAVE_COMPUTED_SIMILARITIES, PERFORM_NUMBERS_DETECTION, COMPUTE_IMAGE_SIMILARITIES, \
    COMPUTE_TEXT_SIMILARITIES
from .classifier_handler.evaluate_classifier import train_classifier, evaluate_classifier, setup_classifier


def split_dataframes(dataset):
    """
    Split preprocessed dataframe into dataframe with detected keywords and without them
    @param dataset: preprocessed dataframe
    @return: two dataframes with detected keywords and without them
    """
    columns = [col for col in dataset.columns if 'no_detection' in col] + ['all_texts', 'price']
    dataset_without_marks = dataset[[col for col in columns]]
    dataset_without_marks.columns = dataset_without_marks.columns.str.replace('_no_detection', '')
    dataset = dataset[[col for col in dataset.columns if col not in columns]]
    return dataset, dataset_without_marks


def create_image_and_text_similarities(dataset1, dataset2, tf_idfs, descriptive_words, dataset2_starting_index, pool,
                                       num_cpu,
                                       dataset_folder='',
                                       dataset_dataframe=None,
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
    @param dataset_dataframe: dataframe of pairs to be compared
    @param dataset_images_kvs1: key-value-store client where the images for the source dataset are stored
    @param dataset_images_kvs2: key-value-store client where the images for the target dataset are stored
    @param dataset2_starting_index: starting index of the data from second dataset in tf_idfs and descriptive_words
    @return: list of dataframes with image and text similarities
    """
    product_pairs_idx = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
        os.path.join(dataset_folder, "product_pairs.csv"))

    if not COMPUTE_IMAGE_SIMILARITIES and not COMPUTE_TEXT_SIMILARITIES:
        print(
            'No similarities to be computed. Check value of COMPUTE_IMAGE_SIMILARITIES and COMPUTE_TEXT_SIMILARITIES.')
        exit()

    if COMPUTE_TEXT_SIMILARITIES:
        print("Text similarities computation started")
        name_similarities = create_text_similarities_data(dataset1, dataset2, product_pairs_idx, tf_idfs,
                                                          descriptive_words, dataset2_starting_index,
                                                          pool, num_cpu)
        print("Text similarities computation finished")
    else:
        name_similarities = pd.DataFrame()

    if COMPUTE_IMAGE_SIMILARITIES:
        print("Image similarities computation started")
        pair_identifications = []
        for source_id, target_ids in product_pairs_idx.items():
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

    if len(name_similarities) == 0:
        return image_similarities
    if len(image_similarities) == 0:
        return name_similarities
    return pd.concat([name_similarities, image_similarities['hash_similarity']], axis=1)


def prepare_data_for_classifier(dataset1, dataset2, images_kvs1_client, images_kvs2_client,
                                filter_data):
    """
    Preprocess data, possibly filter data pairs and compute similarities
    @param dataset1: Source dataframe of products
    @param dataset2: Target dataframe with products to be searched in for the same products
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param filter_data: True whether filtering during similarity computations should be performed
    @return: dataframe with image and text similarities
    """
    # setup parallelling stuff
    pool = Pool()
    num_cpu = os.cpu_count()

    # preprocess data
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
    dataset1 = dataset1.sort_index()
    dataset2 = dataset2.sort_index()
    dataset1, dataset1_without_marks = split_dataframes(dataset1)
    dataset2, dataset2_without_marks = split_dataframes(dataset2)
    dataset1.to_csv('data1.csv')
    dataset2.to_csv('data2.csv')
    dataset2_starting_index = len(dataset1_without_marks)
    # create tf_idfs
    tf_idfs, descriptive_words = create_tf_idfs_and_descriptive_words(dataset1_without_marks, dataset2_without_marks)
    print("Text preprocessing finished")

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

    # create image and text similarities
    print("Similarities creation started")
    image_and_text_similarities = create_image_and_text_similarities(dataset1, dataset2, tf_idfs, descriptive_words,
                                                                     dataset2_starting_index, pool, num_cpu,
                                                                     dataset_folder='.',
                                                                     dataset_dataframe=pairs_dataset_idx,
                                                                     dataset_images_kvs1=images_kvs1_client,
                                                                     dataset_images_kvs2=images_kvs2_client
                                                                     )

    print("Similarities creation ended")
    return image_and_text_similarities


def evaluate_executor_results(classifier, preprocessed_pairs, task_id):
    """
    Evaluate results of executors predictions and filtering
    @param classifier: classifier used for predicting pairs
    @param preprocessed_pairs: dataframe with predicted and filtered pairs
    @param task_id: unique identification of the currently evaluated Product Mapping task
    """
    print('{}_unlabeled_data.csv'.format(task_id))
    labeled_dataset = pd.read_csv('{}_unlabeled_data.csv'.format(task_id))
    print("Labeled dataset")
    print(labeled_dataset.shape)

    matching_pairs = labeled_dataset[['id1', 'id2', 'name1', 'name2', 'url1', 'url2', 'match', 'price1', 'price2']]
    predicted_pairs = preprocessed_pairs[['id1', 'id2', 'predicted_scores', 'predicted_match']]

    print("Predicted pairs")
    print(predicted_pairs[predicted_pairs['predicted_match'] == 1].shape)

    merged_data = predicted_pairs.merge(matching_pairs, on=['id1', 'id2'], how='outer')

    predicted_pairs[predicted_pairs['predicted_match'] == 1][['id1', 'id2']].to_csv("predicted.csv")

    merged_data['match'] = merged_data['match'].fillna(0)
    merged_data['predicted_scores'] = merged_data['predicted_scores'].fillna(0)
    merged_data['predicted_match'] = merged_data['predicted_match'].fillna(0)

    merged_data_to_save = merged_data[merged_data['match'] == 1]
    merged_data_to_save = merged_data_to_save[merged_data_to_save['predicted_match'] == 0]
    merged_data_to_save.to_csv("merged.csv")

    merged_data = merged_data.drop(['id1', 'id2', 'url1', 'url2', 'price1', 'price2'], axis=1)
    stats = evaluate_classifier(classifier, merged_data, merged_data, False)
    print(stats)


def load_model_create_dataset_and_predict_matches(
        dataset1,
        dataset2,
        images_kvs1_client,
        images_kvs2_client,
        classifier_type,
        model_key_value_store_client=None,
        task_id="basic",
        is_on_platform=IS_ON_PLATFORM
):
    """
    For each product in first dataset find same products in the second dataset
    @param dataset1: Source dataset of products
    @param dataset2: Target dataset with products to be searched in for the same products
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param classifier_type: Classifier used for product matching
    @param model_key_value_store_client: key-value-store client where the classifier model is stored
    @param task_id: unique identification of the current Product Mapping task
    @param is_on_platform: True if this is running on the platform
    @return: List of same products for every given product
    """
    classifier = setup_classifier(classifier_type)
    classifier.load(key_value_store=model_key_value_store_client)
    preprocessed_pairs_file_path = "preprocessed_pairs_{}.csv".format(task_id)
    preprocessed_pairs_file_exists = os.path.exists(preprocessed_pairs_file_path)

    if LOAD_PREPROCESSED_DATA and preprocessed_pairs_file_exists:
        preprocessed_pairs = pd.read_csv(preprocessed_pairs_file_path)
    else:
        preprocessed_pairs = prepare_data_for_classifier(dataset1, dataset2, images_kvs1_client,
                                                         images_kvs2_client,
                                                         filter_data=True)
    if not is_on_platform and SAVE_PREPROCESSED_DATA:
        preprocessed_pairs.to_csv(preprocessed_pairs_file_path, index=False)

    if 'index1' in preprocessed_pairs.columns and 'index2' in preprocessed_pairs.columns:
        preprocessed_pairs = preprocessed_pairs.drop(['index1', 'index2'], axis=1)

    if SIMILARITIES_TO_IGNORE:
        preprocessed_pairs = preprocessed_pairs.drop(SIMILARITIES_TO_IGNORE, axis=1, errors='ignore')

    preprocessed_pairs['predicted_match'], preprocessed_pairs['predicted_scores'] = classifier.predict(
        preprocessed_pairs.drop(['id1', 'id2'], axis=1))

    if not is_on_platform:
        evaluate_executor_results(classifier, preprocessed_pairs, task_id)

    predicted_matches = preprocessed_pairs[preprocessed_pairs['predicted_match'] == 1][
        ['id1', 'id2', 'predicted_scores']
    ]
    return predicted_matches


def load_data_and_train_model(
        classifier_type,
        dataset_folder='',
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
    @param dataset_folder: (optional) folder containing data
    @param dataset_dataframe: dataframe of pairs to be compared
    @param images_kvs1_client: key-value-store client where the images for the source dataset are stored
    @param images_kvs2_client: key-value-store client where the images for the target dataset are stored
    @param output_key_value_store_client: key-value-store client where the trained model should be stored
    @param task_id: unique identification of the current Product Mapping task
    @param is_on_platform: True if this is running on the platform
    @return: train and test stats after training
    """
    similarities_file_path = "similarities_{}.csv".format(task_id)
    similarities_file_exists = os.path.exists(similarities_file_path)

    if LOAD_PREPROCESSED_DATA and similarities_file_exists:
        similarities = pd.read_csv(similarities_file_path)
    else:
        product_pairs = dataset_dataframe if dataset_dataframe is not None else pd.read_csv(
            os.path.join(dataset_folder, "product_pairs.csv"))

        product_pairs1 = product_pairs.filter(regex='1')
        product_pairs1.columns = product_pairs1.columns.str.replace("1", "")
        product_pairs2 = product_pairs.filter(regex='2')
        product_pairs2.columns = product_pairs2.columns.str.replace("2", "")
        preprocessed_pairs = prepare_data_for_classifier(product_pairs1, product_pairs2, images_kvs1_client,
                                                         images_kvs2_client, filter_data=False)
        if 'index1' in preprocessed_pairs.columns and 'index2' in preprocessed_pairs.columns:
            preprocessed_pairs = preprocessed_pairs.drop(columns=['index1', 'index2'])
        similarities_to_concat = [preprocessed_pairs]
        if 'match' in product_pairs.columns:
            similarities_to_concat.append(product_pairs['match'])
        similarities = pd.concat(similarities_to_concat, axis=1)
        if not is_on_platform and SAVE_COMPUTED_SIMILARITIES:
            similarities.to_csv(similarities_file_path, index=False)

    classifier = setup_classifier(classifier_type)
    if SIMILARITIES_TO_IGNORE:
        similarities = similarities.drop(SIMILARITIES_TO_IGNORE, axis=1, errors='ignore')
    train_stats, test_stats = train_classifier(classifier, similarities.drop(columns=['id1', 'id2']))
    classifier.save(key_value_store=output_key_value_store_client)
    feature_names = [col for col in similarities.columns if col not in ['id1', 'id2', 'match']]
    if not classifier.use_pca:
        classifier.print_feature_importance(feature_names)
    return train_stats, test_stats


# NOT USED METHODS
def create_dataset_for_predictions(product, maybe_the_same_products):
    """
    Create one dataset for model to predict matches that will consist of following
    pairs: given product with every product from the dataset of possible matches
    @param product: product to be compared with all products in the dataset
    @param maybe_the_same_products: dataset of products that are possibly the same as given product
    @return: one dataset for model to predict pairs
    """
    maybe_the_same_products = maybe_the_same_products.rename(columns=lambda s: s + '2')
    final_dataset = pd.DataFrame(columns=product.index.values)
    for _ in range(0, len(maybe_the_same_products.index)):
        final_dataset = final_dataset.append(product, ignore_index=True)
    final_dataset.reset_index(drop=True, inplace=True)
    maybe_the_same_products.reset_index(drop=True, inplace=True)
    final_dataset = final_dataset.rename(columns=lambda s: s + '1')
    final_dataset = pd.concat([final_dataset, maybe_the_same_products], axis=1)
    return final_dataset


def filter_and_save_fp_and_fn(original_dataset):
    """
    Filter and save FP and FN from predicted matches
    @param original_dataset: dataframe with original data
    @return:
    """
    original_dataset['index'] = original_dataset.index
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    train_test_data = pd.concat([train_data, test_data])
    predicted_pairs = train_test_data.join(original_dataset, on='index1', how='left')
    joined_datasets = predicted_pairs.drop(['index1', 'index'], 1)
    fn_train = joined_datasets[(joined_datasets['match'] == 1) & (joined_datasets['predicted_match'] == 0)]
    fp_train = joined_datasets[(joined_datasets['match'] == 0) & (joined_datasets['predicted_match'] == 1)]
    fn_train.to_csv(f'fn_dataset.csv', index=False)
    fp_train.to_csv(f'fp_dataset.csv', index=False)
