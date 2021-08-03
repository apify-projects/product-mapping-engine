import os
import json
import click
import pandas as pd
import subprocess
import sys

# Adding the higher level directory (scripts/) to sys.path so that we can import from the other folders
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from sklearn.model_selection import train_test_split

from evaluate_classifier import compute_name_similarities
from preprocessing.names.names_preprocessing import detect_ids_brands_and_colors, to_list
from score_computation.names.compute_names_similarity import lower_case, remove_colors, compute_tf_idf
from scripts.preprocessing.images.image_preprocessing import crop_images_contour_detection, create_output_directory

@click.command()
@click.option('--dataset_folder', '-d',
              default='data/wdc_dataset/dataset/preprocessed',
              help='Dataset to use for the evaluation')
@click.option('--classifier', '-c',
              default='Linear',
              type=click.Choice(['Linear']))
@click.option('--classifier_parameters_path', '-p',
              default='classifier_parameters/linear.json')
# Load product names and images compute their similarity
def main(**kwargs):
    data = preprocess_data(os.path.join(os.getcwd(), kwargs['dataset_folder']))
    data.to_csv("data.csv", index=False)

    classifier_class_name = kwargs['classifier'] + "Classifier"
    classifier_class = getattr(__import__('classifiers', fromlist=[classifier_class_name]), classifier_class_name)
    classifier_parameters_path = kwargs["classifier_parameters_path"]
    classifier_parameters_json = '{}'
    with open(classifier_parameters_path, 'r') as classifier_parameters_file:
        classifier_parameters_json = classifier_parameters_file.read()

    classifier_parameters = json.loads(classifier_parameters_json)
    classifier = classifier_class(classifier_parameters)

    evaluate_classifier(classifier, data)


def preprocess_data(dataset_folder):
    name_similarities_path = os.path.join(dataset_folder, "name_similarities.csv")
    image_similarities_path = os.path.join(dataset_folder, "image_similarities.csv")

    name_similarities_exist = os.path.isfile(name_similarities_path)
    image_similarities_exist = os.path.isfile(image_similarities_path)

    product_pairs = pd.read_csv(os.path.join(dataset_folder, "product_pairs.csv"))
    total_count = 0
    imaged_count = 0
    for pair in product_pairs.itertuples():
        total_count += 1
        if pair.image1 > 0 and pair.image2 > 0:
            imaged_count += 1

    if True or not name_similarities_exist or not image_similarities_exist:
        if not name_similarities_exist:
            names = []
            names_by_id = {}
            for pair in product_pairs.itertuples():
                names_by_id[pair.id1] = len(names)
                names.append(pair.name1)
                names_by_id[pair.id2] = len(names)
                names.append(pair.name2)

            names = to_list(names)
            names, _, _ = detect_ids_brands_and_colors(names, compare_words=False)
            names = [' '.join(name) for name in names]
            names = lower_case(names)
            names = remove_colors(names)
            tf_idfs = compute_tf_idf(names)

            name_similarities_list = []
            for pair in product_pairs.itertuples():
                name1_index = names_by_id[pair.id1]
                name2_index = names_by_id[pair.id2]
                name_similarities = compute_name_similarities(
                    names[name1_index],
                    names[name2_index],
                    name1_index,
                    name2_index,
                    tf_idfs
                )
                name_similarities_list.append(name_similarities)

            name_similarities_dataframe = pd.DataFrame(name_similarities_list)
            name_similarities_dataframe.fillna(0, inplace=True)
            name_similarities_dataframe.to_csv(name_similarities_path, index=False)

        if True or not image_similarities_exist:
            # TODO delete the next line and fill this
            img_source_dir = os.path.join(dataset_folder, 'images_cropped')
            img_dir = os.path.join(dataset_folder, 'images')
            create_output_directory(img_source_dir)
            # crop_images_contour_detection(img_dir, img_source_dir)
            hashes_dir = os.path.join(dataset_folder, "hashes_cropped.json")
            # subprocess.call(f'node scripts/preprocessing/images/image_hash_creator/main.js {img_source_dir} {hashes_dir}')
            # ouje! works till here
            # TODO: compute similarity of hashes
            #product_pairs.to_csv(image_similarities_path, index=False)

    name_similarities = pd.read_csv(name_similarities_path)
    image_similarities = pd.read_csv(image_similarities_path)
    return pd.concat([name_similarities, product_pairs["match"]], axis=1)


def evaluate_outputs(data, outputs, data_type):
    data['match_prediction'] = outputs
    data_count = data.shape[0]
    mismatched = data[data['match_prediction'] != data['match']]
    mismatched_count = data[data['match_prediction'] != data['match']].shape[0]
    actual_positive_count = data[data['match'] == 1].shape[0]
    actual_negative_count = data[data['match'] == 0].shape[0]
    true_positive_count = data[(data['match_prediction'] == 1) & (data['match'] == 1)].shape[0]
    false_positive_count = data[(data['match_prediction'] == 1) & (data['match'] == 0)].shape[0]
    true_negative_count = data[(data['match_prediction'] == 0) & (data['match'] == 0)].shape[0]
    false_negative_count = data[(data['match_prediction'] == 0) & (data['match'] == 1)].shape[0]
    print("Classifier results for {} data".format(data_type))
    print("----------------------------")
    print("Accuracy: {}".format((data_count - mismatched_count) / data_count))
    print("Recall: {}".format(true_positive_count / actual_positive_count))
    print("Specificity: {}".format(true_negative_count / actual_negative_count))
    print("Precision: {}".format(true_positive_count / (true_positive_count + false_positive_count)))
    print("\n\n")

    mismatched.to_csv("mismatches.csv")


def evaluate_classifier(classifier, data):
    train, test = train_test_split(data, test_size=0.25)
    classifier.fit(train)
    out_train = classifier.predict(train)
    out_test = classifier.predict(test)
    evaluate_outputs(train, out_train, "train")
    evaluate_outputs(test, out_test, "test")


if __name__ == "__main__":
    main()
