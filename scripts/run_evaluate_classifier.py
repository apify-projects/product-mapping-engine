import click
import os
import pandas as pd
from score_computation.names.compute_names_similarity import lower_case, remove_colors, compute_tf_idf
from preprocessing.names.names_preprocessing import detect_ids_brands_and_colors, to_list
from evaluate_classifier import compute_name_similarities

@click.command()
@click.option('--dataset_folder', '-d',
              default='./dataset',
              help='Dataset to use for the evaluation')
@click.option('--classifier', '-c',
              default='Linear',
              type=click.Choice(['Linear']))

@click.option('--classifier_parameters', '-p',
              default='')
# Load product names and images compute their similarity
def main(**kwargs):
    data = preprocess_data(os.path.join(os.getcwd(), kwargs['dataset_folder']))
    data.to_csv("data.csv", index=False)

    classifier_class_name = kwargs['classifier'] + "Classifier"
    classifier_class = getattr(__import__('classifiers', fromlist=[classifier_class_name]), classifier_class_name)
    classifier_parameters = kwargs["classifier_parameters"]
    unpacked_classifier_parameters = [] if classifier_parameters == "" else classifier_parameters.split(",")
    # TODO deal with this
    #classifier = classifier_class(unpacked_classifier_parameters)
    classifier = classifier_class({"words": 1, "cos": 2, "threshold": 0.5})

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
            names = [ ' '.join(name) for name in names ]
            names = lower_case(names)
            names = remove_colors(names)
            tf_idfs = compute_tf_idf(names)

            name_similarities_list = []
            for pair in product_pairs.itertuples():
                name_similarities = compute_name_similarities(pair.name1, pair.name2, names_by_id[pair.id1], names_by_id[pair.id2], tf_idfs)
                name_similarities_list.append(name_similarities)

            name_similarities_dataframe = pd.DataFrame(name_similarities_list)
            name_similarities_dataframe.to_csv(name_similarities_path, index=False)

        if True or not image_similarities_exist:
            # TODO delete the next line and fill this
            product_pairs.to_csv(image_similarities_path, index=False)

    name_similarities = pd.read_csv(name_similarities_path)
    image_similarities = pd.read_csv(image_similarities_path)
    return pd.concat([name_similarities, product_pairs["match"]], axis=1)

def evaluate_classifier(classifier, data):
    classifier.fit(data)
    outputs = classifier.predict(data)
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
    print("Classifier result attributes")
    print("----------------------------")
    print("Accuracy: {}".format((data_count - mismatched_count) / data_count))
    print("Recall: {}".format(true_positive_count / actual_positive_count))
    print("Specificity: {}".format(true_negative_count / actual_negative_count))
    print("Precision: {}".format(true_positive_count / (true_positive_count + false_positive_count)))

    mismatched.to_csv("mismatches.csv")

if __name__ == "__main__":
    main()
