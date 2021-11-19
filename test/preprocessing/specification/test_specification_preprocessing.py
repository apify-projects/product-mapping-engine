from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocessing.texts.keywords_detection import detect_ids_brands_colors_and_units
from scripts.preprocessing.texts.text_preprocessing import split_units_and_values
from scripts.score_computation.texts.compute_texts_similarity import remove_markers, create_tf_idf, \
    compare_units_and_values

test_dataset1 = ["Provedení: Pecky 4ml", "Konstrukce: Uzavřená", "Mikrofon: Ano", "Typ připojení: Bluetooth",
                 "Verze: Bluetooth 5.0", "Maximální výdrž baterie: 25 h", "Výdrž baterie (sluchátka): 5 h",
                 "Výdrž baterie (pouzdro): 20 h", "Nabíjení: USB-C, V pouzdře", "Barva: Zlatá", "Hmotnost: 57 g"]
test_dataset2 = ["Provedení: Špunty", "Konstrukce: Uzavřená", "Mikrofon: Ano", "Typ připojení: Bluetooth",
                 "Verze Bluetooth: 5.0", "Typ připojení: Bluetooth", "Verze Bluetooth: 5.0",
                 "Maximální výdrž baterie: 32 h", "Výdrž baterie (sluchátka): 8 h", "Výdrž baterie (pouzdro): 24 h",
                 "Nabíjení: USB-C, V pouzdře", "Barva: Bílá", "Hmotnost: 73 g"]


def separate_parameter_names_and_values(dataset, separator):
    separated_dataset = []
    for data in dataset:
        separated_data = data.split(separator)
        separated_dataset.append([separated_data[0], separated_data[1]])
    return separated_dataset


def main():
    separated_dataset1 = separate_parameter_names_and_values(test_dataset1, separator=': ')
    separated_dataset2 = separate_parameter_names_and_values(test_dataset2, separator=': ')

    separated_dataset1 = detect_and_convert_unit_values(separated_dataset1)
    separated_dataset2 = detect_and_convert_unit_values(separated_dataset2)

    score = compare_parameters(separated_dataset1, separated_dataset2)
    dataset1 = [[d for word in we for d in word.split(' ')] for we in separated_dataset1]
    dataset2 = [[d for word in we for d in word.split(' ')] for we in separated_dataset2]
    dataset1 = [item for sublist in dataset1 for item in sublist]
    dataset2 = [item for sublist in dataset2 for item in sublist]
    score = compare_units_and_values(dataset1, dataset2)
    separated_dataset1 = remove_markers(separated_dataset1)
    separated_dataset2 = remove_markers(separated_dataset2)
    cos_sim_score = compute_cos_similarity_score(separated_dataset1, separated_dataset2)
    print('score: ' + str(score))
    print('cos_sim_socre: ' + str(cos_sim_score))

def compute_cos_similarity_score(dataset1, dataset2):
    dataset1 = [' '.join(d) for d in dataset1]
    dataset2 = [' '.join(d) for d in dataset2]
    tf_idfs = create_tf_idf([dataset1], [dataset2])
    similarity = cosine_similarity([tf_idfs.iloc[0].values, tf_idfs.iloc[1].values])[0][1]
    return similarity

def compare_parameters(dataset1, dataset2):
    score = 0
    for [name1, value1] in dataset1:
        for [name2, value2] in dataset2:
            if name1 == name2 and value1 == value2:
                score += 1
    return score / len(dataset1)


def detect_and_convert_unit_values(dataset):
    for i, [_, parameter_value] in enumerate(dataset):
        parameter_value_split = split_units_and_values(parameter_value.split(' '))
        parameter_value_detected = detect_ids_brands_colors_and_units([parameter_value_split], id_detection=False,
                                                                      color_detection=False,
                                                                      brand_detection=False, units_detection=True)
        parameter_value = ' '.join(parameter_value_detected[0])
        dataset[i][1] = parameter_value
    return dataset


if __name__ == "__main__":
    main()
