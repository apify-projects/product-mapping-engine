from scripts.preprocessing.texts.keywords_detection import detect_ids_brands_colors_and_units
from scripts.preprocessing.texts.text_preprocessing import split_units_and_values


def detect_and_convert_unit_values(dataset):
    """
    Detect units in specification and convert them to their basis form
    @param dataset: list of pairs of parameter names and their values
    @return: dataset with detected units
    """
    for i, [_, parameter_value] in enumerate(dataset):
        parameter_value_split = split_units_and_values(parameter_value.split(' '))
        parameter_value_detected = detect_ids_brands_colors_and_units([parameter_value_split], id_detection=False,
                                                                      color_detection=False,
                                                                      brand_detection=False, units_detection=True)
        parameter_value = ' '.join(parameter_value_detected[0])
        dataset[i][1] = parameter_value
    return dataset


def separate_parameter_names_and_values(dataset, separator):
    """
    Separate names and values of parameters from the specification pairs
    @param dataset: list with items that contain both parameter and its value together
    @param separator: separator, that should be used from separation of names and values of the parameters
    @return: list of pairs of parameters names and parameter values
    """
    separated_dataset = []
    for data in dataset:
        separated_data = data.split(separator)
        separated_dataset.append([separated_data[0], separated_data[1]])
    return separated_dataset


def preprocess_specification(dataset, separator):
    """
    Preprocess specifications for each product: separate names and values and detect units and convert their values to the basic form
    @param separator: separator, that should be used for separation of names and values in parameters
    @param dataset: list of products specifications
    @return: list of preprocessed products specifications
    """
    preprocessed_dataset = []
    for product_specification in dataset:
        product_specification = separate_parameter_names_and_values(product_specification, separator)
        product_specification = detect_and_convert_unit_values(product_specification)
        preprocessed_dataset.append(product_specification)
    return preprocessed_dataset
