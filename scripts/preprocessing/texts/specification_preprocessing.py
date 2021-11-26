from scripts.preprocessing.texts.keywords_detection import detect_ids_brands_colors_and_units
from scripts.preprocessing.texts.text_preprocessing import preprocess_text


def preprocess_specification_as_normal_text(dataset):
    """
    Preprocess specifications to create texts for similarity computations by the same way as all other textual data
    @param dataset: list of product specifications where each consist of list of both parameter name and value
    @return: List of preprocessed texts containing whole specification as one long string
    """
    joined_dataset = []
    for product_specification in dataset:
        product_specification = ' '.join(product_specification)
        joined_dataset.append(product_specification)
    preprocessed_dataset = preprocess_text(joined_dataset)
    preprocessed_dataset = detect_ids_brands_colors_and_units(preprocessed_dataset, id_detection=False,
                                                              color_detection=False,
                                                              brand_detection=False,
                                                              units_detection=True)
    return preprocessed_dataset



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
    Preprocess specifications for further similarity computations - separate parameter name and value and detect units
    @param separator: separator, that should be used for separation of names and values in specification
    @param dataset: list of products specifications
    @return: list of preprocessed products specifications
    """
    preprocessed_dataset = []
    for product_specification in dataset:
        specification_dict = {}
        product_specification = separate_parameter_names_and_values(product_specification, separator)
        for item in product_specification:
            item = preprocess_text(item)
            name = ' '.join(item[0])
            value = detect_ids_brands_colors_and_units([item[1]], id_detection=False,
                                                              color_detection=False,
                                                              brand_detection=False,
                                                              units_detection=True)
            specification_dict[name] = ' '.join(value[0])
        preprocessed_dataset.append(specification_dict)
    return preprocessed_dataset
