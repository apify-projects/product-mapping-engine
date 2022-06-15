# RUNNING CONFIGURATION
IS_ON_PLATFORM = False
NUMBER_OF_TRAINING_RUNS = 3
LOAD_PRECOMPUTED_SIMILARITIES = True
SAVE_PRECOMPUTED_SIMILARITIES = True
LOAD_PRECOMPUTED_MATCHES = False
SAVE_PRECOMPUTED_MATCHES = False

# TEXT PREPROCESSING CONFIGURATION
# Text column names that should be preprocessed and used for similarity computations
COLUMNS_TO_BE_PREPROCESSED = ['name', 'short_description', 'long_description', 'specification_text', 'all_texts']
# Elements
SIMILARITIES_TO_BE_COMPUTED = ['id', 'brand', 'words', 'cos', 'descriptives', 'units', 'numbers']
# Keywords not to detect and which similarities not to compute specified for each column name
KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED = {'long_description': ['id', 'brand', 'color', 'words'],
                                                                  'specification_text': ['descriptives', 'cos',
                                                                                         'words'],
                                                                  'all_texts': ['id', 'brand', 'color', 'numbers',
                                                                                'words', 'units']}
ALL_KEYWORDS_SIMILARITIES = ['all_units_list', 'all_brands_list', 'all_ids_list', 'all_numbers_list']
LOWER_CASE_TEXT = True
LANGUAGE = 'czech'  # czech, english
TEXT_HASH_SIZE = 16

# KEYWORDS DETECTION CONFIGURATION
PERFORM_ID_DETECTION = True
PERFORM_COLOR_DETECTION = True
PERFORM_BRAND_DETECTION = True
PERFORM_UNITS_DETECTION = True
PERFORM_NUMBERS_DETECTION = True

# TEXT FILTERING CONFIGURATION
MINIMAL_DETECTABLE_ID_LENGTH = 4
NUMBER_OF_TOP_DESCRIPTIVE_WORDS = 50
MAX_DESCRIPTIVE_WORD_OCCURRENCES_IN_TEXTS = 0.5
MIN_DESCRIPTIVE_WORDS_FOR_MATCH = 0
MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH = 2
MIN_MATCH_PRICE_RATIO = 0.5
MAX_MATCH_PRICE_RATIO = 2

# IMAGE PREPROCESSING CONFIGURATION
IMAGE_HASHES_SIZE = 8  # TODO: call from js somehow
IMAGE_RESIZE_WIDTH = 1024
IMAGE_RESIZE_HEIGHT = 1024
IMAGE_FILTERING = True
IMAGE_FILTERING_THRESH = 0.9
HEX_GROUPS_FOR_IMAGE_HASHES = 1

# SIMILARITY COMPUTATIONS CONFIGURATION
KEY_SIMILARITY_LIMIT = 0.9
NUMBER_SIMILARITY_DEVIATION = 0.1
STRING_SIMILARITY_DEVIATION = 0.1
UNITS_AND_VALUES_DEVIATION = 0.05
COMPUTE_TEXT_SIMILARITIES = True
COMPUTE_IMAGE_SIMILARITIES = True

# TRAINING CONFIGURATION
TEST_DATA_PROPORTION = 0.2
PRINCIPAL_COMPONENT_COUNT = 20
PERFORM_PCA_ANALYSIS = False
EQUALIZE_CLASS_IMPORTANCE = False
POSITIVE_CLASS_UPSAMPLING_RATIO = 10

# EVALUATION CONFIGURATION
NUMBER_OF_THRESHES = 30
NUMBER_OF_THRESHES_FOR_AUC = 30
PRINT_ROC_AND_STATISTICS = True
PRINT_CORRELATION_MATRIX = False
CORRELATION_LIMIT = 0.7
MINIMAL_PRECISION = 0.2
MINIMAL_RECALL = 0.2
BEST_MODEL_SELECTION_CRITERION = 'max_precision'  # max_precision, max_recall, balanced_precision_recall

# CLASSIFIER PARAMETERS CONFIGURATION
LinearRegression_CLASSIFIER_PARAMETERS = {}
LogisticRegression_CLASSIFIER_PARAMETERS = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                                            'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
                                            'max_iter': [10, 20, 30, 50, 100, 150, 200, 300, 500],
                                            'class_weight': 'balanced'}
SupportVectorMachine_CLASSIFIER_PARAMETERS = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                              'degree': [2, 3, 4, 5],
                                              'max_iter': [10, 20, 30, 50, 100, 150, 200, 300, 500],
                                              'class_weight': 'balanced', 'probability': True}
DecisionTree_CLASSIFIER_PARAMETERS = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 8, 12, 15, 18, 21],
                                      'min_samples_split': [2, 5, 8, 11, 14, 17],
                                      'class_weight': 'balanced'}
RandomForests_CLASSIFIER_PARAMETERS = {'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
                                       'criterion': ['gini', 'entropy'],
                                       'max_depth': [5, 8, 12, 15, 18, 21], 'min_samples_split': [2, 5, 8, 11, 14, 17],
                                       'class_weight': 'balanced'}
NeuralNetwork_CLASSIFIER_PARAMETERS = {'hidden_layer_sizes': [(10), (50), (100), (10, 10), (50, 50), (100, 100),
                                                              (10, 10, 10), (50, 50, 50), (100, 100, 100),
                                                              (50, 10, 50), (100, 50, 100)],
                                       'activation': ['relu', 'logistic', 'tanh'],
                                       'solver': ['adam', 'sgd', 'lbfgs'],
                                       'batch_size': 'auto', 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                       'learning_rate_init': [0.01, 0.001, 0.0001],
                                       'max_iter': [50, 100, 500]}
EnsembleModelling_CLASSIFIER_PARAMETERS = {
    'LogisticRegression': [{'class_weight': 'balanced'}],
    'SupportVectorMachine': [{'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True}],
    'DecisionTree': [{'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 8, 'class_weight': 'balanced'}],
    'RandomForests': [{'n_estimators': 400, 'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 14,
                       'class_weight': 'balanced'}],
    'NeuralNetwork': [
        {'hidden_layer_sizes': (50, 50), 'activation': 'tanh', 'solver': 'lbfgs', 'batch_size': 'auto',
         'learning_rate': 'invscaling', 'learning_rate_init': 0.01, 'max_iter': 200}
    ]
}

# BEST CLASSIFIER PARAMETERS SEARCH CONFIGURATION
PERFORMED_PARAMETERS_SEARCH = 'none'  # grid, random, none
RANDOM_SEARCH_ITERATIONS = 10
NUMBER_OF_TRAINING_REPETITIONS_TO_AVERAGE_RESULTS = 1
