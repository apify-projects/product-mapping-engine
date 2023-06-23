# RUNNING CONFIGURATION
TASK_ID = 'promapen'  # promapen promapcz
CLASSIFIER_TYPE = 'NeuralNetwork'
IS_ON_PLATFORM = False
LOAD_PRECOMPUTED_SIMILARITIES = True
SAVE_PRECOMPUTED_SIMILARITIES = False
LOAD_PRECOMPUTED_MATCHES = False
SAVE_PRECOMPUTED_MATCHES = False
LOAD_PREPROCESSED_DATA = False
SAVE_PREPROCESSED_DATA = False
DATA_FOLDER = 'data/'
MODEL_FOLDER = 'model/'

# TRAINING CONFIGURATION
PERFORM_TRAIN_TEST_SPLIT = False
SAMPLE_VALIDATION_DATA_FROM_TRAIN_DATA = False
VALIDATION_DATA_PROPORTION = 0.2
SAVE_TRAIN_TEST_SPLIT = False
TEST_DATA_PROPORTION = 0.2
NUMBER_OF_TRAINING_RUNS = 1
PRINCIPAL_COMPONENT_COUNT = 10
PERFORM_PCA_ANALYSIS = False
EQUALIZE_CLASS_IMPORTANCE = False
POSITIVE_CLASS_UPSAMPLING_RATIO = 2
LOAD_PRECOMPUTED_MODEL = False
SAVE_COMPUTED_MODEL = False
MODEL_NAME = 'promapen_MLPClassifier'  # none promapen_MLPClassifier  promapcz_MLPClassifier amazon_walmart_MLPClassifier amazon_google_MLPClassifier
JUST_EVALUATE_LOADED_TEST_DATA = False
EXCLUDE_CATEGORIES = False
CATEGORIES_CZ = ['1_pets', '2_bags', '3_garden', '4_appliances', '5_phones', '6_household', '7_laptops', '8_tvs',
                 '9_headphones', '10_fridges']
CATEGORIES_EN = ['1_pets', '2_bags', '3_garden', '4_appliances', '5_phones', '6_household', '7_laptops', '8_toys',
                 '9_clothes', '10_health']
EXCLUDED_SIMILARITIES = {}
# EVALUATION CONFIGURATION
SET_THRESHOLD = True
NUMBER_OF_THRESHES = 100
NUMBER_OF_THRESHES_FOR_AUC = 30
PRINT_ROC_AND_STATISTICS = False
PRINT_FEATURE_IMPORTANCE = False
PRINT_CORRELATION_MATRIX = False
CORRELATION_LIMIT = 0.5
MINIMAL_PRECISION = 0.5
MINIMAL_RECALL = 0.5
BEST_MODEL_SELECTION_CRITERION = 'max_f1'  # max_precision, max_recall, balanced_precision_recall, max_f1

# TEXT PREPROCESSING CONFIGURATION
# Text column names that should be preprocessed and used for similarity computations
COLUMNS_TO_BE_PREPROCESSED = ['name', 'short_description', 'long_description', 'specification_text', 'all_texts']
# Elements
SIMILARITIES_TO_BE_COMPUTED = ['id', 'brand', 'words', 'cos', 'descriptives', 'units', 'numbers']
# Keywords not to detect and which similarities not to compute specified for each column name
KEYWORDS_NOT_TO_BE_DETECTED_OR_SIMILARITIES_NOT_TO_BE_COMPUTED_DURING_PREPROCESSING = {
    'long_description': ['id', 'brand', 'color', 'words'],
    'specification_text': ['descriptives', 'cos'
                           'words'],
    'all_texts': ['id', 'brand', 'color', 'numbers',
                  'words', 'units']}
ALL_KEYWORDS_SIMILARITIES = ['all_units_list', 'all_brands_list', 'all_ids_list', 'all_numbers_list']
LOWER_CASE_TEXT = True
LANGUAGE = 'english'  # czech, english
TEXT_HASH_SIZE = 16
LEMMATIZER = 'morphoditta'  # majka morphoditta none
DISTANCE = 'cos'  # cos manhattan euclid

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
MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH = 1
MIN_LONG_PRODUCT_NAME_SIMILARITY_FOR_MATCH = 2
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
COMPUTE_IMAGE_SIMILARITIES = False
# CLASSIFIER PARAMETERS CONFIGURATION
LogisticRegression_CLASSIFIER_PARAMETERS = {'penalty': 'none',
                                            'solver': 'newton-cg',
                                            'max_iter': 50,
                                            'class_weight': 'balanced'}
SupportVectorMachine_CLASSIFIER_PARAMETERS = {'kernel': 'poly',
                                              'degree': 5,
                                              'max_iter': 100,
                                              'class_weight': 'balanced',
                                              'probability': True}
DecisionTree_CLASSIFIER_PARAMETERS = {'criterion': 'entropy',
                                      'max_depth': 10,
                                      'min_samples_split': 5,
                                      'class_weight': 'balanced'}
RandomForests_CLASSIFIER_PARAMETERS = {'n_estimators': 100,
                                       'criterion': 'entropy',
                                       'max_depth': 10,
                                       'min_samples_split': 15,
                                       'class_weight': 'balanced'}
NeuralNetwork_CLASSIFIER_PARAMETERS = {'hidden_layer_sizes': (50, 10, 50),
                                       'activation': 'logistic',
                                       'solver': 'lbfgs',
                                       'batch_size': 'auto',
                                       'learning_rate': 'constant',
                                       'learning_rate_init': 0.0001,
                                       'max_iter': 50
                                       }
NeuralNetwork_CLASSIFIER_PARAMETERSs = {'hidden_layer_sizes': (50, 50),
                                       'activation': 'relu',
                                       'solver': 'adam',
                                       'batch_size': 'auto',
                                       'learning_rate': 'constant',
                                       'learning_rate_init': 0.0001,
                                       'max_iter': 200
                                       }
NeuralNetwork_CLASSIFIER_PARAMETERScz = {'hidden_layer_sizes': (50, 10, 50),
                                         'activation': 'relu',
                                         'solver': 'adam',
                                         'batch_size': 'auto',
                                         'learning_rate': 'adaptive',
                                         'learning_rate_init': 0.001, # 0.0001 orig CZ
                                         'max_iter': 100
                                         }
NeuralNetwork_CLASSIFIER_PARAMETERSaw = {'hidden_layer_sizes': (10, 10),
                                       'activation': 'relu',
                                       'solver': 'adam',
                                       'batch_size': 'auto',
                                       'learning_rate': 'constant',
                                       'learning_rate_init': 0.001,
                                       'max_iter': 50
                                       }
NeuralNetwork_CLASSIFIER_PARAMETERSag = {'hidden_layer_sizes': (10, 10, 10),
                                         'activation': 'tanh',
                                         'solver': 'lbfgs',
                                         'batch_size': 'auto',
                                         'learning_rate': 'adaptive',
                                         'learning_rate_init': 0.001,
                                         'max_iter': 200
                                         }
Bagging_CLASSIFIER_PARAMETERS = {  # best for en
    'NeuralNetwork': [
                         {'hidden_layer_sizes': (50, 50, 50),
                          'activation': 'logistic',
                          'solver': 'lbfgs',
                          'batch_size': 'auto',
                          'learning_rate': 'constant',
                          'learning_rate_init': 0.0001,
                          'max_iter': 50}] * 4,
    'RandomForests': [{'n_estimators': 88,
                       'criterion': 'entropy',
                       'max_depth': 7,
                       'min_samples_split': 6,
                       'class_weight': 'balanced'}] * 4

}
Boosting_CLASSIFIER_PARAMETERS = {  # best for cz
    'NeuralNetwork': [{'hidden_layer_sizes': (50, 50, 50),
                       'activation': 'logistic',
                       'solver': 'lbfgs',
                       'batch_size': 'auto',
                       'learning_rate': 'constant',
                       'learning_rate_init': 0.0001,
                       'max_iter': 50}] * 2,
    'RandomForests': [{'n_estimators': 436,
                       'criterion': 'entropy',
                       'max_depth': 15,
                       'min_samples_split': 5,
                       'class_weight': 'balanced'}] * 2
}
AdaBoost_CLASSIFIER_PARAMETERS = {}
GradientBoosting_CLASSIFIER_PARAMETERS = {}

# CLASSIFIER PARAMETERS CONFIGURATION FOR PARAMETERS SEARCH
LogisticRegression_CLASSIFIER_PARAMETERS_SEARCH = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                                                   'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
                                                   'max_iter': [10, 20, 50, 100, 200, 500],
                                                   'class_weight': 'balanced'}
SupportVectorMachine_CLASSIFIER_PARAMETERS_SEARCH = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                                     'degree': [2, 3, 4, 5],
                                                     'max_iter': [10, 20, 50, 100, 200, 500],
                                                     'class_weight': 'balanced',
                                                     'probability': True}
DecisionTree_CLASSIFIER_PARAMETERS_SEARCH = {'criterion': ['gini', 'entropy'],
                                             'max_depth': [5, 10, 15, 20],
                                             'min_samples_split': [2, 5, 10, 15, 20],
                                             'class_weight': 'balanced'}
RandomForests_CLASSIFIER_PARAMETERS_SEARCH = {'n_estimators': [50, 100, 200, 500],
                                              'criterion': ['gini', 'entropy'],
                                              'max_depth': [5, 10, 20, 50],
                                              'min_samples_split': [2, 5, 10, 20],
                                              'class_weight': 'balanced'}
NeuralNetwork_CLASSIFIER_PARAMETERS_SEARCH = {'hidden_layer_sizes': [(10, 100), (50, 50), (10, 50),
                                                                     (50, 10, 50), (50, 50, 50)],
                                              'activation': ['relu', 'tanh'],
                                              'solver': ['adam', 'lbfgs'],
                                              'batch_size': 'auto',
                                              'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                              'learning_rate_init': [0.01, 0.001, 0.0001],
                                              'max_iter': [50, 100, 200]}

AdaBoost_CLASSIFIER_PARAMETERS_SEARCH = {}
GradientBoosting_CLASSIFIER_PARAMETERS_SEARCH = {}

# BEST CLASSIFIER PARAMETERS SEARCH CONFIGURATION
PERFORMED_PARAMETERS_SEARCH = 'none'  # grid, random, none
RANDOM_SEARCH_ITERATIONS = 100
NUMBER_OF_TRAINING_REPETITIONS_TO_AVERAGE_RESULTS = 1
