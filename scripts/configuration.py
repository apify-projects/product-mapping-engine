# RUNNING CONFIGURATION SETTING
IS_ON_PLATFORM = False
SAVE_PREPROCESSED_PAIRS = True
SAVE_SIMILARITIES = True

# TEXT PREPROCESSING SETTING
COLUMNS_TO_BE_PREPROCESSED = ['name', 'short_description', 'long_description', 'specification_text', 'all_texts']
SIMILARITIES_TO_BE_COMPUTED = ['id', 'brand', 'words', 'cos', 'descriptives', 'units']
SIMILARITIES_TO_IGNORE = []
LOWER_CASE_TEXT = True
TEXT_LEMMATIZER = None

# KEYWORDS DETECTION SETTING
PERFORM_ID_DETECTION = True
PERFORM_COLOR_DETECTION = True
PERFORM_BRAND_DETECTION = True
PERFORM_UNITS_DETECTION = True

# TEXT FILTERING SETTING
MINIMAL_DETECTABLE_ID_LENGTH = 5
NUMBER_OF_TOP_DESCRIPTIVE_WORDS = 50
MAX_WORD_OCCURRENCES_IN_TEXTS = 0.5
MIN_DESCRIPTIVE_WORDS_FOR_MATCH = 0
MIN_PRODUCT_NAME_SIMILARITY_FOR_MATCH = 0
MIN_MATCH_PRICE_RATIO = 0.67
MAX_MATCH_PRICE_RATIO = 1.33

# IMAGE PREPROCESSING SETTING
IMAGE_HASHES_SIZE = 8  # TODO: call from js somehow
IMAGE_RESIZE_WIDTH = 1024
IMAGE_RESIZE_HEIGHT = 1024
IMAGE_FILTERING = True
IMAGE_FILTERING_THRESH = 0.9
HEX_GROUPS_FOR_IMAGE_HASHES = 1

# SIMILARITY COMPUTATIONS SETTING
KEY_SIMILARITY_LIMIT = 0.9
NUMBER_SIMILARITY_DEVIATION = 0.1
STRING_SIMILARITY_DEVIATION = 0.1
UNITS_AND_VALUES_DEVIATION = 0.05
COMPUTE_TEXT_SIMILARITIES = True
COMPUTE_IMAGE_SIMILARITIES = True

# TRAINING CONFIGURATION SETTING
TEST_DATA_PROPORTION = 0.25
PRINCIPAL_COMPONENT_COUNT = 33
PERFORM_PCA_ANALYSIS = True

# EVALUATION_CONFIGURATION SETTING
THRESHOLD_SETTING = True
NUMBER_OF_THRESHES = 100
NUMBER_OF_THRESHES_FOR_AUC = 10
MAX_FP_RATE = 0.05
