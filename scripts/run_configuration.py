# TEXT PREPROCESSING
COLUMNS = ['name', 'short_description', 'long_description', 'specification_text', 'all_texts']
SIMILARITY_NAMES = ['id', 'brand', 'words', 'cos', 'descriptives', 'units']

# KEYWORDS DETECTION
ID_MARK = '#id#'
BRAND_MARK = '#bnd#'
COLOR_MARK = '#col#'
UNIT_MARK = '#unit#'
MARKS = [ID_MARK, BRAND_MARK, COLOR_MARK, UNIT_MARK]
SIZE_UNITS = ['XXS', 'XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']

ID_LEN = 5
TOP_WORDS = 50
FILTER_LIMIT = 0.5

# IMAGE PREPROCESSING
RESIZE_WIDTH = 1024
RESIZE_HEIGHT = 1024

# TRAINING CONFIGURATION
TEST_SIZE = 0.25

# EVALUATION_CONFIGURATION
NUMBER_OF_THRESHS = 100
MAX_FP_RATE = 0.05
