# Product Mapping Project
Source codes for automatic mapping of products from different e-shop websites based on comparison of images and texts chracterising each product.

### Folder `data`
This folder contains all necessary data for text preprocessing and keywords detection.
- Folder `vocabularies`
   This folder contains the vocabularies and corpus and other stuff for texts preprocessing
    - Folder `corpus`
      - This folder contains files and scripts for processing CZ corpus from [here](https://www.paracrawl.eu/index.php)
      - Folder `preprocessed`
        - This folder contains preprocessed corpus with split words to Czech and English vocabularies which are used to detect existing words in texts characterising products
        - `cz_dict.csv` and `en_dict.csv` are the files with unique words parsed from corpus
        - `en_dict_cleaned.csv` and `cz_dict_cleaned.csv` contain words from dictionary that were also found in MORPHODITTA
      - NOTE: source corpus file `en-cs.txt` is not in git as it has 8 GB
      - `corpus_preprocessing.py`
        - Load corpus file, split Czech and English sentences and create dictionary of unique words for each language 
      - `vocabulary_cleaner.py`
        - Check whether all words in manually created vocabulary from source corpus are existing words using MORPHODITTA
    - `brands.txt`
      - Contains brands automatically extracted from Alza 
    - `colors.txt`
      - Contains colors in English and Czech manually extracted from [here](https://www.color-ize.com/color-list.php) and [here](https://cs.wikipedia.org/wiki/Seznam_barev)

### Folder `scripts`
This folder contains all the necessary scripts for product mapping
<details>
  <summary>Folder `classifier_handler`</summary>
  <p>
    This folder contains configurable parameters for model classificators
    - `classifiers.py`
      - Contains all possible models to be trained to find matching products
    - `evaluate_classifier.py`
      - Contains all necessary functions for evaluation of classifiers
  </p>
</details>

<details>
  <summary>Folder `dataset_handler`</summary>
  <p>
    This folder contains all the necessary scripts for dataset preprocessing and similarity computations
    -  Folder `dataset_upload`
        This folder contains all the necessary scripts for uploading the dataset from platform and its scraping
        - `combine_initial_datasets.py`
          - TODO: 
        - `pairs_to_url.py`
          - TODO:
        - `repair_excel_with_hyperlinks.py`
          - TODO:
        - `scraped_datasets_to_pairs_datasets.py`
          - TODO:
        - `upload_dataset.py`
          - TODO:
    - Folder `preprocessing` 
        This folder contains all the necessary scripts for images and texts preprocessing
        - Folder `images`
            This folder contains all the necessary scripts for images preprocessing
            - folder image_hash_creator
              - This folder contains javascript code to create hashes from images using apify run from cmd call main.js which creates hashes of images in given folder
              - `image_preprocessing.py`
                - Preprocess images: crop and if necessary resize them, detect objects in them 
        - Folder `texts`
            This folder contains all the necessary scripts for texts preprocessing
            - `text_preprocessing.py`
              - Preprocess all texts characterising the products
            - `keywords_detection.py`
              - Detect all important words in products - ids, colors, brands, units, parameters, unspecified numbers, words out of vocabularies
    - Folder `similarity_computation`
        This folder contains all the necessary scripts for similarity score computation
        - folder *images*
          - This folder contains all the necessary scripts for comparison images of the products
          - `compute_hashes_similarity.py`
            - Compares created hashes using bit distance (comparison of % of bits that differs) and select the most similar images - they have the highest similarity of hashes
        - folder `texts`
          - This folder contains all the necessary scripts for comparison names of the products
          - `compute_texts_similarity.py`
            - Compute similarity between two texts according to the ids, colors, brands, units, parameters, unspecified numbers, words out of vocabularies and selected descriptive and the most characterising words and according to the cosine similarity of vectors created from all words by tf.idf
          - `compute_specifications_similarity.py`
            - Compute similarity between two specifications according to matching the parameters names and comparing their values
        - `dataset_analyser.py`
          -  Contains functions for analysing the datasets
        - `pairs_filtering.py`
          - Contains methods for filtering of nonmatching pairs in two datasets with products according to the price and descripttive words and words similarity
    -`actor_model_interface.py`
        - Interface for product-mapping-trainer and product-mapping-executor that are used to preprocess datasets to train and save the model and to load trained model and predict possibly matching pairs
    - `congfiguration.py`
        - Contains all parameters and other configuration stuff that can be configured for model training and data preprocessing 
  </p>
</details>
  
- 

 
### Folder `results`
This folder contains all results after runs of scripts.
- folder `classifier_visualization`
  - Contains images of visualised classifiers
- folder `mismatches`
  - Contains mismatched pair from datasets
  - folder `data`
    - Contains wrongly predicted pairs of every classificator 
  - `compare_missclassification.py`
    - Loads misclassified pairs from all classificators and analyses them
- folder `models`
  - Contains saved model parameters


### `ReadMe.md`
- File you should definitely read!

