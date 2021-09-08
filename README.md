# Product Mapping Project
Project for automatic mapping among products from different websites.

## Folder *data*
Contains all necessary datasets for product mapping project


### Folder *vocabularies*
Folder containing all the vocabularies and corpus and other stuff for names preprocessing
- folder *corpus*
  - contains all files and scripts for processing CZ corpus from https://www.paracrawl.eu/index.php
  - folder *preprocessed*
    - contains preprocessed corpus - parsed to Czech and English vocabulary and cleaned them
    - contains text corpus to parse dictionaries from it
    - cz_dict.csv and en_dict.csv are the files with unique words parsed from corpus
    - en_dict_cleaned.csv and cz_dict_cleaned.csv contains words from dictionary that were also found in MORPHODITTA
  - folder *source*
    - contains source corpus (not in git as it has 8 GB)
    - en-cs.txt is the corpus file
  - contains all necessary scripts for corpus preprocessing
  - `corpus_preprocessing.py`
    - `run_corpus_preprocessing.py`
    - load corpus file, split Czech and English sentences and create dictionary of unique words for each language 
  - `vocabulary_cleaner.py`
    - `run_vocabulary_cleaner.py`
    - check whether all words in manually created vocabulary from source corpus are existing words using MORPHODITTA
- brands.txt
    - brands of notebooks manually extracted from Alza, Datart and CZC
- colors.txt
  - colors in English and Czech manually extracted from https://www.color-ize.com/color-list.php and https://cs.wikipedia.org/wiki/Seznam_barev

### Folder *wdc_dataset*
Contains the most common dataset of products from webs 
- folder *dataset*
  - contains all source and preprocessed data
  - folder *preprocessed*
    - contains folder with preprocessed data - file with product pairs
    - images prepro: images and file with image hashes and image hashes similarities
    - names prepro: contains file with name similarities
  - folder *source*
    - contains source data
- `wdc_dataset_preparation.py`
  - script for downloading and preprocessing data from [here](http://webdatacommons.org/)


### Folder *extra_dataset*
Contains datasets of products for first POC received from the website Extra
- folder *dataset*
  - contains all source and preprocessed data
  - folder *results*
    - contains folder with predicted data - file with product pairs
  - folder *source*
    - contains two source dataset to find matching pairs: extra and amazon 

## Folder *scripts*
This is the folder containing all the necessary scripts for product mapping

### Folder *classifier_parameters*
- contains configurable parameters for model classificators
- `classifiers.py`
  - contains all possible models to be trained to find amtching products
- `evaluate_classifier.py`
  - library with all necessary functions for evaluation of classifiers
- `run_evaluate_classifier.py`
  - single classifier evaluator
- `run_compare_classifier.py`
  - multiple classifiers evaluator and comparator

  
### Folder *preprocessing* 
#### Folder *images*
This is the folder containing all the necessary scripts for images preprocessing
- folder image_hash_creator
  - contains javascript code to create hashes from images
  - using apify run from cmd call main.js which creates hashes of images in given folder
  - `image_preprocessing.py`
    - serves to preprocess images: crop, resize, object detection 
  
#### Folder *names*
This is the folder containing all the necessary scripts for names preprocessing
- `names_preprocessing.py`
  - preprocess names of products - detects, ids, colors, brands
 
#### Folder *specification*
This is the folder containing all the necessary scripts for specifition preprocessing
- TODO:


### Folder *score_computation* 
This is the folder containing all the necessary scripts for similarity score computation
- folder *images*
  - This folder contains all the necessary scripts for comparison images of the products
  - `compute_names_similarity.py`
    - compares created hashes using bit distance (comparison of % of bits that differs)
    - select the most similar images - they have the highest similarity of hashes
- folder *names*
  - This folder contains all the necessary scripts for comparison names of the products
  - `compute_name_similarity.py`
    - compute similarity between two names according to the ids, brands and cosine similarity of vectors created from all words by tf.idf
- `actor_model_interface.py`
  - interface for website Extra to train model and save it and then load trained model, preprocess their datasets and predict possibly matching pairs
- `dataset_handler.py`
  -  all necessary stuff for operating with datasets (loda, save, prerocess, analyse, etc)
- `run_analyse_dataset.py`
  - preprocess and analyse dataset


## Folder *test*
Contains data and scripts for testing
- folder *data*
  - folder *10_products*
    - manually extracted the 10 pairs of products from different web pages (Alza, CZC, Datart) also with corresponding image sets
    -  folder *source*
      - contains source data  
    - folder *preprocessed*
      - contains preprocessed data
- folder *preprocessing*
  - folder *images*
    - contains scripts for testing functions for images preprocessing 
  - folder *names*
    - contains scripts for testing functions for names preprocessing 
- folder *score_computation*
  - contains scripts for testing functions for compute names and images similarity 
 
## Folder *results*
Contains all results after runs of scripts.
- folder *classifier_visualization*
  - contains images of classifiers
- folder *mismatches*
  - contains mismatched pair
  - folder *data*
    - contains wrongly predicted pairs of every classificator 
  - `run_compare_missclassification.py`
    - loads misclassified pairs from all classificators and analyses them
- folder *models*
  - contains saved model parameters
### ReadMe.md
- File you should definitely read!



