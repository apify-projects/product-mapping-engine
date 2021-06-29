# Product Mapping Project
Project for automatic mapping among products from different websites.

## Folder *data*
Contains all necessary datasets for product mapping

### Folder *preprocessed*
Contains all preprocessed source datasets 
- folder *dataset_name*
  - folder *names*
  - folder *images*
    - folder *cropped*
      - contains cropped images by cropping white surrounding 
    - folder *cropped_masked*
      - contains cropped images using masking and object detection 
    - folder *cropped_resized*
      - contains images after resizing them to unique size
    - folder *hashes*
      - contains hashes generated from images

### Folder *source*
Contains all source datasets
- folder *dataset_name*
  - folder *names*
    - contains dataset with source names of products
  - folder *images*
    - contains dataset with source images of products
      - folder *cropped*
      - folder *cropped_masked*
      - folder *cropped_resized*
      - folder *hashes*

### Folder *vocabularies*
Folder containing all the vocabularies and corpus and other stuff for names preprocessing
- folder *corpus*
  - contains text corpus to parse dictionaries from it
  - contains all files before and after processing CZ corpus from https://www.paracrawl.eu/index.php
  - en-cs.txt is the corpus file
  - cz.csv and en.csv are the files with unique words parsed from corpus
  - en_cleaned.txt and cz_cleaned.txt contains words from dictionary that were also found in MORPHODITTA
  - folder *source*
    - contains source corpus (not in git as it has 8 GB)
  - folder *preprocessed*
    - contains preprocessed corpus - parsed to Czech and English vocabulary and cleaned them 
- brands.txt
    - brands of notebooks manually extracted from Alza, Datart and CZC
- colors.txt
  - colors in English and Czech manually extracted from https://www.color-ize.com/color-list.php and https://cs.wikipedia.org/wiki/Seznam_barev

## Folder *results*
Contains all results after runs of scripts.

### Folder *similarity_score*
- contains all results after computation of names similarity score
- folder *dataset_name*
  - folder *names*
    - contains similarity of names
  - folder *images*
    - contains similarity of images
  - folder *names_and_images*
    - contains similarity of names and images together


## Folder *scripts*
This is the folder containing all the necessary scripts for product mapping

### Folder *preprocessing* 
#### Folder *corpus_stuff*
- contains all necessary scripts for corpus preprocessing
- `corpus_preprocessing.py`
- `run_corpus_preprocessing.py`
  - load corpus file, split Czech and English sentences and create dictionary of unique words for each language 
- `vocabulary_cleaner.py`
- `run_vocabulary_cleaner.py`
  - check whether all words in manually created vocabulary from source corpus are existing words using MORPHODITA

#### Folder *images*
This is the folder containing all the necessary scripts for images preprocessing
- folder image_hash_creator
  - contains javascript code to create hashes from images
  - using apify run from cmd call main.js which creates hashes of images in given folder
- other scripts serve to preprocess images  
  - `run_crop_images_contour_detection.py`
    - crop image using object detection using edges detection and crop image to the biggest object found in it
  - `run_crop_images_simple.py`
    - crop white background around images using finding the edge nonwhite pixels
    - does not work well because there can be small logo in the corner of the image
  - `run_unify_image_size.py`
    - unify all images in the folder to the chosen shape
  
#### Folder *names*
This is the folder containing all the necessary scripts for names preprocessing
- `names_preprocessing.py`
- `run_names_preprocessing.py`
  - preprocess names of products - detects, ids, colors, brands
    
### Folder *score_computation* 
This is the folder containing all the necessary scripts for similarity score computation
- folder *images*
  - This folder contains all the necessary scripts for comparison images of the products
  - `compute_names_similarity.py`
  - `run_compute_names_similarity.py`
    - compares created hashes using bit distance (comparison of % of bits that differs)
    - select the most similar images - they have the highest similarity of hashes
- folder *images_and_names*
  - This folder contains all the necessary scripts for comparison names and images of the products
  - `compute_total_similarity.py`
  - `run_compute_total_similarity.py`
- folder *names*
  - This folder contains all the necessary scripts for comparison names of the products
  - `compute_name_similarity.py`
  - `run_compute_name_similarity.py`
    - compute similarity between two names according to the ids, brands and cosine similarity of vectors created from all words by tf.idf

### ReadMe.md
- File you should definitely read!


# Datasets
Used and created datasets.
## 10_products
- two files with the same 10 products with different names 
- manually extracted from different web pages (Alza, CZC, Datart) also with corresponding image sets
## 100_products
- one file with 20 different products each with 5 different variants of names
- manually extracted from different web pages using Heureka



